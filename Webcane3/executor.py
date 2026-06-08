
import time
import os
import base64
import re
import requests
from typing import Dict, List, Optional

from .config import Config
from .browser_controller import BrowserController
from .som_annotator import SoMAnnotator

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from .fine_tuned_vlm import FineTunedVLM, QWEN_VLM_AVAILABLE
    FINE_TUNED_VLM_AVAILABLE = QWEN_VLM_AVAILABLE
except ImportError:
    FINE_TUNED_VLM_AVAILABLE = False

class Executor:

    def __init__(
        self,
        browser: BrowserController,
        groq_api_key: str = None,
        gemini_api_key: str = None,
        vlm_only_mode: bool = False
    ):

        self.browser = browser
        self.annotator = SoMAnnotator()
        self.vlm_only_mode = vlm_only_mode

        if self.vlm_only_mode:
            print("[Executor] INITIALIZED IN VLM-ONLY MODE")

        self.dom_client = None
        self.dom_model = Config.GROQ_DOM_MODEL  # openai/gpt-oss-120b
        if GROQ_AVAILABLE:
            try:
                dom_key = groq_api_key or Config.GROQ_API_KEY
                if dom_key:
                    self.dom_client = Groq(api_key=dom_key)
                    print(f"[Executor] DOM Agent ready ({self.dom_model}) via Groq")
                else:
                    print("[Executor] GROQ_API_KEY not configured for DOM Agent")
            except Exception as e:
                print(f"[Executor] DOM Agent setup failed: {e}")

        self.gemini_client = None
        if GENAI_AVAILABLE:
            try:
                api_key = gemini_api_key or Config.GEMINI_API_KEY
                if api_key:
                    self.gemini_client = genai.Client(api_key=api_key)
                    print("[Executor] Gemini ready for Vision fallback")
            except Exception as e:
                print(f"[Executor] Gemini setup failed: {e}")

        self.vision_client = None
        if GENAI_AVAILABLE:
            try:
                vision_key = Config.GEMINI_API_KEY2
                if vision_key:
                    self.vision_client = genai.Client(api_key=vision_key)
                    print(f"[Executor] Vision Agent ready ({Config.GEMINI_VISION_MODEL}) via Gemini API")
                else:
                    print("[Executor] GEMINI_API_KEY2 not configured for Vision Agent")
            except Exception as e:
                print(f"[Executor] Vision Agent setup failed: {e}")

        self.stats = {
            'dom_success': 0,
            'vision_success': 0,
            'macro_success': 0,
            'failures': 0
        }

        self.last_action_was_click = False

        self.last_vision_reasoning = None

        self.fine_tuned_vlm = None
        if FINE_TUNED_VLM_AVAILABLE:
            adapter_path = os.path.join(
                os.path.dirname(__file__),
                "archive"
            )
            self.fine_tuned_vlm = FineTunedVLM(adapter_path)
            print("[Executor] Fine-tuned VLM ready for fallback")

    def execute_action(self, action: Dict) -> Dict:

        action_type = action.get('action', '').lower()
        target = action.get('target', '')
        query = action.get('query', '')

        print(f"[Executor] {action_type}: {target}" + (f" (query: {query})" if query else ""))

        if action_type == 'search':
            return self.execute_search(target, query)
        elif action_type == 'scroll_find':
            return self.execute_scroll_find(target)
        elif action_type == 'dismiss':
            return self._execute_dismiss(target)

        if action_type == 'navigate':
            self.last_action_was_click = False
            return self._execute_navigate(target)
        elif action_type == 'click' or action_type == 'find_and_click':
            result = self._execute_click(target)
            self.last_action_was_click = result.get('success', False)
            return result
        elif action_type == 'type':
            if not self.last_action_was_click:
                print("[Executor] WARNING: Typing without prior click - attempting to focus first...")
                self._try_focus_input()
            return self._execute_type(target)
        elif action_type == 'press_key':
            self.last_action_was_click = False
            return self._execute_press_key(target)
        elif action_type == 'scroll':
            return self._execute_scroll(target)
        elif action_type == 'strong_scroll':
            return self._execute_strong_scroll(target)
        elif action_type == 'wait':
            return self._execute_wait(target)
        elif action_type == 'go_back':
            return self._execute_go_back()
        else:
            return {'success': False, 'error': f'Unknown action: {action_type}'}

    def execute_search(self, target: str, query: str) -> Dict:

        print(f"[Executor] MACRO: Search for '{query}'")

        elements = self.browser.extract_elements()

        search_input_id = self._find_search_input(elements)

        click_success = False

        if search_input_id >= 0:
            print(f"[Executor] Found search input at element {search_input_id}")
            click_success = self.browser.click_element(search_input_id, elements)
            if click_success:
                print(f"[Executor] Clicked search input successfully")

        if not click_success:
            print(f"[Executor] Direct search failed, trying DOM matching for: {target}")
            click_result = self._execute_click(target)
            click_success = click_result.get('success', False)

            if not click_success:
                alternatives = ["search", "Search", "search input", "search field", "search box"]
                for alt in alternatives:
                    if alt.lower() != target.lower():
                        print(f"[Executor] Trying alternative: {alt}")
                        click_result = self._execute_click(alt)
                        if click_result.get('success'):
                            click_success = True
                            break

        if not click_success:
            self.stats['failures'] += 1
            return {
                'success': False,
                'method': 'search_macro',
                'error': f'Could not find search input: {target}'
            }

        time.sleep(0.5)

        try:
            self.browser.page.keyboard.press("Control+a")
            time.sleep(0.1)
            self.browser.page.keyboard.press("Delete")
            time.sleep(0.2)
        except:
            pass

        print(f"[Executor] Typing query: '{query}'")
        try:
            self.browser.page.keyboard.type(query, delay=50)  # 50ms per char
            print(f"[Executor] Typed query successfully")
        except Exception as e:
            self.stats['failures'] += 1
            return {
                'success': False,
                'method': 'search_macro',
                'error': f'Could not type query: {e}'
            }

        time.sleep(0.8)

        current_url = ""
        try:
            current_url = self.browser.page.url.lower()
        except:
            pass

        google_like_sites = ['google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com']
        is_google_like = any(site in current_url for site in google_like_sites)

        print(f"[Executor] Pressing Enter to submit search...")
        try:
            url_before = self.browser.page.url

            if is_google_like:
                print(f"[Executor] Google-like site: dismissing autocomplete first...")
                self.browser.page.keyboard.press("Escape")
                time.sleep(0.3)

            self.browser.page.keyboard.press("Enter")

            try:
                self.browser.page.wait_for_load_state('networkidle', timeout=8000)
            except:
                time.sleep(2)  # Fallback wait

            url_after = self.browser.page.url
            if url_before != url_after:
                print(f"[Executor] Navigation detected: {url_after[:80]}")
            else:
                print(f"[Executor] Enter failed, trying to click Search button...")
                elements = self.browser.extract_elements()
                for el in elements:
                    text_lower = (el.get('text') or '').lower()
                    tag = el.get('tag', '')
                    if ('search' in text_lower or 'google search' in text_lower) and tag in ['button', 'input']:
                        if 'lucky' not in text_lower:  # Skip "I'm Feeling Lucky"
                            click_success = self.browser.click_element(el['id'], elements)
                            if click_success:
                                print(f"[Executor] Clicked Search button (ID {el['id']})")
                                try:
                                    self.browser.page.wait_for_load_state('networkidle', timeout=8000)
                                except:
                                    time.sleep(2)
                                break

        except Exception as e:
            print(f"[Executor] Enter/navigation error: {e}")

        self.stats['macro_success'] += 1
        time.sleep(Config.STEP_DELAY)

        return {
            'success': True,
            'method': 'search_macro',
            'query': query
        }

    def _find_search_input(self, elements: List[Dict]) -> int:

        search_keywords = ['search', 'query', 'find', 'q', 'keywords']

        for el in elements:
            tag = el.get('tag', '').lower()
            text = (el.get('text', '') or '').lower()
            el_type = (el.get('type', '') or '').lower()
            html_class = (el.get('html_classes', '') or '').lower()
            html_id = (el.get('html_id', '') or '').lower()

            if tag in ['input', 'textarea']:
                if el_type in ['text', 'search', 'button']:
                    combined = f"{text} {html_class} {html_id}"
                    if any(kw in combined for kw in search_keywords):
                        print(f"[Executor] Found search input: id={el['id']}, text='{el.get('text', '')}'")
                        return el['id']

        for el in elements:
            tag = el.get('tag', '').lower()
            text = (el.get('text', '') or '').lower()

            if tag in ['input', 'textarea'] and 'search' in text:
                print(f"[Executor] Found search input (text match): id={el['id']}")
                return el['id']

        print("[Executor] No search input found in elements")
        return -1

    def execute_scroll_find(self, target: str, max_scrolls: int = None) -> Dict:

        max_scrolls = max_scrolls or Config.MAX_SCROLL_ATTEMPTS
        print(f"[Executor] MACRO: Scroll-find '{target}' (max {max_scrolls} scrolls)")

        for attempt in range(max_scrolls):
            print(f"[Executor] Scroll-find attempt {attempt + 1}/{max_scrolls}")

            elements = self.browser.extract_elements()
            if not elements:
                self.browser.scroll('down', 600)
                time.sleep(1.5)
                continue

            element_id = self._find_element_by_vision(elements, target)

            if element_id >= 0:
                if self.browser.click_element(element_id, elements):
                    self.stats['macro_success'] += 1
                    time.sleep(Config.STEP_DELAY)
                    return {
                        'success': True,
                        'method': 'scroll_find_macro',
                        'attempts': attempt + 1,
                        'element_id': element_id
                    }

            if attempt < max_scrolls - 1:
                self.browser.scroll('down', 600)
                time.sleep(1.5)

        print(f"[Executor] Element not found after scrolling down. Scrolling back to top...")
        try:
            self.browser.page.keyboard.press("Home")  # Scroll to top
            time.sleep(1)
        except:
            for _ in range(max_scrolls):
                self.browser.scroll('up', 600)
                time.sleep(0.5)

        self.stats['failures'] += 1
        return {
            'success': False,
            'method': 'scroll_find_macro',
            'error': f'Element not found after {max_scrolls} scroll attempts (scrolled back to top): {target}'
        }

    def _execute_dismiss(self, target: str) -> Dict:

        print(f"[Executor] Dismissing: {target}")

        dismiss_targets = [
            target,
            "close button",
            "close",
            "dismiss",
            "x button",
            "cancel",
            "no thanks",
            "skip",
            "not now"
        ]

        for dismiss_target in dismiss_targets:
            result = self._execute_click(dismiss_target)
            if result.get('success'):
                return {
                    'success': True,
                    'method': 'dismiss',
                    'target_used': dismiss_target
                }

        try:
            self.browser.press_key("Escape")
            time.sleep(0.5)
            return {
                'success': True,
                'method': 'dismiss_escape'
            }
        except:
            pass

        return {
            'success': False,
            'method': 'dismiss',
            'error': f'Could not dismiss: {target}'
        }

    def _try_focus_input(self, target_hint: str = "") -> bool:

        try:
            elements = self.browser.extract_elements()

            input_elements = []
            for el in elements:
                tag = (el.get('tag') or '').lower()
                if tag in ['input', 'textarea']:
                    el_type = (el.get('type') or 'text').lower()
                    if el_type not in ['text', 'search', 'password', 'email', 'tel', 'url', 'number', '']:
                        continue
                    input_elements.append(el)

            if not input_elements:
                print("[Executor] No input elements found on page!")
                return False

            if target_hint:
                hint_lower = target_hint.lower()
                for el in input_elements:
                    text = (el.get('text') or '').lower()
                    placeholder = (el.get('placeholder') or '').lower()
                    html_id = (el.get('html_id') or '').lower()
                    if hint_lower in text or hint_lower in placeholder or hint_lower in html_id:
                        if self.browser.click_element(el['id'], elements):
                            print(f"[Executor] Focused input {el['id']} (matched hint: {target_hint})")
                            time.sleep(0.3)
                            return True

            for el in input_elements:
                if self.browser.click_element(el['id'], elements):
                    print(f"[Executor] Focused first available input {el['id']}")
                    time.sleep(0.3)
                    return True

            try:
                self.browser.page.keyboard.press("Tab")
                time.sleep(0.2)
                print("[Executor] Used Tab to focus next input")
                return True
            except:
                pass

            return False
        except Exception as e:
            print(f"[Executor] Focus input error: {e}")
            return False

    def _soft_validate_element(self, element: Dict, target: str) -> bool:

        if not element:
            return False

        target_lower = target.lower()
        el_text = (element.get('text') or '').lower()
        el_tag = (element.get('tag') or '').lower()
        el_type = (element.get('type') or '').lower()  # Handle None from DOM extraction

        keywords = []
        for word in ['cart', 'add', 'buy', 'button', 'link', 'search', 'submit',
                     'login', 'sign', 'play', 'video', 'image', 'thumbnail']:
            if word in target_lower:
                keywords.append(word)

        if not keywords:
            return True

        combined = f"{el_text} {el_tag} {el_type}"
        for kw in keywords:
            if kw in combined:
                return True

        if 'cart' in target_lower and ('cart' in combined or 'add' in combined):
            return True
        if 'button' in target_lower and el_tag in ['button', 'a', 'input']:
            return True
        if 'link' in target_lower and el_tag == 'a':
            return True

        if not el_text.strip():
            return True

        return True

    def _execute_navigate(self, url: str) -> Dict:

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        success = self.browser.navigate(url)
        time.sleep(Config.STEP_DELAY)

        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Navigation failed'
        }

    def _execute_click(self, target: str) -> Dict:

        visual_keywords = ['thumbnail', 'image', 'picture', 'photo', 'icon',
                          'video with', 'look', 'appear', 'color', 'green',
                          'blue', 'red', 'first', 'second', 'third']
        is_visual_task = any(kw in target.lower() for kw in visual_keywords)

        if is_visual_task:
            print(f"[Executor] Visual task detected - prioritizing Vision agent")

        for scroll_attempt in range(Config.MAX_SCROLL_ATTEMPTS + 1):
            if scroll_attempt > 0:
                print(f"[Executor] Scroll attempt {scroll_attempt}/{Config.MAX_SCROLL_ATTEMPTS}")
                self.browser.scroll('down', 600)
                time.sleep(1.5)

            elements = self.browser.extract_elements()
            if not elements:
                continue

            page_info = self.browser.get_page_info()

            if self.vlm_only_mode and self.fine_tuned_vlm and self.fine_tuned_vlm.is_available():
                print(f"[Executor] VLM-ONLY MODE: Using fine-tuned model for '{target}'")
                screenshot = self.browser.take_screenshot()
                if screenshot:
                    prompt = f"Click {target}"
                    if "click" in target.lower():
                         prompt = target

                    element_id = self._try_fine_tuned_vlm(screenshot, prompt, elements)
                    if element_id >= 0:
                        if self.browser.click_element(element_id, elements):
                            self.stats['vision_success'] += 1
                            time.sleep(Config.STEP_DELAY)
                            return {
                                'success': True,
                                'method': 'vlm_only',
                                'element_id': element_id,
                                'vision_reasoning': f"VLM-ONLY: {prompt}"
                            }
                print("[Executor] VLM-only mode failed to find element")
                return {
                    'success': False,
                    'error': f'VLM-ONLY mode failed to find: {target}',
                    'method': 'vlm_only_failed'
                }

            captcha_keywords = ['captcha', 'robot', 'recaptcha', 'verification', 'human', 'challenge']
            is_captcha = any(kw in target.lower() for kw in captcha_keywords)

            if is_captcha and self.fine_tuned_vlm and self.fine_tuned_vlm.is_available():
                print(f"[Executor] CAPTCHA detected in target '{target}' - invoking fine-tuned VLM directly")

                screenshot = self.browser.take_screenshot()
                if screenshot:
                    captcha_prompts = [
                        "Click on the square box near I'm not a robot",
                        "Click the checkbox for I'm not a robot",
                        f"Click {target}"  # Fallback to original target
                    ]

                    for prompt in captcha_prompts:
                        print(f"[Executor] Trying CAPTCHA prompt: '{prompt}'")
                        element_id = self._try_fine_tuned_vlm(screenshot, prompt, elements)

                        if element_id >= 0:
                            print(f"[Executor] fine-tuned VLM found CAPTCHA element {element_id}")
                            if self.browser.click_element(element_id, elements):
                                self.stats['vision_success'] += 1
                                time.sleep(Config.STEP_DELAY)
                                return {
                                    'success': True,
                                    'method': 'vision_captcha',
                                    'element_id': element_id,
                                    'vision_reasoning': f"CAPTCHA (Fine-tuned VLM): {prompt}"
                                }

                    print("[Executor] Fine-tuned VLM failed for CAPTCHA, falling back to standard pipeline...")

            if is_visual_task:
                print("[Executor] Trying Vision first for visual task...")
                element_id = self._find_element_by_vision(elements, target)

                if element_id >= 0 and element_id < len(elements):
                    clicked_element = elements[element_id] if element_id < len(elements) else None

                    if clicked_element and self._soft_validate_element(clicked_element, target):
                        if self.browser.click_element(element_id, elements):
                            self.stats['vision_success'] += 1
                            time.sleep(Config.STEP_DELAY)
                            return {
                                'success': True,
                                'method': 'vision',
                                'element_id': element_id,
                                'scroll_attempts': scroll_attempt,
                                'vision_reasoning': self.last_vision_reasoning
                            }
                    else:
                        print(f"[Executor] Vision ID {element_id} doesn't match '{target}', checking nearby...")
                        for offset in [-1, 1, -2, 2]:
                            nearby_id = element_id + offset
                            if 0 <= nearby_id < len(elements):
                                nearby_el = elements[nearby_id]
                                if self._soft_validate_element(nearby_el, target):
                                    print(f"[Executor] Found better match at ID {nearby_id}")
                                    if self.browser.click_element(nearby_id, elements):
                                        self.stats['vision_success'] += 1
                                        time.sleep(Config.STEP_DELAY)
                                        return {
                                            'success': True,
                                            'method': 'vision_corrected',
                                            'element_id': nearby_id,
                                            'scroll_attempts': scroll_attempt,
                                            'vision_reasoning': f"Corrected from {element_id} to {nearby_id}"
                                        }
                        print(f"[Executor] No better match found, proceeding with ID {element_id}")
                        if self.browser.click_element(element_id, elements):
                            self.stats['vision_success'] += 1
                            time.sleep(Config.STEP_DELAY)
                            return {
                                'success': True,
                                'method': 'vision',
                                'element_id': element_id,
                                'scroll_attempts': scroll_attempt,
                                'vision_reasoning': self.last_vision_reasoning
                            }
                print("[Executor] Vision failed, trying DOM fallback...")

            element_id = self._find_element_by_text(elements, target, page_info)

            if element_id >= 0:
                if self.browser.click_element(element_id, elements):
                    self.stats['dom_success'] += 1
                    time.sleep(Config.STEP_DELAY)
                    return {
                        'success': True,
                        'method': 'dom',
                        'element_id': element_id,
                        'scroll_attempts': scroll_attempt
                    }

            if not is_visual_task:
                print("[Executor] DOM failed, trying Vision fallback...")
                element_id = self._find_element_by_vision(elements, target)

                if element_id >= 0 and element_id < len(elements):
                    if self.browser.click_element(element_id, elements):
                        self.stats['vision_success'] += 1
                        time.sleep(Config.STEP_DELAY)
                        return {
                            'success': True,
                            'method': 'vision',
                            'element_id': element_id,
                            'scroll_attempts': scroll_attempt,
                            'vision_reasoning': self.last_vision_reasoning  # What Vision saw
                        }

        self.stats['failures'] += 1
        return {
            'success': False,
            'method': 'failed',
            'error': f'Element not found: {target}',
            'scroll_attempts': Config.MAX_SCROLL_ATTEMPTS
        }

    def _find_element_by_text(
        self,
        elements: List[Dict],
        target: str,
        page_info: Dict
    ) -> int:

        if not self.dom_client:
            print("[DOM Agent] NVIDIA DOM client not available, using local fallback")
            return self._find_element_local(elements, target)

        try:
            time.sleep(Config.API_DELAY)

            target_lower = target.lower()
            action_keywords = ['add to cart', 'buy', 'checkout', 'submit', 'button', 'click',
                               'sign in', 'login', 'close', 'dismiss', 'continue', 'proceed',
                               'confirm', 'apply', 'select', 'choose', 'place order']
            is_action_target = any(kw in target_lower for kw in action_keywords)

            elem_list = []
            valid_ids = set()

            if is_action_target:
                action_elements = [el for el in elements
                                   if el.get('tag', '') in ('button', 'a', 'input', 'select')
                                   and el.get('text', '')]
                candidates = action_elements if len(action_elements) >= 5 else elements
                print(f"[DOM Agent] Action target → filtered to {len(candidates)} interactive elements")
            else:
                candidates = elements

            for el in candidates[:80]:
                text = el['text'][:50] if el['text'] else ""
                el_type = el.get('type', el['tag'])
                tag = el['tag']
                elem_list.append(f"[{el['id']}] <{tag}> type={el_type}: \"{text}\"")
                valid_ids.add(el['id'])

            prompt = f

            print("\n" + "=" * 50)
            print("[DOM Agent] Sending query to Groq (GPT-OSS-120B)...")
            print(f"[DOM Agent] Target: \"{target}\"")
            print(f"[DOM Agent] Elements count: {len(elements)}")

            response = self.dom_client.chat.completions.create(
                model=self.dom_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=100,  # Well under 8000 TPM limit
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()

            print(f"[DOM Agent] Raw response: \"{result}\"")

            match = next((int(n) for n in result.split() if n.lstrip('-').isdigit()), -1)

            print(f"[DOM Agent] Parsed ID: {match}")

            if match >= 0 and match in valid_ids:
                for el in elements:
                    if el['id'] == match:
                        print(f"[DOM Agent] Matched element: [{match}] {el['tag']}: \"{el.get('text', '')[:30]}\"")
                        break
                print("=" * 50 + "\n")
                return match
            else:
                print(f"[DOM Agent] Invalid ID {match} (not in valid_ids: {sorted(list(valid_ids)[:10])}...)")
                print("=" * 50 + "\n")
                return -1

        except Exception as e:
            print(f"[DOM Agent] Groq error: {e}")
            import traceback
            traceback.print_exc()
            return self._find_element_local(elements, target)

    def _find_element_local(self, elements: List[Dict], target: str) -> int:

        try:
            target_lower = target.lower()
            keywords = target_lower.split()

            for el in elements:
                text = (el.get('text', '') or '').lower()
                if any(kw in text for kw in keywords):
                    return el['id']

            return -1
        except:
            return -1

    def _try_gemma_vision(self, annotated_bytes: bytes, target: str) -> int:

        if not self.vision_client:
            print("[Executor] Gemma Vision: No vision client configured")
            return -1

        try:
            vision_prompt = f

            print(f"[Executor] Gemma Vision: Calling API ({Config.GEMINI_VISION_MODEL})...")

            response = None
            for attempt in range(2):
                try:
                    response = self.vision_client.models.generate_content(
                        model=Config.GEMINI_VISION_MODEL,
                        contents=[
                            types.Part.from_bytes(data=annotated_bytes, mime_type="image/png"),
                            vision_prompt
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=1000,
                        )
                    )
                    break  # Success
                except Exception as e:
                    if attempt == 0:
                        print(f"[Executor] Gemma Vision: API call failed (attempt 1), retrying in 3s... Error: {e}")
                        time.sleep(3)
                    else:
                        print(f"[Executor] Gemma Vision: API call failed on retry")
                        return -1

            if not response or not response.text:
                print("[Executor] Gemma Vision: Empty response")
                return -1

            result = response.text.strip()

            print("\n[Executor] Gemma Vision Output:")
            print("-" * 50)
            print(result[:300] if len(result) > 300 else result)
            print("-" * 50)

            self.last_vision_reasoning = result[:200]  # Store first 200 chars

            answer_match = re.search(r'ANSWER:\s*(-?\d+)', result, re.IGNORECASE)
            if answer_match:
                return int(answer_match.group(1))

            numbers = re.findall(r'(-?\d+)', result)
            if numbers:
                return int(numbers[-1])

            return -1

        except Exception as e:
            print(f"[Executor] Gemma Vision error: {e}")
            return -1

    def _find_element_by_vision(self, elements: List[Dict], target: str) -> int:

        time.sleep(Config.API_DELAY)

        som_image_path = os.path.join(os.path.dirname(__file__), "som_annotated.png")

        try:
            screenshot = self.browser.take_screenshot()
            if not screenshot:
                print("[Vision Agent] No screenshot available")
                return -1

            annotated_bytes, filtered = self.annotator.annotate(screenshot, elements)

            if not filtered:
                print("[Vision Agent] No elements to annotate")
                return -1

            try:
                with open(som_image_path, 'wb') as f:
                    f.write(annotated_bytes)
                print(f"[Vision Agent] SoM image saved ({len(filtered)} elements annotated)")
            except Exception as e:
                print(f"[Vision Agent] Failed to save SoM image: {e}")

            display_index = self._try_gemma_vision(annotated_bytes, target)
            if display_index >= 0:
                element_id = self.annotator.get_element_id(display_index)
                if element_id >= 0:
                    print(f"[Vision Agent] Gemma 4: display index {display_index} -> element ID {element_id}")
                    return element_id
                else:
                    print(f"[Vision Agent] Gemma 4 returned index {display_index} but mapping failed")

            if self.gemini_client:
                try:
                    gemini_prompt = f

                    response = self.gemini_client.models.generate_content(
                        model=Config.GEMINI_PLANNING_MODEL,
                        contents=[
                            types.Part.from_bytes(data=annotated_bytes, mime_type="image/png"),
                            gemini_prompt
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=700,
                        )
                    )

                    result = response.text.strip()

                    print("\n[Vision Agent] Gemini Output:")
                    print("-" * 40)
                    print(result[:300] if len(result) > 300 else result)
                    print("-" * 40)

                    answer_match = re.search(r'ANSWER:\s*(-?\d+)', result, re.IGNORECASE)
                    if answer_match:
                        display_index = int(answer_match.group(1))
                    else:
                        display_index = next((int(n) for n in result.split() if n.lstrip('-').isdigit()), -1)

                    if display_index >= 0:
                        element_id = self.annotator.get_element_id(display_index)
                        if element_id >= 0:
                            print(f"[Vision Agent] Gemini: display index {display_index} -> element ID {element_id}")
                            self.last_vision_reasoning = f"Vision Agent chose ID {element_id} based on Gemini output: {result[:200]}"
                            return element_id

                except Exception as e:
                    print(f"[Vision Agent] Gemini error: {e}")

            if self.fine_tuned_vlm and self.fine_tuned_vlm.is_available():
                element_id = self._try_fine_tuned_vlm(screenshot, target, elements)
                if element_id >= 0:
                    return element_id

            print("[Vision Agent] All vision agents failed to find element")
            return -1

        except Exception as e:
            print(f"[Vision Agent] Error: {e}")
            import traceback
            traceback.print_exc()
            return -1

    def _try_fine_tuned_vlm(
        self,
        screenshot: bytes,
        target: str,
        elements: List[Dict]
    ) -> int:

        try:
            print("[Fine-tuned VLM] Trying fine-tuned Qwen2-VL...")

            try:
                viewport = self.browser.page.viewport_size
                width = viewport['width'] if viewport else 1920
                height = viewport['height'] if viewport else 1080
            except:
                width, height = 1920, 1080

            instruction = f"Click {target}"

            screen_x, screen_y = self.fine_tuned_vlm.predict_click(
                screenshot,
                instruction,
                image_size=(width, height)
            )

            if screen_x < 0 or screen_y < 0:
                print("[Fine-tuned VLM] Failed to get coordinates")
                return -1

            print(f"[Fine-tuned VLM] Predicted coordinates: ({screen_x}, {screen_y})")

            element_id = self._find_element_at_point(screen_x, screen_y, elements)

            if element_id >= 0:
                print(f"[Fine-tuned VLM] Found element {element_id} at ({screen_x}, {screen_y})")
                self.last_vision_reasoning = f"Fine-tuned VLM predicted click at ({screen_x}, {screen_y})"
                self.stats['vision_success'] += 1
                return element_id

            print("[Fine-tuned VLM] No element at exact point, checking nearby...")
            for offset_x, offset_y in [(0, 0), (-20, 0), (20, 0), (0, -20), (0, 20), (-40, -40), (40, 40)]:
                element_id = self._find_element_at_point(
                    screen_x + offset_x,
                    screen_y + offset_y,
                    elements
                )
                if element_id >= 0:
                    print(f"[Fine-tuned VLM] Found element {element_id} at offset ({offset_x}, {offset_y})")
                    self.last_vision_reasoning = f"Fine-tuned VLM predicted click near ({screen_x}, {screen_y})"
                    self.stats['vision_success'] += 1
                    return element_id

            print("[Fine-tuned VLM] No element found at predicted location")
            return -1

        except Exception as e:
            print(f"[Fine-tuned VLM] Error: {e}")
            import traceback
            traceback.print_exc()
            return -1

    def _find_element_at_point(self, x: int, y: int, elements: List[Dict]) -> int:

        for elem in elements:
            bbox = elem.get('bbox', {})
            if bbox:
                x1 = bbox.get('x', 0)
                y1 = bbox.get('y', 0)
                w = bbox.get('width', 0)
                h = bbox.get('height', 0)

                if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                    return elem.get('id', -1)

        return -1

    def _execute_type(self, text: str) -> Dict:

        success = self.browser.type_text(text)
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Type failed'
        }

    def _execute_press_key(self, key: str) -> Dict:

        success = self.browser.press_key(key)
        time.sleep(Config.STEP_DELAY)
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Key press failed'
        }

    def _execute_scroll(self, target: str) -> Dict:

        parts = target.lower().split()
        direction = parts[0] if parts and parts[0] in ['up', 'down'] else 'down'

        pixels = 600  # default
        if len(parts) > 1 and parts[1].isdigit():
            pixels = int(parts[1])

        success = self.browser.scroll(direction, pixels)
        time.sleep(1)
        return {
            'success': success,
            'method': 'mouse_wheel',
            'pixels': pixels,
            'error': None if success else 'Scroll failed'
        }

    def _execute_strong_scroll(self, target: str) -> Dict:

        direction = target.lower() if target.lower() in ['up', 'down'] else 'down'
        success = self.browser.strong_scroll(direction)
        time.sleep(1.5)
        return {
            'success': success,
            'method': 'strong_scroll',
            'pixels': 1200,
            'error': None if success else 'Strong scroll failed'
        }

    def _execute_wait(self, seconds: str) -> Dict:

        try:
            wait_time = int(seconds) if seconds.isdigit() else 2
            time.sleep(wait_time)
            return {'success': True, 'method': 'direct'}
        except:
            return {'success': False, 'method': 'direct', 'error': 'Invalid wait time'}

    def _execute_go_back(self) -> Dict:

        success = self.browser.go_back()
        time.sleep(Config.STEP_DELAY)
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Go back failed'
        }

    def get_stats(self) -> Dict:

        return self.stats.copy()
