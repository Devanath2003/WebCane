import json
import re
import time
from typing import Dict, List, Optional

from .config import Config

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[Supervisor] openai package not installed. Run: pip install openai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[Supervisor] groq sdk not installed. Run: pip install groq")


class Supervisor:
    SUPPORTED_ACTIONS = [
        "navigate", "search", "click", "type", "scroll_find",
        "scroll", "press_key", "dismiss", "COMPLETE", "FAILED"
    ]

    def __init__(self, supervisor_model: str = "deepseek"):
        self.provider = supervisor_model
        self.client = None
        self.available = False

        if self.provider == "gpt-oss":
            self.model_name = Config.GROQ_SUPERVISOR_MODEL
            self.api_key = Config.GROQ_API_KEY3
            if not GROQ_AVAILABLE:
                print("[Supervisor] Groq SDK not available")
                return
            if not self.api_key:
                print("[Supervisor] GROQ_API_KEY3 not configured (required for GPT-OSS supervisor)")
                return
            try:
                self.client = Groq(api_key=self.api_key)
                self.available = True
                print(f"[Supervisor] Ready ({self.model_name}) via Groq")
            except Exception as e:
                print(f"[Supervisor] Groq setup failed: {e}")
        else:
            self.model_name = Config.NVIDIA_SUPERVISOR_MODEL
            self.api_key = Config.NVIDIA_API_KEY3
            if not OPENAI_AVAILABLE:
                print("[Supervisor] OpenAI SDK not available")
                return
            if not self.api_key:
                print("[Supervisor] NVIDIA_API_KEY3 not configured")
                return
            try:
                self.client = OpenAI(
                    base_url=Config.NVIDIA_API_URL,
                    api_key=self.api_key
                )
                self.available = True
                print(f"[Supervisor] Ready ({self.model_name}) via NVIDIA OpenAI-compatible API")
            except Exception as e:
                print(f"[Supervisor] NVIDIA setup failed: {e}")

    def decide_next_action(
        self,
        goal: str,
        observation: str,
        blockers: List[str],
        execution_history: List[Dict],
        current_url: str,
        key_elements: List[str] = None
    ) -> Dict:
        if not self.available:
            return self._fallback_decision(goal, observation, blockers)

        try:
            history_text = self._format_history(execution_history[-5:])
            blockers_text = "None" if not blockers else "\n".join(f"- {b}" for b in blockers)
            elements_text = "Unknown" if not key_elements else "\n".join(f"- {e}" for e in key_elements[:5])
            loop_warning = self._detect_loops(execution_history)

            prompt = f"""You are a web automation supervisor. Decide the NEXT SINGLE ACTION to take.

GOAL: {goal}

CURRENT STATE:
- URL: {current_url}
- Page: {observation}

BLOCKING ELEMENTS (handle these FIRST if any):
{blockers_text}

KEY ELEMENTS VISIBLE:
{elements_text}

RECENT ACTIONS (last 5):
{history_text}

{loop_warning}

AVAILABLE ACTIONS:
1. navigate - Go to a URL. Example: {{"action": "navigate", "target": "https://amazon.in"}}
2. search - Find search input, type query, press Enter (MACRO). Example: {{"action": "search", "target": "search bar", "query": "samsung phones"}}
3. click - Click an element by description. Example: {{"action": "click", "target": "first product result"}}
4. type - Type text into already-focused field. Example: {{"action": "type", "target": "samsung phones"}}
5. scroll_find - Scroll to find a visual element. Example: {{"action": "scroll_find", "target": "video with nature thumbnail"}}
6. scroll - Simple scroll. Example: {{"action": "scroll", "target": "down"}}
7. press_key - Press a key. Example: {{"action": "press_key", "target": "Enter"}}
8. dismiss - Dismiss popup/modal. Example: {{"action": "dismiss", "target": "close button"}}
9. COMPLETE - Goal is FULLY achieved.
10. FAILED - Cannot proceed. Example: {{"action": "FAILED", "reason": "Product is unavailable"}}

CRITICAL RULES:
- ONLY do what the user asked for, nothing more!
- If goal_blockers exist in observation, output FAILED immediately with the reason.
- COMPLETE once the goal's expected outcome is visible (search results, clicked item, etc.)
- Do NOT click results unless goal explicitly asks to.
- NEVER navigate away if already on the correct page.
- For search goals: ONLY COMPLETE if URL contains /search or q= parameter.
- If stuck after 3+ failed attempts, try alternative approach or FAILED.

E-COMMERCE:
- On product page (URL has /dp/, /product/): go straight to Add to Cart.
- Use search macro over manual click+type for searching.

Output ONLY a valid JSON object. No explanation."""

            time.sleep(Config.API_DELAY)
            print("[Supervisor] Thinking and deciding next action...")

            reasoning_content = ""
            response_content = ""

            if self.provider == "gpt-oss":
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_completion_tokens=600,
                        top_p=1,
                        reasoning_effort="high",
                        stream=True,
                        stop=None
                    )
                    print(f"[Supervisor] Streaming response via Groq ({self.model_name})...")
                    for chunk in completion:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            print(content, end="", flush=True)
                            response_content += content
                    print()
                except Exception as e:
                    print(f"[Supervisor] Groq inference error: {e}")
            else:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"content": prompt, "role": "user"}],
                        temperature=1,
                        top_p=0.95,
                        max_tokens=16384,
                        extra_body={"chat_template_kwargs": {"thinking": False}},
                        stream=True
                    )
                    print(f"[Supervisor] Streaming response via NVIDIA ({self.model_name})...")
                    for chunk in completion:
                        if not getattr(chunk, "choices", None):
                            continue
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            response_content += chunk.choices[0].delta.content
                except Exception as e:
                    print(f"[Supervisor] NVIDIA inference error: {e}")

            if reasoning_content:
                print("\n[Supervisor] Thinking:")
                print("-" * 40)
                print(reasoning_content[:500] + ("..." if len(reasoning_content) > 500 else ""))
                print("-" * 40)

            if not response_content.strip():
                print("[Supervisor] WARNING: Empty response content!")
                if reasoning_content:
                    decision = self._parse_decision(reasoning_content)
                    if decision.get('action') == 'FAILED':
                        decision = self._fallback_decision(goal, observation, blockers)
                else:
                    decision = self._fallback_decision(goal, observation, blockers)
            else:
                decision = self._parse_decision(response_content)

            if decision.get('action') not in self.SUPPORTED_ACTIONS:
                print(f"[Supervisor] Invalid action: {decision.get('action')}")
                decision = self._normalize_action(decision)

            print("\n" + "-" * 50)
            print("SUPERVISOR DECISION:")
            print(f"  Action: {decision.get('action')}")
            if decision.get('target'):
                print(f"  Target: {decision.get('target')}")
            if decision.get('query'):
                print(f"  Query: {decision.get('query')}")
            if decision.get('reason'):
                print(f"  Reason: {decision.get('reason')}")
            print("-" * 50)

            return decision

        except Exception as e:
            print(f"[Supervisor] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_decision(goal, observation, blockers)

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "No actions taken yet"
        lines = []
        for entry in history:
            action = entry.get('action', 'unknown')
            target = entry.get('target', '')[:30]
            success = "SUCCESS" if entry.get('success') else "FAILED"
            vision_info = ""
            if entry.get('vision_reasoning'):
                vision_info = f"\n  [Vision Agent Feedback: {entry['vision_reasoning']}]"
            elif entry.get('vision_confirmed'):
                vision_info = f"\n  [Vision Agent Confirmed: {entry['vision_confirmed']}]"
            lines.append(f"- {action} '{target}' -> {success}{vision_info}")
        return "\n".join(lines)

    def _detect_loops(self, history: List[Dict]) -> str:
        if len(history) < 2:
            return ""
        recent = history[-3:]
        failed_targets = [h.get('target', '') for h in recent if not h.get('success')]
        if len(failed_targets) >= 2:
            from collections import Counter
            counts = Counter(failed_targets)
            for target, count in counts.items():
                if count >= 2:
                    return f"""
WARNING: Action on "{target}" has FAILED {count} times recently!
You MUST try a DIFFERENT approach:
- Try a different element selector/description
- Scroll first to find the element
- Check if there's a blocker preventing interaction
- Consider if goal is achievable
"""
        return ""

    def _parse_decision(self, content: str) -> Dict:
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        match = re.search(r'\{[^{}]*\}', content)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        print(f"[Supervisor] Could not parse: {content[:100]}")
        return {"action": "FAILED", "reason": "Could not parse supervisor response"}

    def _normalize_action(self, decision: Dict) -> Dict:
        action = decision.get('action', '').lower()
        mappings = {
            'find_and_click': 'click', 'find_click': 'click', 'tap': 'click',
            'enter': 'press_key', 'input': 'type', 'write': 'type',
            'go': 'navigate', 'goto': 'navigate', 'open': 'navigate',
            'close': 'dismiss', 'done': 'COMPLETE', 'complete': 'COMPLETE',
            'success': 'COMPLETE', 'fail': 'FAILED', 'error': 'FAILED', 'wait': 'scroll',
        }
        if action in mappings:
            decision['action'] = mappings[action]
            if decision['action'] == 'press_key' and not decision.get('target'):
                decision['target'] = 'Enter'
        return decision

    def _fallback_decision(self, goal: str, observation: str, blockers: List[str]) -> Dict:
        print("[Supervisor] Using fallback decision logic")
        if blockers:
            return {"action": "dismiss", "target": "popup close button"}
        goal_lower = goal.lower()
        sites = {
            'youtube': 'https://www.youtube.com',
            'amazon': 'https://www.amazon.in',
            'flipkart': 'https://www.flipkart.com',
            'google': 'https://www.google.com'
        }
        for site, url in sites.items():
            if site in goal_lower and url not in observation.lower():
                return {"action": "navigate", "target": url}
        if 'search' in goal_lower:
            match = re.search(r'search\s+(?:for\s+)?(.+?)(?:\s+and|\s+on|\s*$)', goal_lower)
            if match:
                return {"action": "search", "target": "search bar", "query": match.group(1).strip()}
        if 'first' in goal_lower or 'click' in goal_lower:
            return {"action": "click", "target": "first result or product"}
        return {"action": "FAILED", "reason": "Fallback logic could not determine next action"}
