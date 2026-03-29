"""
Observer agent for WebCane3 ReAct workflow.
Provides action-oriented page analysis using NVIDIA Vision API.
"""

import os
import io
import time
import base64
import json
import requests
from typing import Optional, Dict, List

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .config import Config


class Observer:
    """
    Action-oriented page observer for ReAct workflow.
    Analyzes pages to provide context for the Supervisor's next action decision.
    """
    
    # Debug file paths
    DEBUG_DIR = os.path.dirname(__file__)
    SCREENSHOT_PATH = os.path.join(DEBUG_DIR, "current_screenshot.png")
    OBSERVATION_PATH = os.path.join(DEBUG_DIR, "last_observation.json")
    
    def __init__(self, api_key: str = None):
        """
        Initialize the observer.
        
        Args:
            api_key: NVIDIA API key (uses NVIDIA_API_KEY from config)
        """
        self.api_key = None
        self.model_name = Config.NVIDIA_OBSERVER_MODEL
        self.api_url = Config.NVIDIA_VISION_URL
        self.available = False
        self.last_observation = None
        
        try:
            self.api_key = api_key or Config.NVIDIA_API_KEY
            if not self.api_key:
                print("[Observer] No NVIDIA API key (NVIDIA_API_KEY) provided")
                return
            
            self.available = True
            print(f"[Observer] Ready ({self.model_name}) via NVIDIA API")
            
        except Exception as e:
            print(f"[Observer] Setup failed: {e}")
    
    def _save_screenshot(self, screenshot_bytes: bytes):
        """Save screenshot to file for debugging."""
        try:
            with open(self.SCREENSHOT_PATH, 'wb') as f:
                f.write(screenshot_bytes)
            print(f"[Observer] Screenshot saved to current_screenshot.png")
        except Exception as e:
            print(f"[Observer] Failed to save screenshot: {e}")
    
    def _save_observation(self, observation: Dict):
        """Save observation JSON to file for debugging."""
        try:
            with open(self.OBSERVATION_PATH, 'w', encoding='utf-8') as f:
                json.dump(observation, f, indent=2, ensure_ascii=False)
            print(f"[Observer] Observation saved to last_observation.json")
        except Exception as e:
            print(f"[Observer] Failed to save observation: {e}")
    
    def _compress_image(self, screenshot_bytes: bytes, max_size: int = 1024, quality: int = 60) -> str:
        """
        Compress screenshot to JPEG and resize to fit NVIDIA API context limits.
        Returns base64-encoded JPEG string.
        """
        if PIL_AVAILABLE:
            try:
                img = Image.open(io.BytesIO(screenshot_bytes))
                # Resize if larger than max_size
                w, h = img.size
                if w > max_size or h > max_size:
                    ratio = min(max_size / w, max_size / h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                # Convert to RGB (JPEG doesn't support alpha)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                # Save as JPEG
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=quality)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                print(f"[Observer] Compressed image: {len(screenshot_bytes)} bytes -> {len(buf.getvalue())} bytes (JPEG q={quality})")
                return b64, "jpeg"
            except Exception as e:
                print(f"[Observer] Compression failed, using raw PNG: {e}")
        
        # Fallback: use raw PNG
        return base64.b64encode(screenshot_bytes).decode('utf-8'), "png"
    
    def analyze_for_action(
        self,
        screenshot_bytes: bytes,
        goal: str,
        last_action: Optional[Dict] = None,
        last_action_success: Optional[bool] = None
    ) -> Dict:
        """
        Analyze a screenshot with action-oriented focus.
        
        This is the primary method for the ReAct workflow. It provides
        structured analysis that the Supervisor can use to decide the next action.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            goal: The user's goal
            last_action: The previous action taken (if any)
            last_action_success: Whether the last action succeeded
            
        Returns:
            Dict with:
                - page_state: Current page description
                - blockers: List of detected popups/modals
                - previous_action_result: Analysis of last action outcome
                - key_elements: Notable interactive elements
        """
        # Save screenshot for debugging
        if screenshot_bytes:
            self._save_screenshot(screenshot_bytes)
        
        if not self.available:
            print("[Observer] Not available, returning minimal observation")
            return {
                "page_state": "Unknown page state",
                "blockers": [],
                "previous_action_result": "UNKNOWN",
                "key_elements": []
            }
        
        try:
            # Rate limiting
            time.sleep(Config.API_DELAY)
            
            # Encode and compress screenshot
            b64_image, img_format = self._compress_image(screenshot_bytes)
            
            # Build context about last action
            last_action_context = ""
            if last_action:
                action_type = last_action.get('action', 'unknown')
                target = last_action.get('target', 'unknown')
                query = last_action.get('query', '')
                success_text = "SUCCESS" if last_action_success else "FAILED" if last_action_success is False else "UNKNOWN"
                last_action_context = f"""
PREVIOUS ACTION: {action_type} on "{target}" {f'with query "{query}"' if query else ''}
RESULT: {success_text}
"""
            
            prompt = f"""Analyze this webpage screenshot for a web automation task.

GOAL: {goal}
{last_action_context}

Provide a DETAILED JSON response with these fields:

1. "page_state": A detailed description of the current page including:
   - What website this is
   - What page/section we're on (homepage, search results, product page, etc.)
   - DETAILED CONTENT STATE:
     * Checkbox states: Mention exactly which checkboxes are CHECKED or UNCHECKED
     * Radio buttons: Mention which option is selected
     * Dropdowns: Mention the current selected value
     * Input fields: Mention if they have text in them or are empty
     * Buttons: Mention if any are disabled or loading
     * Errors: Describe any visible error messages or validation alerts

2. "blockers": An array of blocking elements that must be handled first:
   - Popups, modals, overlays
   - Cookie consent banners
   - Login/signup prompts
   - Age verification
   - Any element covering the main content
   Empty array if none visible.

3. "previous_action_result": If there was a previous action, analyze if it worked:
   - "SUCCESS - only do if expected action is there for cases like typing something into a textbox[detailed evidence]" (e.g., "SUCCESS - search results now visible for 'phones', URL changed to /search")
   - "FAILED - [detailed reason]" (e.g., "FAILED - still on login page, error message 'Invalid password' visible")
   - "PARTIAL - [explanation]"
   - "N/A" if no previous action

4. "key_elements": Array of 5-8 notable interactive elements visible:
   - Be specific with descriptions and STATES (e.g., "Submit button [Disabled]", "Type checkbox [Checked]")
   - Include position hints
   - For lists/grids: mention items by name and position
   - Don't be hallucinated and output an empty textbox with 'text' as the value just seeing the goal demands.
   - Example: ["Search bar (empty) at top", "Submit button (enabled) bottom right", "'Terms' checkbox (unchecked)", "Category dropdown (selected: 'All')"]

5. "goal_blockers": Any visible text/messages that would PREVENT achieving the goal:
   - "Product currently unavailable" or "Out of stock" (blocks add-to-cart goals)
   - "Login required" or "Please sign in" (blocks actions requiring authentication)
   - "Item no longer exists" or "Page not found"
   - "Maximum quantity reached" or "Already in cart"
   - ANY status message, warning, or notice that indicates the goal CANNOT be achieved
   - This is DIFFERENT from UI blockers (popups) - these are informational messages
   - Return empty array if no such messages visible

6. "goal_progress": Brief assessment of how close we are to completing the goal:
   - "NOT_STARTED"
   - "IN_PROGRESS"
   - "ALMOST_DONE"
   - "COMPLETE"
   - "BLOCKED" (use this if goal_blockers has items)
   - "IMPOSSIBLE" (use if goal cannot be achieved based on visible information)

Example response:
{{
    "page_state": "Amazon product page for Samsung Galaxy. Shows product title, price Rs 74,999, and 'Currently unavailable' notice. No Add to Cart button visible.",
    "blockers": [],
    "goal_blockers": ["Currently unavailable - We don't know when or if this item will be back in stock"],
    "previous_action_result": "SUCCESS - Navigated to product page",
    "key_elements": ["Product title", "Price display", "Unavailable notice", "Similar products section"],
    "goal_progress": "IMPOSSIBLE"
}}

Respond with ONLY the JSON object, no other text."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{b64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.2,
                "top_p": 0.7,
                "stream": False,
                "chat_template_kwargs": {"thinking": False}
            }
            
            # Retry logic for NVIDIA free tier timeouts
            response = None
            for attempt in range(2):
                try:
                    timeout = 90 if attempt == 0 else 120
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                    break  # Success, exit retry loop
                except requests.exceptions.ReadTimeout:
                    if attempt == 0:
                        print(f"[Observer] NVIDIA API timeout ({timeout}s), retrying in 3s...")
                        time.sleep(3)
                    else:
                        print(f"[Observer] NVIDIA API timeout on retry, giving up")
                        raise
                except requests.exceptions.ConnectionError as e:
                    print(f"[Observer] NVIDIA API connection error: {e}")
                    raise
            
            if response.status_code != 200:
                print(f"[Observer] NVIDIA API error {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
            
            result_data = response.json()
            
            result_text = result_data["choices"][0]["message"]["content"].strip()
            
            # Try to parse JSON
            try:
                # Handle potential markdown code blocks
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                    result_text = result_text.strip()
                
                observation = json.loads(result_text)
                
                # Validate required fields
                if "page_state" not in observation:
                    observation["page_state"] = "Unknown"
                if "blockers" not in observation:
                    observation["blockers"] = []
                if "previous_action_result" not in observation:
                    observation["previous_action_result"] = "N/A"
                if "key_elements" not in observation:
                    observation["key_elements"] = []
                if "goal_progress" not in observation:
                    observation["goal_progress"] = "IN_PROGRESS"
                
            except json.JSONDecodeError:
                # Fallback: create structured response from text
                observation = {
                    "page_state": result_text[:300],
                    "blockers": [],
                    "previous_action_result": "N/A",
                    "key_elements": [],
                    "goal_progress": "UNKNOWN"
                }
            
            self.last_observation = observation
            
            # Save observation to file
            self._save_observation(observation)
            
            # Print detailed observation for debugging
            print("\n" + "=" * 60)
            print("OBSERVER ANALYSIS:")
            print("=" * 60)
            print(f"Page State: {observation.get('page_state', 'Unknown')[:200]}...")
            if observation.get('blockers'):
                print(f"BLOCKERS DETECTED: {observation['blockers']}")
            print(f"Previous Action: {observation.get('previous_action_result', 'N/A')}")
            print(f"Goal Progress: {observation.get('goal_progress', 'UNKNOWN')}")
            print(f"Key Elements ({len(observation.get('key_elements', []))}):")
            for i, elem in enumerate(observation.get('key_elements', [])[:5], 1):
                print(f"  {i}. {elem}")
            print("=" * 60 + "\n")
            
            return observation
            
        except Exception as e:
            print(f"[Observer] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "page_state": "Analysis failed",
                "blockers": [],
                "previous_action_result": "UNKNOWN",
                "key_elements": [],
                "goal_progress": "UNKNOWN"
            }
    
    def describe_page(self, screenshot_bytes: bytes, save_screenshot: bool = True) -> Optional[str]:
        """
        Legacy method: Simple page description.
        Kept for backward compatibility.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            save_screenshot: Whether to save screenshot to file
            
        Returns:
            Description of the page, or None on failure
        """
        if save_screenshot and screenshot_bytes:
            self._save_screenshot(screenshot_bytes)
        
        if not self.available:
            return None
        
        try:
            time.sleep(Config.API_DELAY)
            b64_image, img_format = self._compress_image(screenshot_bytes)
            
            prompt = """Describe this webpage screenshot in detail.
Focus on:
1. What website/page is this?
2. What is the current state (home page, search results, video playing, etc.)?
3. What interactive elements are visible?
4. Any popups, modals, or blockers?

Be comprehensive (3-5 sentences)."""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{b64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 400,
                "temperature": 0.2,
                "top_p": 0.7,
                "stream": False,
                "chat_template_kwargs": {"thinking": False}
            }
            
            # Retry logic for NVIDIA free tier timeouts
            response = None
            for attempt in range(2):
                try:
                    timeout = 90 if attempt == 0 else 120
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                    break
                except requests.exceptions.ReadTimeout:
                    if attempt == 0:
                        print(f"[Observer] describe_page timeout ({timeout}s), retrying...")
                        time.sleep(3)
                    else:
                        raise
            
            if response.status_code != 200:
                print(f"[Observer] NVIDIA API error {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
            
            result_data = response.json()
            
            description = result_data["choices"][0]["message"]["content"].strip()
            return description
            
        except Exception as e:
            print(f"[Observer] describe_page failed: {e}")
            return None
