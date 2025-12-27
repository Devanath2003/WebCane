"""
Verification Agent - Action Success Verification
Checks if automation actions succeeded by comparing page states
"""

import ollama
import json
import re
from typing import Dict, Optional


class VerificationAgent:
    """
    Verifies if automation actions succeeded by analyzing state changes
    Uses heuristics for simple cases, LLM for ambiguous verification
    """
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize verification agent
        
        Args:
            model: Ollama model name for verification reasoning
        """
        self.model = model
        
        print(f"✓ Verification Agent initialized with model: {model}")
        
        # Test Ollama connection
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            if not any(model in name for name in model_names):
                print(f"⚠️  Model '{model}' not found")
            else:
                print(f"✅ Connected to Ollama for verification")
        except Exception as e:
            print(f"⚠️  Ollama connection test failed: {e}")
    
    def verify_action(
        self,
        action: Dict,
        before_state: Dict,
        after_state: Dict
    ) -> Dict:
        """
        Verify if an action succeeded by comparing states
        
        Args:
            action: Action that was executed
            before_state: Page state before action
            after_state: Page state after action
            
        Returns:
            Verification result dict
        """
        try:
            action_type = action.get('action', '').lower()
            
            print(f"\n🔍 Verifying: [{action_type}] {action.get('description', '')}")
            
            # Detect changes
            changes = self._detect_changes(before_state, after_state)
            
            # Try heuristic verification first (fast)
            heuristic_result = self._heuristic_verification(action, changes, before_state, after_state)
            
            if heuristic_result is not None:
                # Heuristic gave us a clear answer
                print(f"   ⚡ Fast verification: {'✅ SUCCESS' if heuristic_result['success'] else '❌ FAILED'}")
                return heuristic_result
            
            # Need LLM for ambiguous case
            print(f"   🤖 Using LLM for verification...")
            llm_result = self._verify_with_llm(action, changes)
            
            return llm_result
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'reason': f'Verification error: {str(e)}',
                'retry_recommended': True
            }
    
    def _detect_changes(self, before: Dict, after: Dict) -> Dict:
        """
        Detect what changed between states
        
        Args:
            before: State before action
            after: State after action
            
        Returns:
            Dictionary of detected changes
        """
        changes = {
            'url_changed': False,
            'title_changed': False,
            'new_url': after.get('url', ''),
            'old_url': before.get('url', ''),
            'new_title': after.get('title', ''),
            'old_title': before.get('title', ''),
            'element_count_change': 0,
            'new_elements_types': []
        }
        
        # URL change
        if before.get('url', '') != after.get('url', ''):
            changes['url_changed'] = True
        
        # Title change
        if before.get('title', '') != after.get('title', ''):
            changes['title_changed'] = True
        
        # Element count change
        before_count = before.get('element_count', 0)
        after_count = after.get('element_count', 0)
        changes['element_count_change'] = after_count - before_count
        
        return changes
    
    def _heuristic_verification(
        self,
        action: Dict,
        changes: Dict,
        before_state: Dict,
        after_state: Dict
    ) -> Optional[Dict]:
        """
        Fast heuristic verification for common cases
        Returns None if heuristic can't determine result (needs LLM)
        
        Args:
            action: Action executed
            changes: Detected changes
            before_state: State before
            after_state: State after
            
        Returns:
            Verification result or None if ambiguous
        """
        action_type = action.get('action', '').lower()
        target = action.get('target', '').lower()
        
        # NAVIGATE: Check URL contains target
        if action_type == 'navigate':
            return self._verify_navigation(target, changes['new_url'])
        
        # WAIT: Always succeeds
        if action_type == 'wait':
            return {
                'success': True,
                'confidence': 1.0,
                'reason': 'Wait action completed',
                'retry_recommended': False
            }
        
        # PRESS_KEY: Hard to verify without more context
        if action_type == 'press_key':
            # If URL or title changed, likely successful (e.g., Enter submitted form)
            if changes['url_changed'] or changes['title_changed']:
                return {
                    'success': True,
                    'confidence': 0.8,
                    'reason': f'Page changed after pressing {target}',
                    'retry_recommended': False
                }
            # Otherwise, ambiguous - use LLM
            return None
        
        # FIND_AND_CLICK: Check for page changes
        if action_type == 'find_and_click':
            return self._verify_element_interaction(action, changes)
        
        # TYPE: Hard to verify without checking element value
        if action_type == 'type':
            # If we have element info showing input was updated, great
            # Otherwise it's ambiguous
            return None
        
        # SCROLL: Hard to verify
        if action_type == 'scroll':
            # Assume success if no errors
            return {
                'success': True,
                'confidence': 0.7,
                'reason': 'Scroll action completed',
                'retry_recommended': False
            }
        
        # VERIFY: Can't heuristically verify a verification action
        if action_type == 'verify':
            return None
        
        # Unknown action or ambiguous
        return None
    
    def _verify_navigation(self, target: str, actual_url: str) -> Dict:
        """
        Verify navigation action
        
        Args:
            target: Expected URL or domain
            actual_url: Actual current URL
            
        Returns:
            Verification result
        """
        # Clean up target
        target_clean = target.replace('https://', '').replace('http://', '').strip('/')
        actual_clean = actual_url.replace('https://', '').replace('http://', '')
        
        # Check if target is in actual URL
        success = target_clean.lower() in actual_clean.lower()
        
        return {
            'success': success,
            'confidence': 0.95 if success else 0.9,
            'reason': f"URL is '{actual_url}' {'(contains target)' if success else '(does not contain target)'}",
            'retry_recommended': not success
        }
    
    def _verify_element_interaction(self, action: Dict, changes: Dict) -> Optional[Dict]:
        """
        Verify element interaction (click, etc.)
        
        Args:
            action: Action executed
            changes: Detected changes
            
        Returns:
            Verification result or None if ambiguous
        """
        # Strong indicators of success
        if changes['url_changed']:
            return {
                'success': True,
                'confidence': 0.9,
                'reason': f"URL changed to {changes['new_url']}",
                'retry_recommended': False
            }
        
        if changes['title_changed']:
            return {
                'success': True,
                'confidence': 0.85,
                'reason': f"Page title changed to '{changes['new_title']}'",
                'retry_recommended': False
            }
        
        # New elements appeared
        if changes['element_count_change'] > 5:
            return {
                'success': True,
                'confidence': 0.8,
                'reason': f"{changes['element_count_change']} new elements appeared",
                'retry_recommended': False
            }
        
        # Some new elements (moderate confidence)
        if changes['element_count_change'] > 0:
            return {
                'success': True,
                'confidence': 0.6,
                'reason': f"{changes['element_count_change']} elements changed",
                'retry_recommended': False
            }
        
        # Nothing changed - likely failed
        if changes['element_count_change'] == 0 and not changes['url_changed']:
            return {
                'success': False,
                'confidence': 0.7,
                'reason': 'No visible changes detected after click',
                'retry_recommended': True
            }
        
        # Ambiguous
        return None
    
    def _verify_with_llm(self, action: Dict, changes: Dict) -> Dict:
        """
        Use LLM to verify action success for ambiguous cases
        
        Args:
            action: Action executed
            changes: Detected changes
            
        Returns:
            Verification result
        """
        try:
            prompt = self._create_verification_prompt(action, changes)
            
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.1,
                    'num_predict': 200
                }
            )
            
            llm_response = response['response'].strip()
            
            # Parse LLM response
            result = self._parse_verification_response(llm_response)
            
            return result
            
        except Exception as e:
            print(f"⚠️  LLM verification error: {e}")
            return {
                'success': False,
                'confidence': 0.3,
                'reason': f'LLM verification failed: {str(e)}',
                'retry_recommended': True
            }
    
    def _create_verification_prompt(self, action: Dict, changes: Dict) -> str:
        """
        Create prompt for LLM verification
        
        Args:
            action: Action executed
            changes: Detected changes
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are verifying if a web automation action succeeded.

ACTION TAKEN:
- Type: {action.get('action', 'unknown')}
- Target: {action.get('target', 'unknown')}
- Description: {action.get('description', 'unknown')}

OBSERVED CHANGES:
- URL changed: {changes['url_changed']}
  {f"From: {changes['old_url']}" if changes['url_changed'] else ""}
  {f"To: {changes['new_url']}" if changes['url_changed'] else ""}
- Title changed: {changes['title_changed']}
  {f"To: {changes['new_title']}" if changes['title_changed'] else ""}
- Element count change: {changes['element_count_change']} elements

QUESTION: Did this action likely succeed?

Consider:
- For navigation: URL should change to target
- For clicks: Usually see URL change, title change, or new elements
- For typing: Hard to verify without seeing input value
- For wait: Always succeeds

Respond with ONLY a JSON object (no other text):
{{
  "success": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation",
  "retry_recommended": true or false
}}

Your response:"""
        
        return prompt
    
    def _parse_verification_response(self, response: str) -> Dict:
        """
        Parse LLM verification response
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed verification result
        """
        try:
            # Try direct JSON parse
            try:
                result = json.loads(response)
                return self._validate_verification_result(result)
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return self._validate_verification_result(result)
            
            # Fallback: Try to extract fields
            success = 'true' in response.lower() or 'success' in response.lower()
            
            return {
                'success': success,
                'confidence': 0.5,
                'reason': 'Could not parse LLM response fully',
                'retry_recommended': not success
            }
            
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
            return {
                'success': False,
                'confidence': 0.3,
                'reason': f'Parse error: {str(e)}',
                'retry_recommended': True
            }
    
    def _validate_verification_result(self, result: Dict) -> Dict:
        """
        Validate and clean verification result
        
        Args:
            result: Raw verification result
            
        Returns:
            Validated result
        """
        return {
            'success': bool(result.get('success', False)),
            'confidence': float(result.get('confidence', 0.5)),
            'reason': str(result.get('reason', 'No reason provided')),
            'retry_recommended': bool(result.get('retry_recommended', True))
        }
    
    def verify_navigation(self, expected_url_contains: str, actual_url: str) -> bool:
        """
        Quick navigation verification
        
        Args:
            expected_url_contains: Expected substring in URL
            actual_url: Actual current URL
            
        Returns:
            True if navigation succeeded
        """
        expected_clean = expected_url_contains.lower().replace('https://', '').replace('http://', '')
        actual_clean = actual_url.lower()
        
        return expected_clean in actual_clean
    
    def verify_element_interaction(
        self,
        action: Dict,
        before_count: int,
        after_count: int
    ) -> bool:
        """
        Quick element interaction verification
        
        Args:
            action: Action executed
            before_count: Element count before
            after_count: Element count after
            
        Returns:
            True if interaction likely succeeded
        """
        # For clicks, we expect some change
        if action.get('action') == 'find_and_click':
            return after_count != before_count
        
        # For other interactions, harder to tell
        return True


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("VERIFICATION AGENT TEST")
    print("=" * 70)
    
    agent = VerificationAgent()
    
    # Test Case 1: Successful navigation
    print("\n" + "=" * 70)
    print("TEST 1: Successful Navigation")
    print("=" * 70)
    
    action = {
        'action': 'navigate',
        'target': 'youtube.com',
        'description': 'Navigate to YouTube'
    }
    
    before = {
        'url': 'https://google.com',
        'title': 'Google',
        'element_count': 20
    }
    
    after = {
        'url': 'https://www.youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    # Test Case 2: Failed click (nothing changed)
    print("\n" + "=" * 70)
    print("TEST 2: Failed Click")
    print("=" * 70)
    
    action = {
        'action': 'find_and_click',
        'target': 'login button',
        'description': 'Click login button'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50  # Nothing changed!
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    # Test Case 3: Successful click (URL changed)
    print("\n" + "=" * 70)
    print("TEST 3: Successful Click (URL change)")
    print("=" * 70)
    
    action = {
        'action': 'find_and_click',
        'target': 'search button',
        'description': 'Click search button'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com/results?search_query=cats',
        'title': 'cats - YouTube',
        'element_count': 60
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    # Test Case 4: Wait action (always succeeds)
    print("\n" + "=" * 70)
    print("TEST 4: Wait Action")
    print("=" * 70)
    
    action = {
        'action': 'wait',
        'target': '2',
        'description': 'Wait for page load'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    # Test Case 5: Press Enter (page changed)
    print("\n" + "=" * 70)
    print("TEST 5: Press Enter (successful submission)")
    print("=" * 70)
    
    action = {
        'action': 'press_key',
        'target': 'Enter',
        'description': 'Submit search'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com/results?search_query=test',
        'title': 'test - YouTube',
        'element_count': 55
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    print("\n" + "=" * 70)
    print("✅ All verification tests complete!")
    print("=" * 70)