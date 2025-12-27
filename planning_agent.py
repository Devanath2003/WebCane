"""
Planning Agent - Task Decomposition
Breaks complex multi-step goals into atomic actions
"""

import ollama
import json
import re
from typing import List, Dict, Optional


class PlanningAgent:
    """
    Decomposes high-level goals into step-by-step action plans
    Uses LLM to understand intent and break down complex tasks
    """
    
    # Valid action types
    VALID_ACTIONS = [
        'navigate',       # Go to a URL
        'find_and_click', # Find and click an element
        'type',           # Type text
        'wait',           # Wait N seconds
        'scroll',         # Scroll page
        'verify',         # Verify something exists
        'press_key'       # Press a keyboard key (Enter, Tab, etc.)
    ]
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize planning agent
        
        Args:
            model: Ollama model name for task decomposition
        """
        self.model = model
        
        print(f"🧠 Planning Agent initialized with model: {model}")
        
        # Test Ollama connection
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            if not any(model in name for name in model_names):
                print(f"⚠️  Model '{model}' not found. Available: {model_names}")
            else:
                print(f"✅ Connected to Ollama")
        except Exception as e:
            print(f"⚠️  Ollama connection test failed: {e}")
    
    def decompose_task(self, goal: str, current_url: str = "about:blank") -> List[Dict]:
        """
        Decompose high-level goal into atomic action steps
        
        Args:
            goal: High-level task description
            current_url: Current page URL for context
            
        Returns:
            List of action dictionaries, or empty list on failure
        """
        if not goal or not goal.strip():
            print("❌ Empty goal provided")
            return []
        
        try:
            # Simplify and clean the goal
            simplified_goal = self.simplify_task(goal)
            
            print(f"\n🧠 Planning: {simplified_goal}")
            print(f"   Current URL: {current_url}")
            
            # Create planning prompt
            prompt = self._create_planning_prompt(simplified_goal, current_url)
            
            # Generate plan with LLM
            print(f"🤖 Generating action plan...")
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.2,  # Lower for more consistent planning
                    'num_predict': 500,  # Allow longer plans
                }
            )
            
            llm_response = response['response'].strip()
            print(f"🤖 LLM Response length: {len(llm_response)} chars")
            
            # Parse the plan
            plan = self._parse_plan(llm_response)
            
            if not plan:
                print("❌ Failed to generate valid plan")
                return []
            
            # Validate plan
            if not self.validate_plan(plan):
                print("⚠️  Plan validation failed, but returning anyway")
            
            print(f"✅ Generated plan with {len(plan)} steps")
            return plan
            
        except Exception as e:
            print(f"❌ Task decomposition failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_planning_prompt(self, goal: str, current_url: str) -> str:
        """
        Create prompt for LLM to generate action plan
        
        Args:
            goal: Simplified goal
            current_url: Current page URL
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a web automation planning agent. Break down high-level goals into step-by-step actions.

CURRENT STATE:
URL: {current_url}

USER GOAL: {goal}

Break this goal into atomic actions. Each action must be ONE of these types:
- navigate: Go to a URL (target = full URL or domain)
- find_and_click: Find and click an element (target = element description like "search button", "login link")
- type: Type text into the currently focused element (target = text to type)
- wait: Wait for page to load (target = number of seconds, usually 1-3)
- scroll: Scroll the page (target = "down", "up", or pixel amount)
- press_key: Press a keyboard key (target = "Enter", "Tab", "Escape", etc.)
- verify: Check if something exists (target = what to verify)

IMPORTANT RULES:
1. Be SPECIFIC in element descriptions (e.g., "search button" not "button")
2. Include "wait" steps after actions that load new content
3. Keep each step atomic (ONE action per step)
4. Number steps sequentially starting from 1
5. For searches: click search box → type query → press Enter OR click search button
6. For navigation: use "navigate" action with just the domain (e.g., "youtube.com")

Return ONLY a valid JSON array with this exact format:
[
  {{"step": 1, "action": "navigate", "target": "youtube.com", "description": "Navigate to YouTube"}},
  {{"step": 2, "action": "find_and_click", "target": "search box", "description": "Click search box"}},
  {{"step": 3, "action": "type", "target": "search query text", "description": "Type search query"}},
  {{"step": 4, "action": "press_key", "target": "Enter", "description": "Submit search"}}
]

EXAMPLES:

Goal: "search for cats on youtube"
[
  {{"step": 1, "action": "navigate", "target": "youtube.com", "description": "Navigate to YouTube"}},
  {{"step": 2, "action": "find_and_click", "target": "search box", "description": "Click search box"}},
  {{"step": 3, "action": "type", "target": "cats", "description": "Type search query"}},
  {{"step": 4, "action": "press_key", "target": "Enter", "description": "Submit search"}},
  {{"step": 5, "action": "wait", "target": "2", "description": "Wait for results"}}
]

Goal: "go to github and click login"
[
  {{"step": 1, "action": "navigate", "target": "github.com", "description": "Navigate to GitHub"}},
  {{"step": 2, "action": "wait", "target": "2", "description": "Wait for page load"}},
  {{"step": 3, "action": "find_and_click", "target": "login button", "description": "Click login button"}}
]

Now plan for this goal: {goal}

Return ONLY the JSON array, nothing else:"""
        
        return prompt
    
    def _parse_plan(self, response: str) -> List[Dict]:
        """
        Parse LLM response to extract action plan
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of action dicts, or empty list on failure
        """
        try:
            # Try to find JSON array in response
            # Look for patterns like [...] or [{...}]
            
            # First, try direct JSON parse
            try:
                plan = json.loads(response)
                if isinstance(plan, list):
                    return self._validate_and_clean_plan(plan)
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from text
            # Look for [...] pattern
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    plan = json.loads(json_str)
                    if isinstance(plan, list):
                        return self._validate_and_clean_plan(plan)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract multiple {...} objects
            object_pattern = r'\{[^}]+\}'
            objects = re.findall(object_pattern, response, re.DOTALL)
            if objects:
                plan = []
                for obj_str in objects:
                    try:
                        obj = json.loads(obj_str)
                        plan.append(obj)
                    except json.JSONDecodeError:
                        continue
                
                if plan:
                    return self._validate_and_clean_plan(plan)
            
            print(f"⚠️  Could not parse JSON from response")
            print(f"Response preview: {response[:200]}...")
            return []
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            return []
    
    def _validate_and_clean_plan(self, plan: List[Dict]) -> List[Dict]:
        """
        Validate and clean parsed plan
        
        Args:
            plan: Raw parsed plan
            
        Returns:
            Cleaned and validated plan
        """
        cleaned = []
        
        for idx, step in enumerate(plan, 1):
            # Ensure required fields
            if not isinstance(step, dict):
                continue
            
            action = step.get('action', '').lower()
            if action not in self.VALID_ACTIONS:
                print(f"⚠️  Invalid action '{action}' in step {idx}, skipping")
                continue
            
            # Build cleaned step
            cleaned_step = {
                'step': step.get('step', idx),
                'action': action,
                'target': str(step.get('target', '')),
                'description': step.get('description', f"{action} action")
            }
            
            cleaned.append(cleaned_step)
        
        # Renumber steps
        for idx, step in enumerate(cleaned, 1):
            step['step'] = idx
        
        return cleaned
    
    def validate_plan(self, plan: List[Dict]) -> bool:
        """
        Validate that plan is well-formed
        
        Args:
            plan: Action plan to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not plan:
            return False
        
        for idx, step in enumerate(plan, 1):
            # Check required fields
            if 'step' not in step or 'action' not in step or 'target' not in step:
                print(f"⚠️  Step {idx} missing required fields")
                return False
            
            # Check action type
            if step['action'] not in self.VALID_ACTIONS:
                print(f"⚠️  Step {idx} has invalid action: {step['action']}")
                return False
            
            # Check step numbering
            if step['step'] != idx:
                print(f"⚠️  Step numbering issue at {idx} (has {step['step']})")
                # Don't fail on this, just warn
        
        return True
    
    def simplify_task(self, complex_task: str) -> str:
        """
        Simplify and clean task description
        
        Args:
            complex_task: Raw user input
            
        Returns:
            Cleaned and simplified task
        """
        # Remove extra whitespace
        task = ' '.join(complex_task.split())
        
        # Convert to lowercase for consistency
        task = task.lower()
        
        # Remove common filler words but keep meaning
        # (Be careful not to remove too much)
        task = task.replace(' please ', ' ')
        task = task.replace(' could you ', '')
        task = task.replace(' can you ', '')
        task = task.replace(' i want to ', '')
        task = task.replace(' i need to ', '')
        
        return task.strip()
    
    def print_plan(self, plan: List[Dict]):
        """
        Print plan in readable format
        
        Args:
            plan: Action plan to display
        """
        if not plan:
            print("❌ No plan to display")
            return
        
        print("\n" + "=" * 70)
        print(f"📋 ACTION PLAN ({len(plan)} steps)")
        print("=" * 70)
        
        for step in plan:
            action_icon = {
                'navigate': '🌐',
                'find_and_click': '🖱️',
                'type': '⌨️',
                'wait': '⏳',
                'scroll': '📜',
                'verify': '✓',
                'press_key': '⌨️'
            }.get(step['action'], '▶️')
            
            print(f"\n{step['step']:2d}. {action_icon} [{step['action'].upper()}]")
            print(f"    Target: {step['target']}")
            print(f"    {step['description']}")
        
        print("\n" + "=" * 70)


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("PLANNING AGENT TEST")
    print("=" * 70)
    
    agent = PlanningAgent()
    
    # Test cases
    test_cases = [
        {
            'goal': "search for Mr Beast on youtube",
            'url': "https://google.com"
        },
        {
            'goal': "go to github and find the login button",
            'url': "about:blank"
        },
        {
            'goal': "search wikipedia for Taj Mahal",
            'url': "https://google.com"
        },
        {
            'goal': "go to youtube, search for cats, and play the first video",
            'url': "about:blank"
        }
    ]
    
    print("\n🧪 Running test cases...\n")
    
    for idx, test in enumerate(test_cases, 1):
        print("\n" + "=" * 70)
        print(f"TEST CASE {idx}")
        print("=" * 70)
        print(f"📋 GOAL: {test['goal']}")
        print(f"🌐 Current URL: {test['url']}")
        print("-" * 70)
        
        plan = agent.decompose_task(test['goal'], test['url'])
        
        if plan:
            agent.print_plan(plan)
            
            # Show validation result
            is_valid = agent.validate_plan(plan)
            print(f"\n{'✅' if is_valid else '⚠️'} Plan validation: {'PASSED' if is_valid else 'ISSUES FOUND'}")
        else:
            print("\n❌ Failed to generate plan")
        
        if idx < len(test_cases):
            input("\nPress Enter for next test case...")
    
    print("\n" + "=" * 70)
    print("✅ All tests complete!")
    print("=" * 70)
    
    # Interactive mode
    print("\n💡 Interactive mode - Enter your own goals")
    print("   Type 'quit' to exit\n")
    
    while True:
        try:
            goal = input("\nEnter goal (or 'quit'): ").strip()
            
            if goal.lower() in ['quit', 'exit', 'q']:
                break
            
            if not goal:
                continue
            
            url = input("Current URL (or press Enter for 'about:blank'): ").strip()
            if not url:
                url = "about:blank"
            
            plan = agent.decompose_task(goal, url)
            
            if plan:
                agent.print_plan(plan)
            else:
                print("\n❌ Failed to generate plan")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            break
    
    print("\n✅ Planning Agent test complete!")