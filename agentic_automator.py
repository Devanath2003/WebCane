
from hybrid_automator import HybridAutomator
from planning_agent import PlanningAgent
from verification_agent import VerificationAgent
from typing import List, Dict, Optional
import time
import json
from datetime import datetime


class AgenticAutomator:
    """
    Complete agentic automation system with:
    - Planning: Break down complex goals
    - Execution: System 1 (DOM) + System 2 (Vision) fallback
    - Verification: Check each step succeeded
    - Recovery: Retry failed steps
    """
    
    def __init__(self, qwen_model_path: str = None):
        """
        Initialize the complete agentic system
        
        Args:
            qwen_model_path: Path to Qwen3-VL model for vision (optional)
        """
        print("=" * 70)
        print("🤖 AGENTIC WEB AUTOMATOR - Initializing")
        print("=" * 70)
        
        print("\n📦 Loading components...")
        
        # Initialize all agents
        self.automator = HybridAutomator(vision_model_path=qwen_model_path)
        self.planner = PlanningAgent()
        self.verifier = VerificationAgent()
        
        # Execution tracking
        self.execution_log = []
        self.memory = {}
        self.start_time = None
        self.total_retries = 0
        
        print("\n✅ Agentic Automator ready!")
        print("=" * 70 + "\n")
    
    def execute_goal(
        self,
        goal: str,
        starting_url: str = "https://google.com",
        max_retries_per_step: int = 2
    ) -> Dict:
        """
        Execute a complex multi-step goal with planning and verification
        
        Args:
            goal: High-level goal description
            starting_url: Starting URL
            max_retries_per_step: Max retries per failed step
            
        Returns:
            Execution result dictionary
        """
        self.start_time = time.time()
        self.execution_log = []
        
        print("\n" + "=" * 70)
        print("🎯 EXECUTING GOAL")
        print("=" * 70)
        print(f"Goal: {goal}")
        print(f"Starting URL: {starting_url}")
        print("=" * 70)
        
        try:
            # PHASE 1: PLANNING
            print("\n" + "🧠 " * 23)
            print("PHASE 1: PLANNING")
            print("🧠 " * 23)
            
            plan = self.planner.decompose_task(goal, starting_url)
            
            if not plan:
                return {
                    'success': False,
                    'error': 'Failed to generate plan',
                    'steps_completed': 0,
                    'total_steps': 0
                }
            
            self.planner.print_plan(plan)
            
            # Start browser
            print("\n🌐 Starting browser session...")
            if not self.automator.start(starting_url, headless=False):
                return {
                    'success': False,
                    'error': 'Failed to start browser',
                    'steps_completed': 0,
                    'total_steps': len(plan)
                }
            
            # PHASE 2: EXECUTION
            print("\n" + "⚡ " * 23)
            print("PHASE 2: EXECUTION WITH VERIFICATION")
            print("⚡ " * 23)
            
            steps_completed = 0
            errors = []
            
            for step_num, action in enumerate(plan, 1):
                print(f"\n{'━' * 70}")
                print(f"📍 STEP {step_num}/{len(plan)}: {action['description']}")
                print(f"{'━' * 70}")
                
                step_result = self._execute_step_with_retry(
                    action,
                    max_retries_per_step
                )
                
                # Log execution
                self.execution_log.append({
                    'step': step_num,
                    'action': action,
                    'result': step_result,
                    'timestamp': time.time() - self.start_time
                })
                
                if not step_result['success']:
                    error_msg = f"Step {step_num} failed: {step_result.get('reason', 'Unknown')}"
                    errors.append(error_msg)
                    print(f"\n❌ {error_msg}")
                    print("⚠️  Cannot proceed with remaining steps")
                    
                    return {
                        'success': False,
                        'steps_completed': steps_completed,
                        'total_steps': len(plan),
                        'failed_at_step': step_num,
                        'errors': errors,
                        'execution_log': self.execution_log,
                        'elapsed_time': time.time() - self.start_time
                    }
                
                steps_completed += 1
                print(f"✅ Step {step_num} completed successfully")
                
                # Small delay between steps
                time.sleep(0.5)
            
            # SUCCESS
            elapsed = time.time() - self.start_time
            print("\n" + "🎉 " * 23)
            print("ALL STEPS COMPLETED SUCCESSFULLY!")
            print("🎉 " * 23)
            print(f"Total time: {elapsed:.2f}s")
            
            return {
                'success': True,
                'steps_completed': steps_completed,
                'total_steps': len(plan),
                'errors': errors,
                'execution_log': self.execution_log,
                'final_state': self.automator.get_current_state(),
                'elapsed_time': elapsed
            }
            
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'steps_completed': len([log for log in self.execution_log if log['result']['success']]),
                'total_steps': len(plan) if plan else 0,
                'execution_log': self.execution_log
            }
    
    def _execute_step_with_retry(
        self,
        action: Dict,
        max_retries: int
    ) -> Dict:
        """
        Execute a single step with retry logic and verification
        
        Args:
            action: Action to execute
            max_retries: Maximum retry attempts
            
        Returns:
            Execution result
        """
        action_type = action['action']
        target = action['target']
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"\n🔄 Retry attempt {attempt}/{max_retries}")
                time.sleep(2)  # Wait before retry
                self.total_retries += 1
            
            # Get state before
            before_state = self.automator.get_current_state()
            
            # Execute action based on type
            execution_success = self._execute_action(action_type, target)
            
            if not execution_success:
                print(f"   ⚠️  Execution failed")
                if attempt < max_retries:
                    continue
                else:
                    return {
                        'success': False,
                        'reason': 'Execution failed after all retries',
                        'attempts': attempt + 1
                    }
            
            # Wait for page to settle
            time.sleep(1.5)
            
            # Get state after
            after_state = self.automator.get_current_state()
            
            # VERIFICATION
            print("   🔍 Verifying action success...")
            verification = self.verifier.verify_action(action, before_state, after_state)
            
            print(f"   {'✅' if verification['success'] else '❌'} Verification: {verification['reason']}")
            print(f"   📊 Confidence: {verification['confidence']:.0%}")
            
            if verification['success'] or verification['confidence'] > 0.7:
                return {
                    'success': True,
                    'verification': verification,
                    'attempts': attempt + 1
                }
            
            # Failed but should retry?
            if not verification['retry_recommended']:
                return {
                    'success': False,
                    'reason': verification['reason'],
                    'verification': verification,
                    'attempts': attempt + 1
                }
        
        # Exhausted all retries
        return {
            'success': False,
            'reason': 'Failed after all retry attempts',
            'attempts': max_retries + 1
        }
    
    def _execute_action(self, action_type: str, target: str) -> bool:
        """
        Execute a single action
        
        Args:
            action_type: Type of action
            target: Action target
            
        Returns:
            True if execution succeeded, False otherwise
        """
        try:
            action_type = action_type.lower()
            
            if action_type == 'navigate':
                print(f"   🌐 Navigating to: {target}")
                # Use existing browser, don't start a new one!
                if not target.startswith('http'):
                    target = 'https://' + target
                return self.automator.extractor.navigate(target)
            
            elif action_type == 'find_and_click':
                print(f"   🖱️  Finding and clicking: {target}")
                # Use hybrid automator to find and click
                result = self.automator.execute_task(f"click {target}")
                return result['success']
            
            elif action_type == 'type':
                print(f"   ⌨️  Typing: {target}")
                # Access the extractor's page object directly
                return self.automator.extractor.type_text(target)
            
            elif action_type == 'press_key':
                print(f"   ⌨️  Pressing key: {target}")
                return self.automator.extractor.press_key(target)
            
            elif action_type == 'wait':
                seconds = int(target)
                print(f"   ⏳ Waiting {seconds} seconds...")
                time.sleep(seconds)
                return True
            
            elif action_type == 'scroll':
                print(f"   📜 Scrolling {target}...")
                direction = target if target in ['up', 'down'] else 'down'
                return self.automator.extractor.scroll_page(direction)
            
            elif action_type == 'verify':
                print(f"   ✓ Verifying: {target}")
                # Just check current state
                state = self.automator.get_current_state()
                return state['element_count'] > 0
            
            else:
                print(f"   ⚠️  Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            print(f"   ❌ Action execution error: {e}")
            return False
    
    def add_memory(self, key: str, value):
        """
        Store information in memory for use between steps
        
        Args:
            key: Memory key
            value: Value to store
        """
        self.memory[key] = value
        print(f"💾 Stored in memory: {key} = {value}")
    
    def get_memory(self, key: str):
        """Get stored memory value"""
        return self.memory.get(key)
    
    def save_execution_log(self, filepath: str = "execution_log.json"):
        """
        Save execution log to file
        
        Args:
            filepath: Output file path
        """
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'execution_log': self.execution_log,
                'memory': self.memory,
                'total_retries': self.total_retries,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"\n💾 Execution log saved to: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Failed to save log: {e}")
    
    def get_execution_summary(self) -> str:
        """
        Get human-readable execution summary
        
        Returns:
            Formatted summary string
        """
        if not self.execution_log:
            return "No execution data available"
        
        successful_steps = len([log for log in self.execution_log if log['result']['success']])
        total_steps = len(self.execution_log)
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        summary = f"""
{'=' * 70}
EXECUTION SUMMARY
{'=' * 70}

Steps Completed: {successful_steps} / {total_steps}
Success Rate: {(successful_steps/total_steps*100):.1f}%
Total Time: {elapsed:.2f}s
Total Retries: {self.total_retries}

Step Details:
"""
        
        for log in self.execution_log:
            step_num = log['step']
            action = log['action']
            result = log['result']
            status = "✅" if result['success'] else "❌"
            
            summary += f"\n  {status} Step {step_num}: {action['description']}"
            if not result['success']:
                summary += f"\n     Reason: {result.get('reason', 'Unknown')}"
        
        summary += f"\n\n{'=' * 70}"
        
        return summary
    
    def close(self):
        """Close all resources"""
        print("\n" + "=" * 70)
        print("🔄 CLOSING AGENTIC AUTOMATOR")
        print("=" * 70)
        
        # Save log
        self.save_execution_log()
        
        # Close automator
        self.automator.close()
        
        print("✅ Agentic Automator closed")


# Test and interactive mode
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("🤖 AGENTIC WEB AUTOMATOR")
    print("Multi-Step Task Execution with Planning & Verification")
    print("=" * 70)
    
    # Configuration
    QWEN_MODEL_PATH = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
    
    # Initialize
    automator = AgenticAutomator(QWEN_MODEL_PATH)
    
    # Check mode
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        print("\n" + "=" * 70)
        print("TEST MODE")
        print("=" * 70)
        
        test_cases = [
            {
                'name': 'Simple Search',
                'goal': 'go to wikipedia and search for Taj Mahal',
                'url': 'https://google.com'
            },
            {
                'name': 'YouTube Search',
                'goal': 'search for Mr Beast on youtube',
                'url': 'https://youtube.com'
            }
        ]
        
        for idx, test in enumerate(test_cases, 1):
            print(f"\n{'#' * 70}")
            print(f"TEST {idx}: {test['name']}")
            print(f"{'#' * 70}")
            
            result = automator.execute_goal(test['goal'], test['url'])
            
            print(f"\n{'=' * 70}")
            print(f"Result: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
            print(f"Steps: {result.get('steps_completed', 0)}/{result.get('total_steps', 0)}")
            print(f"{'=' * 70}")
            
            if idx < len(test_cases):
                input("\nPress Enter for next test...")
        
        print(automator.get_execution_summary())
    
    else:
        # Interactive mode
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("\n💡 Examples:")
        print("   - 'search for cats on youtube'")
        print("   - 'go to github and click login'")
        print("   - 'search wikipedia for taj mahal'")
        print("\n💡 Type 'quit' to exit")
        print("=" * 70)
        
        try:
            while True:
                print("\n" + "-" * 70)
                goal = input("\n🎯 Enter your goal (or 'quit'): ").strip()
                
                if goal.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not goal:
                    continue
                
                url = input("🌐 Starting URL (Enter for google.com): ").strip()
                if not url:
                    url = "https://google.com"
                
                # Execute
                result = automator.execute_goal(goal, url)
                
                # Show result
                print("\n" + "=" * 70)
                if result['success']:
                    print(f"🎉 SUCCESS!")
                    print(f"   Completed {result['steps_completed']}/{result['total_steps']} steps")
                    print(f"   Time: {result['elapsed_time']:.2f}s")
                else:
                    print(f"❌ FAILED")
                    if 'failed_at_step' in result:
                        print(f"   Failed at step {result['failed_at_step']}/{result['total_steps']}")
                    if 'error' in result:
                        print(f"   Error: {result['error']}")
                print("=" * 70)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
    
    # Cleanup
    automator.close()
    print("\n✅ Session complete!")