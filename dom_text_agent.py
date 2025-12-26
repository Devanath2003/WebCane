"""
DOM Text Agent - System 1: Pure Text-Based Automation
Uses text LLM to analyze DOM elements and find matches for tasks
"""

import ollama
import re
from typing import List, Dict, Optional


class DOMTextAgent:
    """Text-based DOM automation using local LLM (System 1)"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize Ollama connection and test availability
        
        Args:
            model: Ollama model name (default: llama3.2:3b)
        """
        self.model = model
        
        print(f"🤖 Initializing DOMTextAgent with model: {model}")
        
        # Test Ollama connection
        try:
            # Try to list models to verify Ollama is running
            models = ollama.list()
            
            # Check if our model is available
            model_names = [m['name'] for m in models.get('models', [])]
            if not any(model in name for name in model_names):
                print(f"⚠️  Model '{model}' not found. Available models: {model_names}")
                print(f"   Run: ollama pull {model}")
            else:
                print(f"✅ Connected to Ollama - Model '{model}' ready")
                
        except Exception as e:
            print(f"⚠️  Ollama connection test failed: {e}")
            print("   Make sure Ollama is running: https://ollama.ai/")
    
    def find_element_for_task(
        self, 
        elements: List[Dict], 
        task: str, 
        page_info: Dict
    ) -> int:
        """
        Find the element that best matches the task using text analysis only
        
        Args:
            elements: List of DOM elements from DOMExtractor
            task: User task description (e.g., "click search button")
            page_info: Page information (title, url, viewport)
            
        Returns:
            Element ID (0-N) or -1 if no match found
        """
        if not elements:
            print("❌ No elements to analyze")
            return -1
        
        try:
            # Create prompt for LLM
            prompt = self._create_prompt(elements, task, page_info)
            
            print(f"🤖 Analyzing with {self.model}...")
            
            # Query Ollama
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.1,  # Low temperature for consistency
                    'num_predict': 50,   # Short response expected
                }
            )
            
            llm_response = response['response'].strip()
            print(f"🤖 LLM Response: {llm_response}")
            
            # Parse response to get element ID
            element_id = self._parse_response(llm_response, elements)
            
            return element_id
            
        except Exception as e:
            print(f"❌ Error in find_element_for_task: {e}")
            return -1
    
    def _create_prompt(
        self, 
        elements: List[Dict], 
        task: str, 
        page_info: Dict
    ) -> str:
        """
        Create optimized prompt including HTML identifiers for better matching
        """
        # Limit to first 30 elements for prompt efficiency
        limited_elements = elements[:30]
        
        # Build element list
        element_lines = []
        for el in limited_elements:
            # Format position
            x, y = el['bbox']['x'], el['bbox']['y']
            
            # Format text (truncate if needed)
            text = el['text'][:40] if el['text'] else '(no text)'
            
            # --- UPDATED LINE: Added html_id and html_classes ---
            # We include the ID and Class because they often contain words like 'search' or 'logo'
            html_info = f"id='{el['html_id']}' class='{el['html_classes'][:50]}'"
            
            # Build compact line with the new info
            line = f"[{el['id']}] {el['tag']} \"{text}\" {html_info} type={el['type']} at ({x}, {y})"
            element_lines.append(line)
        
        # Create prompt (unchanged structure, but richer content)
        prompt = f"""PAGE: {page_info.get('title', 'Unknown')}
URL: {page_info.get('url', 'Unknown')}

TASK: {task}

INTERACTIVE ELEMENTS (by text/id/class/position):
{chr(10).join(element_lines)}

Which element ID best matches the task?

RULES:
- Match by text content, HTML id/class names, or likely function
- If an element has no text, look at its id or class name for clues (e.g., 'ytSearchbox' suggests search)
- Return ONLY the number (e.g., "1")
- If no match, return "NONE"

Your answer (number only):"""
        
        return prompt
    
    def _parse_response(self, response: str, elements: List[Dict]) -> int:
        """
        Parse LLM response to extract element ID
        
        Args:
            response: Raw LLM response
            elements: List of elements for validation
            
        Returns:
            Element ID or -1 if invalid/none
        """
        try:
            # Check for explicit "NONE" response
            if 'NONE' in response.upper():
                return -1
            
            # Extract first number from response
            numbers = re.findall(r'\b\d+\b', response)
            
            if not numbers:
                print("⚠️  No number found in LLM response")
                return -1
            
            # Get first number
            element_id = int(numbers[0])
            
            # Validate range
            if 0 <= element_id < len(elements):
                return element_id
            else:
                print(f"⚠️  Element ID {element_id} out of range (0-{len(elements)-1})")
                return -1
                
        except Exception as e:
            print(f"⚠️  Failed to parse response: {e}")
            return -1
    
    def explain_selection(self, element: Dict) -> str:
        """
        Generate human-readable explanation of element selection
        
        Args:
            element: Selected element dictionary
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Element type
        explanation_parts.append(f"Selected {element['tag']} element")
        
        # Text content
        if element['text']:
            explanation_parts.append(f"with text '{element['text'][:50]}'")
        
        # Type detail
        if element['type'] != 'button':
            explanation_parts.append(f"(type: {element['type']})")
        
        # Position
        x, y = element['bbox']['x'], element['bbox']['y']
        
        # Describe position
        position_desc = []
        if y < 100:
            position_desc.append("top")
        elif y > 800:
            position_desc.append("bottom")
        
        if x < 200:
            position_desc.append("left")
        elif x > 1500:
            position_desc.append("right")
        
        if position_desc:
            explanation_parts.append(f"at {'-'.join(position_desc)} of page")
        
        # HTML identifiers
        if element['html_id']:
            explanation_parts.append(f"(id='{element['html_id']}')")
        
        return " ".join(explanation_parts)


# Interactive test
if __name__ == "__main__":
    from dom_extractor import DOMExtractor
    
    print("=" * 70)
    print("DOM TEXT AGENT - System 1 Test")
    print("Pure text-based automation (no vision)")
    print("=" * 70)
    
    # Initialize
    extractor = DOMExtractor()
    agent = DOMTextAgent()
    
    try:
        # User input
        print("\n📋 Setup:")
        url = input("Enter URL (e.g., github.com): ").strip()
        
        if not url:
            print("❌ No URL provided")
            exit(1)
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        task = input("Enter task (e.g., 'click search button', 'click login'): ").strip()
        
        if not task:
            print("❌ No task provided")
            exit(1)
        
        print("\n" + "=" * 70)
        
        # Start browser and navigate
        print(f"\n🌐 Starting browser and navigating to {url}...")
        if not extractor.start_browser(headless=False):
            exit(1)
        
        if not extractor.navigate(url):
            exit(1)
        
        # Extract DOM elements
        print("\n🔍 Extracting DOM elements...")
        elements = extractor.extract_elements()
        page_info = extractor.get_page_info()
        
        if not elements:
            print("❌ No elements found on page")
            exit(1)
        
        print(f"✅ Found {len(elements)} interactive elements")
        print(f"📄 Page: {page_info['title']}")
        
        # Run System 1 analysis
        print("\n" + "=" * 70)
        print(f"🤖 SYSTEM 1 ANALYSIS")
        print(f"Task: '{task}'")
        print(f"Analyzing {min(len(elements), 30)} elements...")
        print("=" * 70)
        
        element_id = agent.find_element_for_task(elements, task, page_info)
        
        print("\n" + "=" * 70)
        
        if element_id >= 0:
            element = elements[element_id]
            
            print("✅ SYSTEM 1 SUCCESS!")
            print("=" * 70)
            print(f"\n🎯 Selected Element [{element_id}]:")
            print(f"   Tag:      {element['tag']}")
            print(f"   Type:     {element['type']}")
            print(f"   Text:     {element['text'][:60] if element['text'] else '(no text)'}")
            print(f"   Position: ({element['bbox']['x']}, {element['bbox']['y']})")
            print(f"   Size:     {element['bbox']['w']}x{element['bbox']['h']}")
            
            if element['html_id']:
                print(f"   HTML ID:  {element['html_id']}")
            
            if element['html_classes']:
                print(f"   Classes:  {element['html_classes'][:60]}")
            
            # Explanation
            print(f"\n💡 Explanation:")
            print(f"   {agent.explain_selection(element)}")
            
            # Optional: Actually click
            print("\n" + "=" * 70)
            click_choice = input("🖱️  Click this element? (y/n): ").strip().lower()
            
            if click_choice == 'y':
                print("\n🖱️  Clicking element...")
                if extractor.click_by_id(element_id, elements):
                    print("✅ Click executed successfully!")
                    print("   Waiting 2 seconds to see result...")
                    import time
                    time.sleep(2)
                else:
                    print("❌ Click failed")
            
        else:
            print("❌ SYSTEM 1 FAILED")
            print("=" * 70)
            print("\n📊 Analysis:")
            print(f"   - No matching element found for task: '{task}'")
            print(f"   - Total elements analyzed: {min(len(elements), 30)}")
            print("\n💡 Suggestions:")
            print("   - Try rephrasing the task")
            print("   - Check if element exists on page")
            print("   - Use more specific terms")
            print("\n🔄 Next Step:")
            print("   → Would trigger SYSTEM 2 (Vision-based fallback)")
        
        print("\n" + "=" * 70)
        input("⏸️  Press Enter to close browser...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()
        print("\n✅ Test complete!")