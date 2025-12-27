"""
Vision Agent - System 2: Vision-based Element Finding
Uses Qwen3-VL-4B with SoM annotations to identify elements visually
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import re
from typing import List, Dict, Optional


class VisionAgent:
    """Vision-based element finder using Qwen3-VL-4B (System 2)"""
    
    def __init__(self, model_path: str):
        """
        Initialize Vision Agent with model path
        
        Args:
            model_path: Path to local Qwen3-VL-4B model directory
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        print(f"🤖 VisionAgent initialized")
        print(f"   Model path: {model_path}")
        print(f"   Model will load on first use (lazy loading)")
    
    def _load_model(self):
        """
        Load Qwen3-VL-4B model with 4-bit quantization
        Uses lazy loading to save memory
        """
        if self.model_loaded:
            print("✅ Model already loaded")
            return
        
        print("\n" + "=" * 70)
        print("🔄 Loading Qwen3-VL-4B model with 4-bit quantization...")
        print("   This may take 30-60 seconds on first load...")
        print("=" * 70)
        
        try:
            # 4-bit quantization config for 8GB VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            print("\n📦 Loading model...")
            # Load model with quantization
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            print("📦 Loading processor...")
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model_loaded = True
            
            # Print device info
            device = next(self.model.parameters()).device
            print(f"\n✅ Model loaded successfully!")
            print(f"   Device: {device}")
            print(f"   Quantization: 4-bit")
            
            # Check VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB")
            
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Failed to load model: {e}")
            print("\nTroubleshooting:")
            print("1. Check model path is correct")
            print("2. Ensure model files are downloaded")
            print("3. Install: pip install transformers torch bitsandbytes accelerate")
            raise
    
    def find_element_by_vision(
        self,
        annotated_image_path: str,
        elements: List[Dict],
        task: str
    ) -> int:
        """
        Find element using vision analysis of annotated screenshot
        
        Args:
            annotated_image_path: Path to SoM-annotated screenshot
            elements: List of DOM elements with IDs
            task: Task description (e.g., "click the blue button")
            
        Returns:
            Element ID (0-N) or -1 if not found
        """
        if not elements:
            print("❌ No elements provided")
            return -1
        
        try:
            # Load model if needed (lazy loading)
            if not self.model_loaded:
                self._load_model()
            
            # Create vision prompt
            prompt = self._create_vision_prompt(elements, task)
            
            print(f"🔍 Analyzing screenshot with vision model...")
            print(f"   Task: {task}")
            print(f"   Elements: {len(elements)}")
            
            # Generate response
            response = self._generate_response(annotated_image_path, prompt)
            
            print(f"🤖 Vision Response: {response}")
            
            # Parse response
            element_id = self._parse_response(response, elements)
            
            return element_id
            
        except Exception as e:
            print(f"❌ Vision analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return -1
    
    def _create_vision_prompt(self, elements: List[Dict], task: str) -> str:
        """
        Create prompt for vision model
        
        Args:
            elements: List of elements
            task: User task
            
        Returns:
            Formatted prompt string
        """
        # Build element list (keep it concise)
        element_lines = []
        for el in elements[:20]:  # Limit to first 20 for prompt clarity
            x, y = el['bbox']['x'], el['bbox']['y']
            text = el['text'][:30] if el['text'] else '(no text)'
            element_lines.append(
                f"[{el['id']}] {el['tag']} \"{text}\" at ({x}, {y})"
            )
        
        prompt = f"""You are an intelligent web automation agent. 
You are analyzing a screenshot where interactive elements are marked with red tags (0, 1, 2...) very close to the element.

TASK: {task}

VISIBLE ELEMENTS:
{chr(10).join(element_lines)}

INSTRUCTIONS:
1. Analyze the visual appearance of the candidate elements (color, shape, icon).
2. Compare each element against the task description.
3. Eliminate elements that do not match the visual description.
4. Select the single best match.

FORMAT:
Observation: (Briefly describe the visual style of relevant boxes)
Thought: (Why does one box match better than the others?)
Answer: (The box number only)

Example Response:
Observation: Box 0 is a white input field. Box 2 is a blue rectangular button.
Thought: The task asks for a "blue box". Box 2 is blue. Box 0 is white.
Answer: 2

Your Analysis:"""
        
        return prompt
    
    def _generate_response(self, image_path: str, prompt: str) -> str:
        """
        Generate response using Qwen3-VL model
        
        Args:
            image_path: Path to annotated screenshot
            prompt: Formatted prompt
            
        Returns:
            Model response text
        """
        try:
            # Format messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image_path],
                padding=True,
                return_tensors="pt",
            )
            
            # Move to model's device
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=None,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise
    
    def _parse_response(self, response: str, elements: List[Dict]) -> int:
        """
        Parse vision model response.
        Prioritizes explicit 'Answer: X' and avoids false positive 'NONE's in reasoning text.
        """
        try:
            # 🟢 1. First, look for a strict "Answer: X" pattern
            # This is the most reliable signal.
            match = re.search(r'Answer:\s*(\d+)', response, re.IGNORECASE)
            
            if match:
                element_id = int(match.group(1))
                # Validate range immediately
                if 0 <= element_id < len(elements):
                    return element_id
            
            # 🟡 2. If no strict answer, CHECK FOR "NONE" NOW
            # Only check for NONE if we didn't find a clear answer above.
            # We also strictly look for "Answer: NONE" or just the word NONE by itself to avoid sentence matching
            if re.search(r'Answer:\s*NONE', response, re.IGNORECASE) or response.strip().upper() == 'NONE':
                print("   Model returned NONE")
                return -1
            
            # 🟠 3. Fallback: Last number in text
            # (Only use this if the model forgot to write "Answer:")
            numbers = re.findall(r'\b\d+\b', response)
            if numbers:
                element_id = int(numbers[-1])
                if 0 <= element_id < len(elements):
                    return element_id

            print(f"⚠️  No valid ID found in response: {response}")
            return -1
                
        except Exception as e:
            print(f"⚠️  Failed to parse response: {e}")
            return -1
                
        except Exception as e:
            print(f"⚠️  Failed to parse response: {e}")
            return -1
    
    def unload_model(self):
        """
        Unload model from memory to free VRAM
        """
        if not self.model_loaded:
            print("⚠️  Model not loaded")
            return
        
        try:
            print("\n🔄 Unloading model...")
            
            # Delete model and processor
            del self.model
            del self.processor
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            self.processor = None
            self.model_loaded = False
            
            print("✅ Model unloaded, VRAM freed")
            
        except Exception as e:
            print(f"⚠️  Error unloading model: {e}")


# Interactive test
if __name__ == "__main__":
    from dom_extractor import DOMExtractor
    from som_annotator import SoMAnnotator
    import os
    
    print("=" * 70)
    print("VISION AGENT TEST - System 2")
    print("Vision-based element finding with Qwen3-VL-4B")
    print("=" * 70)
    
    # Configuration
    print("\n📋 Configuration:")
    
    # Ask for model path
    default_path = r"C:\Models\Qwen3-VL-4B"  # Update this default
    model_path = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
    
    if not model_path:
        model_path = default_path
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Warning: Model path does not exist: {model_path}")
        print("   Continuing anyway (will fail if incorrect)...")
    
    # Initialize components
    extractor = DOMExtractor()
    annotator = SoMAnnotator()
    vision_agent = VisionAgent(model_path)
    
    try:
        # User input
        print("\n📋 Test Setup:")
        url = input("Enter URL (e.g., github.com): ").strip()
        
        if not url:
            print("❌ No URL provided")
            exit(1)
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        task = input("Enter VISUAL task (e.g., 'click the blue button', 'click search icon'): ").strip()
        
        if not task:
            print("❌ No task provided")
            exit(1)
        
        print("\n" + "=" * 70)
        
        # Start browser
        print(f"\n🌐 Starting browser and navigating to {url}...")
        if not extractor.start_browser(headless=False):
            exit(1)
        
        if not extractor.navigate(url):
            exit(1)
        
        # Extract elements
        print("\n🔍 Extracting DOM elements...")
        elements = extractor.extract_elements()
        
        if not elements:
            print("❌ No elements found")
            exit(1)
        
        print(f"✅ Found {len(elements)} interactive elements")
        
        # Take screenshot
        print("\n📸 Taking screenshot...")
        screenshot = extractor.take_screenshot()
        
        if not screenshot:
            print("❌ Screenshot failed")
            exit(1)
        
        # Create annotated image
        print("\n🎨 Creating SoM-annotated screenshot...")
        annotated_img, _ = annotator.filter_and_annotate(
            screenshot,
            elements,
            max_elements=15  # Limit for cleaner vision analysis
        )
        
        # Save annotated image
        annotated_path = "test_som_annotated.png"
        with open(annotated_path, 'wb') as f:
            f.write(annotated_img)
        
        print(f"✅ Saved annotated screenshot: {annotated_path}")
        
        # Run System 2 analysis
        print("\n" + "=" * 70)
        print("🤖 SYSTEM 2: Vision Analysis")
        print("=" * 70)
        print(f"Task: '{task}'")
        
        element_id = vision_agent.find_element_by_vision(
            annotated_path,
            elements,
            task
        )
        
        print("\n" + "=" * 70)
        
        if element_id >= 0:
            element = elements[element_id]
            
            print("✅ SYSTEM 2 SUCCESS!")
            print("=" * 70)
            print(f"\n🎯 Vision Model Selected Element [{element_id}]:")
            print(f"   Tag:      {element['tag']}")
            print(f"   Type:     {element['type']}")
            print(f"   Text:     {element['text'][:60] if element['text'] else '(no text)'}")
            print(f"   Position: ({element['bbox']['x']}, {element['bbox']['y']})")
            print(f"   Size:     {element['bbox']['w']}x{element['bbox']['h']}")
            
            if element['html_id']:
                print(f"   HTML ID:  {element['html_id']}")
            
            # Optional click
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
            print("❌ SYSTEM 2 FAILED")
            print("=" * 70)
            print("\n📊 Analysis:")
            print(f"   - Vision model could not identify element for: '{task}'")
            print(f"   - Annotated screenshot: {annotated_path}")
            print("\n💡 Suggestions:")
            print("   - Check if element is visible in screenshot")
            print("   - Try more descriptive visual task")
            print("   - Verify SoM annotations are correct")
        
        print("\n" + "=" * 70)
        input("⏸️  Press Enter to close browser...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Unload model
        print("\n🔄 Cleaning up...")
        vision_agent.unload_model()
        extractor.close()
        print("\n✅ Test complete!")