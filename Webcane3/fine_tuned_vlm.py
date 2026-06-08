import os
import re
import torch
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO

QWEN_VLM_AVAILABLE = False
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    QWEN_VLM_AVAILABLE = True
except ImportError:
    pass


class FineTunedVLM:
    BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, adapter_path: str = None):
        if adapter_path is None:
            adapter_path = os.path.join(os.path.dirname(__file__), "archive")
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.loaded = False
        self._load_error = None
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            self._load_error = f"Adapter config not found at {config_path}"
            print(f"[Fine-tuned VLM] Warning: {self._load_error}")

    def is_available(self) -> bool:
        return QWEN_VLM_AVAILABLE and self._load_error is None

    def load(self) -> bool:
        if self.loaded:
            return True
        if not QWEN_VLM_AVAILABLE:
            print("[Fine-tuned VLM] Required packages not available")
            return False
        if self._load_error:
            print(f"[Fine-tuned VLM] Cannot load: {self._load_error}")
            return False
        try:
            print(f"[Fine-tuned VLM] Loading model from {self.adapter_path}...")
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(self.BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
            self.loaded = True
            print("[Fine-tuned VLM] Model loaded successfully")
            return True
        except Exception as e:
            self._load_error = str(e)
            print(f"[Fine-tuned VLM] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_click(self, image_bytes: bytes, instruction: str, image_size: Tuple[int, int] = None) -> Tuple[int, int]:
        if not self.load():
            return (-1, -1)
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
            if image_size:
                width, height = image_size
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0.0)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f"[Fine-tuned VLM] Raw output: {output_text}")
            norm_x, norm_y = self.parse_output(output_text)
            if norm_x < 0 or norm_y < 0:
                print("[Fine-tuned VLM] Failed to parse coordinates")
                return (-1, -1)
            # Model outputs 0-1000 normalized scale; convert to screen pixels
            screen_x = int((norm_x / 1000) * width)
            screen_y = int((norm_y / 1000) * height)
            print(f"[Fine-tuned VLM] Normalized: ({norm_x}, {norm_y}) -> Screen: ({screen_x}, {screen_y})")
            return (screen_x, screen_y)
        except Exception as e:
            print(f"[Fine-tuned VLM] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return (-1, -1)

    def parse_output(self, output: str) -> Tuple[int, int]:
        try:
            match = re.search(r'<point>(\d+)\s+(\d+)</point>', output)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            numbers = re.findall(r'\d+', output)
            if len(numbers) >= 2:
                x, y = int(numbers[0]), int(numbers[1])
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    return (x, y)
            return (-1, -1)
        except Exception as e:
            print(f"[Fine-tuned VLM] Parse error: {e}")
            return (-1, -1)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Fine-tuned VLM] Model unloaded")


_vlm_instance: Optional[FineTunedVLM] = None


def get_fine_tuned_vlm(adapter_path: str = None) -> Optional[FineTunedVLM]:
    global _vlm_instance
    if not QWEN_VLM_AVAILABLE:
        return None
    if _vlm_instance is None:
        _vlm_instance = FineTunedVLM(adapter_path)
    return _vlm_instance


def predict_click_coordinates(image_bytes: bytes, instruction: str, image_size: Tuple[int, int] = None) -> Tuple[int, int]:
    vlm = get_fine_tuned_vlm()
    if vlm is None:
        return (-1, -1)
    return vlm.predict_click(image_bytes, instruction, image_size)
