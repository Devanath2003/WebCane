import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(_env_path)

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY2: str = os.getenv("GEMINI_API_KEY2", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_KEY2: str = os.getenv("GROQ_API_KEY2", "")
    GROQ_API_KEY3: str = os.getenv("GROQ_API_KEY3", "")
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_API_KEY2: str = os.getenv("NVIDIA_API_KEY2", "")
    NVIDIA_API_KEY3: str = os.getenv("NVIDIA_API_KEY3", "")
    NVIDIA_API_TTS: str = os.getenv("NVIDIA_API_TTS", "")

    GEMINI_PLANNING_MODEL: str = "gemini-2.5-flash"
    GEMINI_VISION_MODEL: str = "gemma-4-31b-it"

    NVIDIA_SUPERVISOR_MODEL: str = "deepseek-ai/deepseek-v4-pro"
    GROQ_SUPERVISOR_MODEL: str = "openai/gpt-oss-120b"
    NVIDIA_API_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_VISION_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    GROQ_DOM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_OBSERVER_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    NVIDIA_OBSERVER_MODEL: str = "gemma-4-31b-it"

    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    GROQ_TTS_MODEL: str = "canopylabs/orpheus-v1-english"
    GROQ_TTS_VOICE: str = "autumn"

    OLLAMA_MODEL: str = "llama3.2:3b"

    QWEN_MODEL_PATH: str = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"

    API_TIMEOUT: int = 90
    OBSERVATION_TIMEOUT: int = 30
    API_DELAY: float = 0.5

    MAX_LOOP_ITERATIONS: int = 25
    EXECUTION_HISTORY_SIZE: int = 10

    MAX_SCROLL_ATTEMPTS: int = 3
    STEP_DELAY: float = 1.5

    BROWSER_VIEWPORT_WIDTH: int = 1440
    BROWSER_VIEWPORT_HEIGHT: int = 900

    CONTEXT_KEYWORDS: list = [
        "now", "current", "this page", "in results", "on this", "here",
        "from these", "among", "in the", "visible", "showing", "displayed"
    ]

    @classmethod
    def validate(cls) -> dict:
        return {
            "gemini_available": bool(cls.GEMINI_API_KEY),
            "groq_dom_available": bool(cls.GROQ_API_KEY),
            "groq_observer_available": bool(cls.GROQ_API_KEY3),
            "nvidia_vision_available": bool(cls.NVIDIA_API_KEY),
            "nvidia_supervisor_available": bool(cls.NVIDIA_API_KEY3),
            "qwen_available": os.path.exists(cls.QWEN_MODEL_PATH),
        }

    @classmethod
    def print_status(cls):
        status = cls.validate()
        print("=" * 60)
        print("WEBCANE3 CONFIGURATION (ReAct Architecture)")
        print("=" * 60)
        print(f"  Gemini API: {'Available' if status['gemini_available'] else 'Not configured'}")
        print(f"  Groq DOM: {'Available' if status['groq_dom_available'] else 'Not configured'}")
        print(f"  Groq Observer: {'Available' if status['groq_observer_available'] else 'Not configured'}")
        print(f"  NVIDIA Vision: {'Available' if status['nvidia_vision_available'] else 'Not configured'}")
        print(f"  NVIDIA Supervisor: {'Available' if status['nvidia_supervisor_available'] else 'Not configured'}")
        print(f"  Qwen3-VL: {'Available' if status['qwen_available'] else 'Not found'}")
        print("=" * 60)

    @classmethod
    def is_context_continuation(cls, goal: str) -> bool:
        """Check if goal indicates staying on current page."""
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in cls.CONTEXT_KEYWORDS)
