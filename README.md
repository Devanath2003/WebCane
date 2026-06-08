# WebCane 3.0


WebCane 3.0 is a cutting-edge, agentic web automation framework that utilizes a **ReAct (Reason + Act)** architecture powered by LangGraph. Designed for robust, intelligent web interaction, it combines the reasoning capabilities of state-of-the-art LLMs with hybrid DOM and Vision-based execution.

##  Key Features

- **ReAct Architecture:** An intelligent loop that continuously observes the web page, reasons about the goal, and executes atomic actions.
- **Hybrid Execution Engine:** 
  - **DOM Agent:** Uses text-based matching via Groq (GPT-OSS-120B) for fast, token-efficient interactions.
  - **Vision Agent:** Uses Set-of-Mark (SoM) visual prompting with Gemma 4 31B (via Gemini API) as a powerful fallback.
  - **VLM Fallback:** Employs a locally fine-tuned Qwen2-VL model to predict exact `(x, y)` click coordinates when semantic and vision matching fail.
- **Dynamic Supervisor:** Driven by NVIDIA DeepSeek-V4 Pro or Groq GPT-OSS, it intelligently decides the next best action, handles blockers (popups/captchas), and evaluates task completion.
- **Voice-First Accessibility:** Built-in Speech-to-Text (Groq Whisper) and Text-to-Speech (NVIDIA Riva) to enable completely hands-free web navigation.

##  Architecture

<img width="1134" height="613" alt="image" src="https://github.com/user-attachments/assets/a73d75cb-5320-49f7-8cb8-324f9c69708f" />


1. **Router:** Intelligently extracts the target starting URL from the user's natural language goal.
2. **Observer:** Captures screenshots and DOM state, utilizing Gemma 4 31B to generate a comprehensive, structured description of the page context.
3. **Supervisor:** Analyzes the observer's output, recent execution history, and blockers to output a precise JSON action command.
4. **Executor:** Translates supervisor commands into Playwright actions via a resilient, multi-tiered approach (Direct DOM -> Smart DOM Match -> SoM Vision -> Qwen2-VL).

##  Getting Started

### Prerequisites
- Python 3.10+
- Playwright (`playwright install`)

### Environment Variables
Configure the following in a `.env` file within the `Webcane3/` directory:
```env
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
GROQ_API_KEY3=your_key
NVIDIA_API_KEY=your_key
NVIDIA_API_KEY3=your_key
NVIDIA_API_TTS=your_key
```

### Running the Application
```bash
python run.py
```
Upon execution, you can select between the **DeepSeek-V4 Pro** or **GPT-OSS-120B** supervisor models, choose Voice or Text mode, and opt to run in a strict VLM-only mode.

##  Project Structure
- `Webcane3/main.py`: The core LangGraph ReAct implementation.
- `Webcane3/supervisor.py`: The reasoning engine that determines the next action.
- `Webcane3/observer.py`: The vision-based state analyzer.
- `Webcane3/executor.py`: The hybrid action execution module.
- `Webcane3/browser_controller.py`: The Playwright interface.
- `Webcane3/som_annotator.py`: The Set-of-Mark visual annotation tool.
- `Webcane3/fine_tuned_vlm.py`: Interface for the local Qwen2-VL click prediction model.

---
*Built with LangGraph, Playwright, Gemini, Groq, and NVIDIA NIMs.*
