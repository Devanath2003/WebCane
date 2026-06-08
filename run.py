from Webcane3.main import WebCane
from Webcane3.voice_interface import VoiceInterface


def select_model():
    print("\n" + "-" * 60)
    print("SUPERVISOR MODEL SELECTION")
    print("-" * 60)
    print("  [1] NVIDIA DeepSeek-V4 Pro (Default)")
    print("  [2] Groq OpenAI GPT-OSS-120b - Fast inference")
    print("-" * 60)
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1" or choice == "":
            return "deepseek"
        elif choice == "2":
            return "gpt-oss"
        else:
            print("Please enter 1 or 2")


def select_mode():
    print("\n" + "-" * 60)
    print("ACCESSIBILITY MODE SELECTION")
    print("-" * 60)
    print("  [1] Voice Mode - Speak your goals (for visually impaired)")
    print("  [2] Text Mode  - Type your goals")
    print("-" * 60)
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return "voice"
        elif choice == "2":
            return "text"
        else:
            print("Please enter 1 or 2")


def select_execution_mode():
    print("\n" + "-" * 60)
    print("EXECUTION MODE SELECTION")
    print("-" * 60)
    print("  [1] Hybrid (DOM + Vision Fallback) - Standard")
    print("  [2] VLM Only (Fine-tuned Qwen2-VL) - All clicks via Vision")
    print("-" * 60)
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1" or choice == "":
            return False
        elif choice == "2":
            return True
        else:
            print("Please enter 1 or 2")


def main():
    print("=" * 60)
    print("WEBCANE3 - ReAct Interactive Mode")
    print("=" * 60)
    print("\nInitializing...")

    sup_model_key = select_model()
    vlm_only = select_execution_mode()

    webcane = WebCane(supervisor_model=sup_model_key, vlm_only_mode=vlm_only)
    voice = VoiceInterface()

    mode = select_mode()
    voice_mode = (mode == "voice")

    if voice_mode:
        print("\n[Voice Mode] Speak your goals when prompted.")
        if voice.tts_available:
            voice.announce("Voice mode activated. Say your goal when you hear the beep.")
    else:
        print("\n[Text Mode] Type your goals.")

    print("\n" + "=" * 60)
    print("READY!")
    print("Commands:")
    if voice_mode:
        print("  - Wait for prompt, then speak your goal")
        print("  - Say 'quit' or 'exit' to close")
    else:
        print("  - Type a goal to execute (e.g., 'Go to youtube and search cats')")
        print("  - Type 'quit' or 'exit' to close")
    print("=" * 60)

    try:
        while True:
            print("\n" + "-" * 60)

            if voice_mode:
                if voice.tts_available:
                    voice.speak("What would you like me to do?", blocking=True)
                    import time
                    time.sleep(1.5)
                print("Enter goal (or speak): ", end="", flush=True)
                goal = voice.listen(duration=6)
                if not goal:
                    goal = input("").strip()
            else:
                goal = input("Enter goal: ").strip()

            if not goal or goal in ['.', '...', '']:
                if voice_mode:
                    print("[Voice] No input detected (user silent)")
                    if voice.tts_available:
                        voice.announce("No input detected. Goodbye!")
                    break
                else:
                    print("Please enter a goal.")
                    continue

            goal_lower = goal.lower()
            stop_words = ['quit', 'exit', 'stop', 'close', 'end', 'bye', 'goodbye']
            stop_phrases = ['stop session', 'end session', 'stop the session', 'close session',
                           'stop running', 'stop the program', 'terminate', 'shut down', 'stop']

            should_stop = (
                goal_lower in stop_words or
                any(phrase in goal_lower for phrase in stop_phrases)
            )

            if should_stop:
                if voice_mode and voice.tts_available:
                    voice.announce("Goodbye!")
                break

            if voice_mode and voice.tts_available:
                voice.speak(f"Working on: {goal[:50]}")

            result = webcane.execute_goal(goal, voice=voice if voice_mode else None)

            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print(f"  Success: {result.get('success')}")
            print(f"  Actions taken: {result.get('actions_taken', 0)}")
            print(f"  Successful actions: {result.get('successful_actions', 0)}")
            print(f"  Time: {result.get('elapsed_time', 0):.2f}s")
            print(f"  Final URL: {result.get('final_url', 'N/A')}")
            if result.get('error'):
                print(f"  Error: {result.get('error')}")
            print("=" * 60)

            if voice_mode and voice.tts_available:
                if result.get('success'):
                    voice.speak("Goal completed successfully.", blocking=True)
                    voice.speak("Do you have another action to do?", blocking=True)
                else:
                    voice.speak(f"Goal failed. {result.get('error', '')[:50]}", blocking=True)
                    voice.speak("Do you want to try something else?", blocking=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        voice.cleanup()
        webcane.close()


if __name__ == "__main__":
    main()
