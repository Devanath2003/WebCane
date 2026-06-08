import os
import time
import threading
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Optional

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[Voice] sounddevice/soundfile not installed. Run: pip install sounddevice soundfile")

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import riva.client
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    print("[Voice] nvidia-riva-client not installed. Run: pip install nvidia-riva-client")

from .config import Config


class VoiceInterface:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORDING_DURATION = 5

    RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
    RIVA_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
    RIVA_VOICE = "Magpie-Multilingual.EN-US.Aria"
    RIVA_LANGUAGE = "en-US"
    RIVA_SAMPLE_RATE = 22050

    def __init__(self, api_key: str = None):
        self.groq_client = None
        self.riva_service = None
        self.available = False
        self.tts_available = False
        self.stt_available = False
        self.temp_dir = tempfile.gettempdir()
        self.recording_path = os.path.join(self.temp_dir, "webcane_recording.wav")

        if GROQ_AVAILABLE:
            try:
                stt_key = api_key or Config.GROQ_API_KEY2
                if stt_key:
                    self.groq_client = Groq(api_key=stt_key)
                    self.stt_available = AUDIO_AVAILABLE
            except Exception as e:
                print(f"[Voice] Groq STT init failed: {e}")

        if RIVA_AVAILABLE and PYGAME_AVAILABLE:
            try:
                tts_key = Config.NVIDIA_API_TTS
                if tts_key:
                    metadata = [
                        ("function-id", self.RIVA_FUNCTION_ID),
                        ("authorization", f"Bearer {tts_key}")
                    ]
                    auth = riva.client.Auth(
                        ssl_root_cert=None,
                        ssl_client_cert=None,
                        ssl_client_key=None,
                        use_ssl=True,
                        uri=self.RIVA_SERVER,
                        metadata_args=metadata
                    )
                    self.riva_service = riva.client.SpeechSynthesisService(auth)
                    self.tts_available = True
                    print("[Voice] NVIDIA Riva TTS ready")
                else:
                    print("[Voice] NVIDIA_API_TTS not configured")
            except Exception as e:
                print(f"[Voice] NVIDIA Riva TTS init failed: {e}")

        self.available = self.stt_available or self.tts_available
        if self.available:
            status = []
            if self.stt_available:
                status.append("STT")
            if self.tts_available:
                status.append("TTS-Riva")
            print(f"[Voice] Ready ({', '.join(status)})")
        else:
            print("[Voice] Not available - missing dependencies or API keys")

    def listen(self, duration: float = None) -> Optional[str]:
        if not self.stt_available or not self.groq_client:
            print("[Voice] STT not available")
            return None
        duration = duration or self.RECORDING_DURATION
        try:
            print(f"[Voice] Recording for {duration} seconds... (speak now)")
            recording = sd.rec(
                int(duration * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='int16'
            )
            sd.wait()
            sf.write(self.recording_path, recording, self.SAMPLE_RATE)
            print("[Voice] Recording complete, transcribing...")
            with open(self.recording_path, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(self.recording_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    response_format="verbose_json"
                )
            text = transcription.text.strip()
            print(f"[Voice] Transcribed: \"{text}\"")
            return text
        except Exception as e:
            print(f"[Voice] STT error: {e}")
            return None

    def speak(self, text: str, blocking: bool = False):
        if not self.tts_available:
            print(f"[Voice] (would say): {text}")
            return
        if blocking:
            self._speak_sync(text)
        else:
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()

    def _speak_sync(self, text: str):
        unique_speech_path = os.path.join(self.temp_dir, f"webcane_speech_{uuid.uuid4().hex[:8]}.wav")
        try:
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except:
                    pass
                time.sleep(0.1)
            resp = self.riva_service.synthesize(
                text, self.RIVA_VOICE, self.RIVA_LANGUAGE,
                sample_rate_hz=self.RIVA_SAMPLE_RATE,
                encoding=AudioEncoding.LINEAR_PCM
            )
            with wave.open(unique_speech_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.RIVA_SAMPLE_RATE)
                wf.writeframesraw(resp.audio)
            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(unique_speech_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
                time.sleep(0.1)
                try:
                    os.remove(unique_speech_path)
                except:
                    pass
        except Exception as e:
            error_msg = str(e)
            try:
                error_msg = e.details()
            except:
                pass
            print(f"[Voice] TTS error: {error_msg}")

    def speak_status(self, action: str, target: str = "", success: bool = None):
        if success is None:
            if action == "navigating":
                self.speak(f"Navigating to {target}")
            elif action == "searching":
                self.speak(f"Searching for {target}")
            elif action == "clicking":
                self.speak(f"Clicking on {target}")
            elif action == "typing":
                self.speak(f"Typing {target}")
            else:
                self.speak(f"{action} {target}")
        elif success:
            self.speak(f"Done. {action} successful.")
        else:
            self.speak(f"Failed to {action}.")

    def announce(self, message: str):
        self.speak(message, blocking=True)

    def cleanup(self):
        try:
            if os.path.exists(self.recording_path):
                os.remove(self.recording_path)
        except:
            pass
