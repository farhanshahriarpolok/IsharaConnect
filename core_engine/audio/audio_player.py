import base64
import threading
import logging
import tempfile
import os
import winsound
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import pygame for better audio support
try:
    import pygame
    from pygame import mixer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not found. Falling back to winsound for audio playback.")

class AudioPlayer:
    """Non-blocking Audio Player for TTS streams and UI chimes."""
    
    def __init__(self):
        self._initialized = False
        self._pygame_failed = False
        
        if PYGAME_AVAILABLE:
            try:
                mixer.init()
                self._initialized = True
            except Exception as e:
                logger.warning(f"pygame.mixer initialization failed: {e}. Falling back to winsound.")
                self._pygame_failed = True
                
    def play_base64(self, base64_str: str) -> bool:
        """Decodes base64 string and routes to play_bytes."""
        try:
            audio_bytes = base64.b64decode(base64_str)
            return self.play_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {e}")
            return False

    def play_bytes(self, audio_bytes: bytes, format: str = "mp3") -> bool:
        """Non-blocking playback of audio bytes."""
        def _play_task():
            try:
                if self._initialized and not self._pygame_failed:
                    # Write to a NamedTemporaryFile to feed pygame.mixer
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as temp_audio:
                        temp_audio.write(audio_bytes)
                        temp_path = temp_audio.name
                        
                    try:
                        mixer.music.load(temp_path)
                        mixer.music.play()
                        while mixer.music.get_busy():
                            time.sleep(0.1)
                    finally:
                        mixer.music.unload()
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                else:
                    # Fallback to winsound (Note: winsound only supports WAV natively)
                    # If format is mp3, this might fail unless we convert it.
                    # Since this is a fallback, we will just try to play it.
                    if format == "wav":
                        winsound.PlaySound(audio_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
                    else:
                        # Write to temp file and try using os.startfile as a last resort fallback on Windows
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as temp_audio:
                            temp_audio.write(audio_bytes)
                            temp_path = temp_audio.name
                        try:
                            os.startfile(temp_path)
                        except Exception as e:
                            logger.error(f"Failed OS playback fallback: {e}")
            except Exception as e:
                logger.error(f"Audio playback failed: {e}")
                
        thread = threading.Thread(target=_play_task, daemon=True)
        thread.start()
        return True

    def play_chime(self, sound_type: str = "success") -> None:
        """Plays a short UI feedback chime."""
        def _chime_task():
            try:
                if sound_type == "success":
                    winsound.Beep(800, 150)
                    winsound.Beep(1000, 200)
                elif sound_type == "notify":
                    winsound.Beep(600, 200)
                else:
                    winsound.Beep(400, 200)
            except Exception as e:
                logger.error(f"Chime playback failed: {e}")
                
        thread = threading.Thread(target=_chime_task, daemon=True)
        thread.start()

# Global singleton
player_instance = AudioPlayer()
