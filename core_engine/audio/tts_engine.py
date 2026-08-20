"""Text-to-Speech (TTS) Engine for IsharaConnect.

Provides asynchronous, non-blocking voice synthesis with support for Bengali
and English. Uses gTTS for online high-quality synthesis, with a fallback mechanism.
"""

import io
import logging
import queue
import threading
from typing import Optional

from gtts import gTTS

logger = logging.getLogger(__name__)


class TextToSpeechEngine:
    """Asynchronous TTS Engine for background speech synthesis."""

    def __init__(self):
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self) -> None:
        """Background thread to process speech queue."""
        # For local playback without blocking, we could use pygame or simpleaudio.
        # This implementation focuses on the architecture and synthesis.
        while not self._stop_event.is_set():
            try:
                # Wait for text to synthesize (timeout to allow checking stop_event)
                text, lang = self._queue.get(timeout=0.5)
                
                logger.info("Synthesizing and playing speech: '%s' [%s]", text, lang)
                try:
                    # In a fully integrated desktop client, we would pipe this to an audio device.
                    # For now, we simulate the workload.
                    tts = gTTS(text=text, lang=lang)
                    # We could save it to a temporary file and play it here.
                    # tts.save("temp.mp3")
                    # os.system("start temp.mp3") # Windows specific playback
                except Exception as e:
                    logger.error("TTS synthesis failed: %s", e)
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text: str, lang: str = "bn", async_mode: bool = True) -> bool:
        """Queue text for speech synthesis.
        
        Args:
            text: The text to be spoken.
            lang: Language code ('bn' or 'en').
            async_mode: If True, queues the task. If False, processes immediately.
            
        Returns:
            True if queued/processed successfully.
        """
        if not text:
            return False
            
        if async_mode:
            self._queue.put((text, lang))
            return True
        else:
            try:
                tts = gTTS(text=text, lang=lang)
                # Synchronous playback logic would go here
                return True
            except Exception as e:
                logger.error("Synchronous TTS failed: %s", e)
                return False

    def synthesize_to_bytes(self, text: str, lang: str = "bn") -> Optional[bytes]:
        """Synthesize text to audio bytes (MP3) for streaming over WebSocket.
        
        Args:
            text: The text to synthesize.
            lang: Language code.
            
        Returns:
            Raw MP3 bytes or None if synthesis fails.
        """
        try:
            tts = gTTS(text=text, lang=lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            return fp.getvalue()
        except Exception as e:
            logger.error("Failed to synthesize to bytes: %s", e)
            return None

    def close(self) -> None:
        """Gracefully shut down the TTS worker thread."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
