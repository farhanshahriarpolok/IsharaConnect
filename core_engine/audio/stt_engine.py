"""Speech-to-Text (STT) Engine for IsharaConnect.

Provides transcription interface for hearing participants.
Supports Bengali and English voice inputs.
"""

import logging
from typing import Dict, Union

import speech_recognition as sr

logger = logging.getLogger(__name__)


class SpeechToTextEngine:
    """Wrapper around SpeechRecognition for real-time transcription."""

    def __init__(self, energy_threshold: int = 300):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        # dynamic_energy_threshold allows adjusting to ambient noise
        self.recognizer.dynamic_energy_threshold = True

    def listen_and_transcribe(self, lang: str = "bn-BD", timeout: int = 5) -> Dict[str, Union[str, bool, float]]:
        """Listen to the microphone and transcribe speech.
        
        Args:
            lang: Language code ('bn-BD' or 'en-US').
            timeout: Maximum seconds to wait for speech to start.
            
        Returns:
            Dictionary with transcription event data.
        """
        transcript = ""
        is_final = False
        confidence = 0.0

        try:
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                logger.info("Listening for speech (%s)...", lang)
                audio_data = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
            logger.info("Processing speech...")
            # Use Google Web Speech API for transcription (free, requires internet)
            transcript = self.recognizer.recognize_google(audio_data, language=lang)
            is_final = True
            confidence = 0.85 # Google Web Speech API doesn't provide confidence directly in standard return
            
            logger.info("Transcription success: %s", transcript)
            
        except sr.WaitTimeoutError:
            logger.debug("Listening timeout - no speech detected.")
        except sr.UnknownValueError:
            logger.warning("Speech not recognized (UnknownValueError).")
        except sr.RequestError as e:
            logger.error("Could not request results from STT service: %s", e)
        except Exception as e:
            logger.error("Unexpected error in STT engine: %s", e)

        return {
            "transcript": transcript,
            "is_final": is_final,
            "confidence": confidence
        }
