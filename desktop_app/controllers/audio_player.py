"""Audio Player Controller for IsharaConnect Desktop Client.

Coordinates:
- Non-blocking audio playback (Pygame Mixer / Windows SAPI fallback)
- Integration with Bengali TTS Engine
- Audio cache pre-population for all 63 BdSL signs
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from core_engine.audio.audio_player import AudioPlayer, player_instance
from core_engine.audio.tts_engine import TextToSpeechEngine, tts_engine_instance

logger = logging.getLogger(__name__)


class AudioPlayerController:
    """Desktop Audio Controller coordinating sound effects and Bengali speech."""

    def __init__(self):
        self.player: AudioPlayer = player_instance
        self.tts: TextToSpeechEngine = tts_engine_instance

    def speak_bengali(self, text: str, async_mode: bool = True) -> bool:
        """Plays or queues Bengali speech without blocking the GUI thread."""
        if not text:
            return False
        return self.tts.speak_bengali(text, async_mode=async_mode)

    def play_sign_audio(self, sign_name_bn: str) -> bool:
        """Plays the spoken name of a sign."""
        return self.speak_bengali(sign_name_bn)

    def play_chime(self, sound_type: str = "success") -> None:
        """Plays an interactive UI chime."""
        self.player.play_chime(sound_type)

    def pre_cache_all_signs(self, labels_file: str = "dataset/labels.json") -> int:
        """Pre-caches offline audio files for all signs in dataset/labels.json."""
        p = Path(labels_file)
        if not p.exists():
            return 0

        cached_count = 0
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sign in data.get("signs", []):
                label_bn = sign.get("label_bn", "")
                if label_bn:
                    data_bytes = self.tts.synthesize_to_bytes(label_bn, lang="bn")
                    if data_bytes:
                        cached_count += 1
        except Exception as e:
            logger.error("Error during sign audio pre-caching: %s", e)

        return cached_count


# Global Desktop Audio Controller Singleton
audio_controller = AudioPlayerController()
