"""Audio Player Controller for IsharaConnect Desktop Client.

Features:
- Windows winsound non-blocking asynchronous audio playback
- Multi-path resolution for dataset/audio_cache/ (.wav and .mp3)
- Real-time Bengali speech synthesis fallback via TextToSpeechEngine / pyttsx3
- Zero GUI thread blocking
"""

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

from core_engine.audio.audio_player import AudioPlayer, player_instance
from core_engine.audio.tts_engine import TextToSpeechEngine, tts_engine_instance

logger = logging.getLogger(__name__)


class AudioPlayerController:
    """Desktop Audio Controller coordinating sound effects and Bengali speech."""

    def __init__(self):
        self.player: AudioPlayer = player_instance
        self.tts: TextToSpeechEngine = tts_engine_instance

    def resolve_audio_file(self, text_or_slug: str) -> Optional[Path]:
        """Resolves audio file in dataset/audio_cache/ for a given slug or Bengali text."""
        cache_dirs = [
            Path(__file__).resolve().parents[2] / "dataset" / "audio_cache",
            Path.cwd() / "dataset" / "audio_cache",
            Path("dataset/audio_cache")
        ]

        slug = "".join(c if c.isalnum() else "_" for c in text_or_slug).strip("_")

        for c_dir in cache_dirs:
            if not c_dir.exists():
                continue

            # Check direct .wav and .mp3
            for ext in [".wav", ".mp3"]:
                p = c_dir / f"{slug}_bn{ext}"
                if p.exists():
                    return p
                p_direct = c_dir / f"{slug}{ext}"
                if p_direct.exists():
                    return p_direct

            # Fuzzy glob
            for f in c_dir.glob(f"*{slug}*"):
                if f.suffix in [".wav", ".mp3"]:
                    return f

        return None

    def speak_bengali(self, text_or_slug: str, async_mode: bool = True) -> bool:
        """Plays cached audio or synthesizes Bengali speech asynchronously."""
        if not text_or_slug:
            return False

        clean_val = text_or_slug.strip()

        # 1. Try local .wav resolution for instant winsound playback on Windows
        audio_path = self.resolve_audio_file(clean_val)
        if audio_path and audio_path.exists() and audio_path.suffix == ".wav" and WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(str(audio_path.resolve()), winsound.SND_FILENAME | winsound.SND_ASYNC)
                return True
            except Exception as e:
                logger.debug(f"winsound file playback failed: {e}")

        # 2. Asynchronous TTS synthesis / player fallback
        return self.tts.speak_bengali(clean_val, async_mode=async_mode)

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
            logger.error(f"Error during sign audio pre-caching: {e}")

        return cached_count


# Global Desktop Audio Controller Singleton
audio_controller = AudioPlayerController()
