"""Unit tests for Bengali Neural & Offline TTS Voice Engine."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core_engine.audio.tts_engine import TextToSpeechEngine, _generate_synthetic_acoustic_wav
from desktop_app.controllers.audio_player import AudioPlayerController


def test_offline_acoustic_wav_generation():
    """Verify offline harmonic formant synthesis produces valid RIFF WAV bytes."""
    wav_bytes = _generate_synthetic_acoustic_wav("ধন্যবাদ", duration_sec=0.3)
    assert wav_bytes is not None
    assert len(wav_bytes) > 100
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:16]


def test_tts_engine_synthesize_cached_or_fallback():
    """Verify synthesize_to_bytes returns valid audio data without crashing."""
    tts = TextToSpeechEngine()
    data = tts.synthesize_to_bytes("টেস্ট বার্তা", lang="bn")
    assert data is not None
    assert len(data) > 0


def test_audio_player_controller_speak_non_blocking():
    """Verify speak_bengali delegates cleanly without blocking the caller."""
    controller = AudioPlayerController()
    with patch.object(controller.tts, "speak_bengali", return_value=True) as mock_speak:
        res = controller.speak_bengali("স্বাগতম")
        assert res is True
        mock_speak.assert_called_once_with("স্বাগতম", async_mode=True)


def test_audio_player_controller_play_chime():
    """Verify play_chime triggers sound without exception."""
    controller = AudioPlayerController()
    with patch.object(controller.player, "play_chime") as mock_chime:
        controller.play_chime("success")
        mock_chime.assert_called_once_with("success")
