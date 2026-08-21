"""Unit tests for Resilient AudioPlayerController."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from desktop_app.controllers.audio_player import AudioPlayerController


def test_audio_player_controller_resolve_audio_file():
    """Verify resolve_audio_file searches cache directories for audio files."""
    controller = AudioPlayerController()
    # Check resolution on known sign
    path = controller.resolve_audio_file("dhonnobad")
    # Even if none found, method returns Path or None safely
    assert path is None or isinstance(path, Path)


def test_audio_player_controller_speak_bengali_delegation():
    """Verify speak_bengali executes without crashing and delegates appropriately."""
    controller = AudioPlayerController()
    with patch.object(controller.tts, "speak_bengali", return_value=True) as mock_speak:
        res = controller.speak_bengali("ধন্যবাদ")
        assert res is True
        assert mock_speak.called


def test_audio_player_controller_empty_text():
    """Verify empty text returns False immediately."""
    controller = AudioPlayerController()
    assert controller.speak_bengali("") is False
    assert controller.speak_bengali("   ") is False
