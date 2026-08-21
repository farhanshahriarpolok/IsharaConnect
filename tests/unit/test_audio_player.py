import pytest
import base64
from unittest.mock import patch, MagicMock
from core_engine.audio.audio_player import AudioPlayer

@pytest.fixture
def player():
    return AudioPlayer()

def test_initialization(player):
    # Depending on the system, pygame might or might not be available,
    # but we just ensure it initializes without crashing.
    assert hasattr(player, "_initialized")
    assert hasattr(player, "_pygame_failed")

@patch("core_engine.audio.audio_player.tempfile.NamedTemporaryFile")
@patch("core_engine.audio.audio_player.threading.Thread.start")
def test_play_bytes_starts_thread(mock_thread_start, mock_temp_file, player):
    dummy_bytes = b"dummy audio data"
    result = player.play_bytes(dummy_bytes)
    assert result is True
    mock_thread_start.assert_called_once()

@patch("core_engine.audio.audio_player.AudioPlayer.play_bytes")
def test_play_base64_decodes(mock_play_bytes, player):
    mock_play_bytes.return_value = True
    dummy_bytes = b"hello"
    dummy_b64 = base64.b64encode(dummy_bytes).decode('utf-8')
    
    result = player.play_base64(dummy_b64)
    assert result is True
    mock_play_bytes.assert_called_once_with(dummy_bytes)

def test_play_base64_invalid(player):
    # Invalid base64 string
    result = player.play_base64("not_base64!@#")
    # Base64 decode throws exception, method should catch and return False
    assert result is False

@patch("core_engine.audio.audio_player.threading.Thread.start")
def test_play_chime_starts_thread(mock_thread_start, player):
    player.play_chime("success")
    mock_thread_start.assert_called_once()
