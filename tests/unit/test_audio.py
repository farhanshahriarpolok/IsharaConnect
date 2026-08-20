"""Unit tests for IsharaConnect Audio Subsystem (TTS/STT)."""

import pytest
from core_engine.audio.tts_engine import TextToSpeechEngine

@pytest.fixture
def tts_engine():
    engine = TextToSpeechEngine()
    yield engine
    engine.close()

def test_tts_speak_queue(tts_engine):
    """Test that text is successfully added to the async queue."""
    success = tts_engine.speak("হ্যালো", lang="bn", async_mode=True)
    assert success is True
    # The queue should have 1 item initially (though background thread might pick it up fast)
    # We just ensure it returns True on success and doesn't block
    
def test_tts_speak_empty(tts_engine):
    """Test that empty text is rejected."""
    success = tts_engine.speak("", async_mode=True)
    assert success is False

def test_synthesize_to_bytes(tts_engine):
    """Test synthesizing Bengali text into audio bytes."""
    audio_bytes = tts_engine.synthesize_to_bytes("ধন্যবাদ", lang="bn")
    assert audio_bytes is not None
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0
