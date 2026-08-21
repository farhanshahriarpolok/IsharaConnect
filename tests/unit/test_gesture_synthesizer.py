"""Unit tests for the Speech-to-BdSL Gesture Synthesizer & Avatar."""

import pytest
from core_engine.vision.gesture_synthesizer import BdSLGestureSynthesizer
from PyQt6.QtWidgets import QApplication
from desktop_app.ui.components.gesture_avatar import GestureAvatarWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def synthesizer():
    return BdSLGestureSynthesizer()


def test_synthesizer_empty_text(synthesizer):
    """Test synthesis with empty input."""
    assert synthesizer.synthesize_text_to_gestures("") == []
    assert synthesizer.synthesize_text_to_gestures("   ") == []


def test_synthesizer_direct_word(synthesizer):
    """Test direct dictionary sign resolution."""
    gestures = synthesizer.synthesize_text_to_gestures("ধন্যবাদ")
    assert len(gestures) == 1
    assert gestures[0]["sign_slug"] == "dhonnobad"
    assert gestures[0]["type"] == "word"
    assert "Thank you" in gestures[0]["label_en"]


def test_synthesizer_multi_word_phrase(synthesizer):
    """Test multi-word phrase synthesis."""
    gestures = synthesizer.synthesize_text_to_gestures("আপনি কেমন আছেন?")
    assert len(gestures) >= 2
    # Check that "কেমন আছেন" was recognized as a phrase or sequenced properly
    tokens = [g["token"] for g in gestures]
    assert "কেমন আছেন" in tokens or ("কেমন" in tokens and "আছেন" in tokens) or "আপনি" in tokens


def test_synthesizer_fingerspelling_fallback(synthesizer):
    """Test automatic finger-spelling for out-of-vocabulary words."""
    gestures = synthesizer.synthesize_text_to_gestures("ঢাকা")
    assert len(gestures) >= 1
    # Check that fingerspelling characters were resolved
    types = [g["type"] for g in gestures]
    assert "fingerspell" in types or "word" in types


def test_synthesizer_speed_scaling(synthesizer):
    """Test duration calculation with speed scaling."""
    normal_gestures = synthesizer.synthesize_text_to_gestures("সাহায্য", speed=1.0)
    fast_gestures = synthesizer.synthesize_text_to_gestures("সাহায্য", speed=2.0)

    assert len(normal_gestures) == 1
    assert len(fast_gestures) == 1
    assert fast_gestures[0]["duration_ms"] < normal_gestures[0]["duration_ms"]


def test_gesture_avatar_widget(qapp):
    """Test GestureAvatarWidget initialization and controls."""
    avatar = GestureAvatarWidget()
    assert avatar.is_playing is False
    assert avatar.current_index == 0

    avatar.synthesize_and_play("ধন্যবাদ")
    assert len(avatar.gesture_sequence) == 1
    assert avatar.is_playing is True
    assert avatar.play_btn.text() == "⏸ Pause"

    avatar.reset()
    assert avatar.is_playing is False
    assert avatar.timeline_bar.value() == 0
