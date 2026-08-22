"""Unit tests for SubtitleTickerWidget and Desktop NLP Live Translation HUD."""

import pytest
from PyQt6.QtWidgets import QApplication

from desktop_app.ui.components.subtitle_ticker import SubtitleTickerWidget
from desktop_app.ui.main_window import IsharaMainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_subtitle_ticker_initialization(qapp):
    """Test SubtitleTickerWidget instantiation and default state."""
    widget = SubtitleTickerWidget()
    assert widget.objectName() == "SubtitleTickerHUD"
    assert widget.sentence_label.text() == "অপেক্ষা করছি..."
    assert widget.confidence_pill.isHidden() is True
    assert len(widget.gloss_buffer) == 0


def test_subtitle_ticker_update_active_glosses(qapp):
    """Test updating active gesture chips."""
    widget = SubtitleTickerWidget()
    widget.update_active_glosses(["আমি", "ভাত"])

    assert widget.gloss_buffer == ["আমি", "ভাত"]
    assert widget.chips_layout.count() == 2


def test_subtitle_ticker_update_translation(qapp):
    """Test updating inflected translation and confidence badge."""
    widget = SubtitleTickerWidget()
    widget.update_translation("আমি ভাত খাচ্ছি।", confidence=0.95, is_final=True)

    assert widget.sentence_label.text() == "আমি ভাত খাচ্ছি।"
    assert widget.confidence_pill.isHidden() is False
    assert "95%" in widget.confidence_pill.text()
    assert widget.current_sentence_bn == "আমি ভাত খাচ্ছি।"


def test_subtitle_ticker_clear(qapp):
    """Test clearing active chips and resetting state."""
    widget = SubtitleTickerWidget()
    widget.update_active_glosses(["আমি", "ভাত"])
    widget.update_translation("আমি ভাত খাচ্ছি।", 0.95, True)
    assert widget.chips_layout.count() == 2

    widget.clear_ticker()
    assert widget.chips_layout.count() == 0
    assert widget.sentence_label.text() == "অপেক্ষা করছি..."
    assert widget.confidence_pill.isHidden() is True
    assert len(widget.gloss_buffer) == 0


def test_main_window_subtitle_ticker_wiring(qapp):
    """Test MainWindow wires SubtitleTickerWidget and GlossToSentenceTranslator."""
    window = IsharaMainWindow(mode="signer")
    assert hasattr(window, "subtitle_ticker")
    assert hasattr(window, "translator")
    assert window.subtitle_ticker is not None
    assert window.translator is not None

    # Simulate sign detection signal
    window._on_sign_detected({
        "label_bn": "ধন্যবাদ",
        "label_en": "Thank you",
        "confidence": 0.96,
        "is_stable": True,
        "is_new_trigger": True
    })

    # Trigger clear HUD
    window._clear_hud()
    assert window.subtitle_ticker.sentence_label.text() == "অপেক্ষা করছি..."

    window.close()
