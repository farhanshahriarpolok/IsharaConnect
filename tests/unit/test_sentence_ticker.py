"""Unit tests for the Sentence Ticker HUD."""

import pytest
from PyQt6.QtWidgets import QApplication
from desktop_app.ui.components.sentence_ticker import SentenceTickerWidget

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

@pytest.fixture
def ticker(qapp):
    return SentenceTickerWidget()

def test_ticker_initial_state(ticker):
    """Test the initial state of the ticker."""
    assert len(ticker.gloss_buffer) == 0
    assert ticker.finalized_sentence_bn == ""
    assert ticker.finalized_sentence_en == ""

def test_ticker_update(ticker):
    """Test updating the ticker with signs."""
    sign_data_1 = {"sign_id": 1, "label_bn": "আমি", "label_en": "I"}
    sign_data_2 = {"sign_id": 2, "label_bn": "সাহায্য", "label_en": "help"}
    
    ticker.update_ticker(sign_data_1)
    assert len(ticker.gloss_buffer) == 1
    assert ticker.finalized_sentence_bn == "আমি।"
    
    ticker.update_ticker(sign_data_2)
    assert len(ticker.gloss_buffer) == 2
    assert ticker.finalized_sentence_bn == "আমি সাহায্য।"
    assert ticker.finalized_sentence_en == "I help."

def test_ticker_clear(ticker):
    """Test clearing the ticker."""
    sign_data_1 = {"sign_id": 1, "label_bn": "আমি", "label_en": "I"}
    ticker.update_ticker(sign_data_1)
    ticker.clear_buffer()
    
    assert len(ticker.gloss_buffer) == 0
    assert ticker.sentence_label.text() == "অপেক্ষা করছি..."
