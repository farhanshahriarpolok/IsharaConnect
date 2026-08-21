"""Unit tests for the Theme Components."""

import pytest
from PyQt6.QtWidgets import QApplication
from desktop_app.ui.theme import ThemeStyles, ThemeColors
from desktop_app.ui.components.circular_gauge import CircularAccuracyGauge
from desktop_app.ui.components.badges import PulsingStatusBadge

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_theme_stylesheet():
    """Test theme string generation."""
    stylesheet = ThemeStyles.get_global_stylesheet()
    assert ThemeColors.BG_DARK in stylesheet
    assert ThemeColors.CYAN_ACCENT in stylesheet

def test_circular_gauge(qapp):
    """Test CircularAccuracyGauge component bounds."""
    gauge = CircularAccuracyGauge()
    assert gauge.value == 0.0
    
    gauge.set_value(50.0)
    assert gauge.value == 50.0
    
    # Test bounds
    gauge.set_value(150.0)
    assert gauge.value == 100.0
    
    gauge.set_value(-10.0)
    assert gauge.value == 0.0

def test_pulsing_badge(qapp):
    """Test PulsingStatusBadge component updating."""
    badge = PulsingStatusBadge()
    assert badge.label.text() == "Online"
    
    badge.set_status("Offline", ThemeColors.CORAL_ERROR)
    assert badge.label.text() == "Offline"
