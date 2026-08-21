"""Unit tests for AcademyDashboard Side-by-Side Dual-Panel View."""

import os
from unittest.mock import MagicMock, patch
import pytest
from PyQt6.QtWidgets import QApplication

from desktop_app.ui.academy_dashboard import AcademyDashboard

@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_academy_dashboard_dual_panel_init(qapp):
    """Verify AcademyDashboard initializes with Target Reference Card and Live Arena widgets."""
    dash = AcademyDashboard()
    assert dash.ref_sign_bn is not None
    assert dash.ref_sign_en is not None
    assert dash.ref_phonetic is not None
    assert dash.listen_btn is not None
    assert dash.camera_feed is not None
    assert dash.match_gauge is not None
    assert dash.posture_hud is not None


def test_academy_dashboard_update_reference_card(qapp):
    """Verify selecting a sign updates reference card labels and metadata."""
    dash = AcademyDashboard()
    dash._update_reference_card("dhonnobad")
    assert dash.current_sign_slug == "dhonnobad"
    assert dash.ref_sign_bn.text() == "ধন্যবাদ"
    assert "Thank you" in dash.ref_sign_en.text()
    assert "উচ্চারণ" in dash.ref_phonetic.text()


def test_academy_dashboard_listen_pronunciation(qapp):
    """Verify clicking listen pronunciation triggers speak_bengali."""
    dash = AcademyDashboard()
    dash._update_reference_card("dhonnobad")
    with patch("desktop_app.ui.academy_dashboard.audio_controller.speak_bengali") as mock_speak:
        dash._on_listen_pronunciation()
        mock_speak.assert_called_once_with("ধন্যবাদ")


def test_academy_dashboard_toggle_practice(qapp):
    """Verify toggling practice starts and stops timer gracefully."""
    dash = AcademyDashboard()
    with patch("cv2.VideoCapture") as mock_cap:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        # Start practice
        dash._toggle_practice()
        assert dash.practice_running is True
        assert dash.timer.isActive() is True

        # Stop practice
        dash._toggle_practice()
        assert dash.practice_running is False
        assert dash.timer.isActive() is False
