"""Unit tests for AcademyDashboard 3-Column Split View with Right-Side Reference Guide."""

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


def test_academy_dashboard_3panel_init(qapp):
    """Verify AcademyDashboard initializes with Left (Lessons), Center (Arena), and Right (Reference Guide) panels."""
    dash = AcademyDashboard()
    
    # Left Panel
    assert dash.tree is not None
    assert dash.exam_btn is not None
    
    # Center Panel
    assert dash.arena_title is not None
    assert dash.camera_feed is not None
    assert dash.progress_bar is not None
    assert dash.match_gauge is not None
    assert dash.posture_hud is not None
    assert dash.start_practice_btn is not None
    assert dash.next_sign_btn is not None
    assert dash.restart_sign_btn is not None

    # Right Panel
    assert dash.ref_sign_bn is not None
    assert dash.ref_sign_en is not None
    assert dash.ref_phonetic is not None
    assert dash.ref_badge is not None
    assert dash.svg_widget is not None
    assert dash.listen_btn is not None
    assert dash.ref_instructions is not None


def test_academy_dashboard_update_reference_card(qapp):
    """Verify selecting a sign updates both Center Arena Title and Right Reference Card."""
    dash = AcademyDashboard()
    dash._update_reference_card("dhonnobad")
    assert dash.current_sign_slug == "dhonnobad"
    assert dash.ref_sign_bn.text() == "ধন্যবাদ"
    assert "Thank you" in dash.ref_sign_en.text()
    assert "উচ্চারণ" in dash.ref_phonetic.text()
    assert "ধন্যবাদ" in dash.arena_title.text()


def test_academy_dashboard_lesson_selection(qapp):
    """Verify programmatic lesson selection updates reference card and center title."""
    dash = AcademyDashboard()
    dash._on_lesson_selected("অ - Vowel A")
    assert "অ" in dash.ref_sign_bn.text()
    assert "অ" in dash.arena_title.text()


def test_academy_dashboard_next_and_restart_controls(qapp):
    """Verify Next Sign advances curriculum and Restart resets progress."""
    dash = AcademyDashboard()
    initial_slug = dash.current_sign_slug
    dash.progress_bar.setValue(80)
    
    dash._on_restart_sign()
    assert dash.progress_bar.value() == 0
    assert len(dash.temporal_frame_buffer) == 0

    dash._on_next_sign()
    assert dash.progress_bar.value() == 0


def test_academy_dashboard_listen_pronunciation(qapp):
    """Verify clicking listen pronunciation triggers speak_bengali."""
    dash = AcademyDashboard()
    dash._update_reference_card("dhonnobad")
    with patch("desktop_app.ui.academy_dashboard.audio_controller.speak_bengali") as mock_speak:
        dash._on_listen_pronunciation()
        mock_speak.assert_called_once_with("ধন্যবাদ")


def test_academy_dashboard_toggle_practice(qapp):
    """Verify toggling practice starts and stops timer and camera capture gracefully."""
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


def test_academy_dashboard_update_camera_feed(qapp):
    """Verify update_camera_feed handles QImage, QPixmap, and ndarray without errors."""
    import numpy as np
    from PyQt6.QtGui import QImage, QPixmap

    dash = AcademyDashboard()
    
    # 1. QImage
    img = QImage(640, 480, QImage.Format.Format_RGB888)
    img.fill(0)
    dash.update_camera_feed(img)
    assert dash.camera_feed.pixmap() is not None

    # 2. QPixmap
    pm = QPixmap(320, 240)
    dash.update_camera_feed(pm)
    assert dash.camera_feed.pixmap() is not None

    # 3. NumPy Array
    arr = np.zeros((240, 320, 3), dtype=np.uint8)
    dash.update_camera_feed(arr)
    assert dash.camera_feed.pixmap() is not None

    # 4. None safety
    dash.update_camera_feed(None)


def test_academy_dashboard_process_prediction(qapp):
    """Verify process_prediction updates accuracy progress bar and posture HUD."""
    dash = AcademyDashboard()
    dash._update_reference_card("dhonnobad")

    pred_data = {
        "label_bn": "ধন্যবাদ",
        "label_en": "Thank you",
        "confidence": 0.88,
        "is_stable": True
    }
    dash.process_prediction(pred_data)
    assert dash.progress_bar.value() == 88
    assert dash.match_gauge.value == 88.0
