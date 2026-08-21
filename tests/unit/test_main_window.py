"""Unit tests for IsharaMainWindow and MainWindow."""

import pytest
from unittest.mock import patch, MagicMock
from PyQt6.QtWidgets import QApplication
from desktop_app.ui.main_window import IsharaMainWindow, MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@patch("desktop_app.ui.main_window.CameraWorker")
@patch("desktop_app.ui.main_window.NetworkWorker")
def test_main_window_initialization(mock_net_worker, mock_cam_worker, qapp):
    """Verify MainWindow initializes with default and custom values and creates fps_lbl."""
    mock_cam_inst = MagicMock()
    mock_cam_worker.return_value = mock_cam_inst
    mock_net_inst = MagicMock()
    mock_net_worker.return_value = mock_net_inst

    win = MainWindow()
    assert win.mode == "signer"
    assert win.room_id == "room_public_01"
    assert hasattr(win, "fps_lbl")
    assert win.fps_lbl is not None
    assert win.fps_lbl.text() == "FPS: 0.0"

    # Close window cleanly
    win.close()


@patch("desktop_app.ui.main_window.CameraWorker")
@patch("desktop_app.ui.main_window.NetworkWorker")
def test_update_fps_slot(mock_net_worker, mock_cam_worker, qapp):
    """Verify _update_fps slot updates label text correctly and handles missing label safely."""
    mock_cam_worker.return_value = MagicMock()
    mock_net_worker.return_value = MagicMock()

    win = IsharaMainWindow(mode="signer", room_id="test_room", server_url="ws://127.0.0.1:8000")
    
    # 1. Normal update
    win._update_fps(30.0)
    assert win.fps_lbl.text() == "FPS: 30.0"

    win._update_fps(59.94)
    assert win.fps_lbl.text() == "FPS: 59.9"

    # 2. Safety check: fps_lbl is None
    win.fps_lbl = None
    win._update_fps(24.0)  # Should not raise AttributeError

    # 3. Safety check: fps_lbl attribute deleted
    del win.fps_lbl
    win._update_fps(15.0)  # Should not raise AttributeError

    win.close()


@patch("desktop_app.ui.main_window.CameraWorker")
@patch("desktop_app.ui.main_window.NetworkWorker")
def test_mode_switching_and_fps_visibility(mock_net_worker, mock_cam_worker, qapp):
    """Verify mode switching and fps_lbl visibility toggling."""
    mock_cam_worker.return_value = MagicMock()
    mock_net_worker.return_value = MagicMock()

    win = IsharaMainWindow(mode="signer", room_id="test_room", server_url="ws://127.0.0.1:8000")
    
    # Switch to Learning Hub
    win._change_app_mode("🎓 Learning Hub")
    assert win.fps_lbl.isHidden()

    # Switch back to Communication Mode
    win._change_app_mode("💬 Communication Mode")
    assert not win.fps_lbl.isHidden()

    # Switch to Speaker Mode
    win._change_mode("Speaker")
    assert win.mode == "speaker"

    win.close()
