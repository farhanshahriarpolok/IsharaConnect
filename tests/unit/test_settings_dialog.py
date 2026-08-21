"""Unit tests for Settings Dialog and Configuration persistence."""

import json
import pytest
from PyQt6.QtWidgets import QApplication

from desktop_app.ui.dialogs.settings_dialog import (
    SettingsDialog,
    load_user_settings,
    save_user_settings,
    DEFAULT_SETTINGS,
    CameraScannerThread,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_load_and_save_user_settings(tmp_path, monkeypatch):
    """Test saving and loading user settings file."""
    custom_settings_file = tmp_path / "user_settings.json"
    monkeypatch.setattr("desktop_app.ui.dialogs.settings_dialog.SETTINGS_FILE_PATH", custom_settings_file)
    
    defaults = load_user_settings()
    assert defaults["camera_id"] == 0
    assert defaults["sensitivity"] == "normal"
    
    new_settings = dict(defaults)
    new_settings["confidence_threshold"] = 0.85
    new_settings["sensitivity"] = "strict"
    new_settings["tts_volume"] = 75
    
    success = save_user_settings(new_settings)
    assert success is True
    assert custom_settings_file.exists()
    
    reloaded = load_user_settings()
    assert reloaded["confidence_threshold"] == 0.85
    assert reloaded["sensitivity"] == "strict"
    assert reloaded["tts_volume"] == 75


def test_settings_dialog_ui(qapp):
    """Test SettingsDialog UI controls and signal emission."""
    init_config = {
        "camera_id": 1,
        "resolution": [1280, 720],
        "confidence_threshold": 0.80,
        "stability_timer": 3.0,
        "sensitivity": "high",
        "mirror_mode": False,
        "tts_volume": 80,
        "tts_muted": True,
        "server_url": "ws://192.168.1.100:8000"
    }
    
    dialog = SettingsDialog(current_settings=init_config)
    assert dialog.conf_slider.value() == 80
    assert dialog.hold_timer_spin.value() == 3.0
    assert dialog.cb_mute.isChecked() is True
    assert dialog.cb_mirror.isChecked() is False
    assert dialog.server_url_input.text() == "ws://192.168.1.100:8000"
    
    # Test changing a value and getting settings
    dialog.conf_slider.setValue(90)
    current = dialog._get_current_settings_from_ui()
    assert current["confidence_threshold"] == 0.90
    
    # Test signal emission
    received_signal = []
    dialog.settings_applied.connect(lambda s: received_signal.append(s))
    dialog._apply_settings()
    assert len(received_signal) == 1
    assert received_signal[0]["confidence_threshold"] == 0.90
    
    dialog.close()


def test_camera_scanner_thread(qapp):
    """Test camera scanner thread emits found devices list."""
    scanner = CameraScannerThread()
    devices_found = []
    scanner.scan_finished.connect(lambda devs: devices_found.append(devs))
    scanner.run()  # Run synchronously in test
    
    assert len(devices_found) == 1
    assert isinstance(devices_found[0], list)
    assert len(devices_found[0]) >= 1
