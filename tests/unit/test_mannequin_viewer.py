"""Unit tests for 3D Mannequin Action Viewer PyQt6 component."""

import sys
from pathlib import Path
import pytest

from PyQt6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

app = QApplication.instance()
if app is None:
    app = QApplication([])

from desktop_app.ui.components.mannequin_action_viewer import MannequinActionViewer


class TestMannequinActionViewer:
    def test_instantiation_and_sign_mapping(self):
        viewer = MannequinActionViewer()
        assert viewer is not None

        # Test gloss mapping
        assert viewer.set_sign("ধন্যবাদ") == "thank_you"
        assert viewer.set_sign("মা") == "mother"
        assert viewer.set_sign("বাবা") == "father"
        assert viewer.set_sign("ডাক্তার") == "doctor"
        assert viewer.set_sign("সালাম") == "salam"
        assert viewer.set_sign("পুলিশ") == "police"
        assert viewer.set_sign("unknown_gloss") == "police"
