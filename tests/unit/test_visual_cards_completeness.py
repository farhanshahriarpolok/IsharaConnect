"""Unit tests for Complete 63 BdSL Visual Cards and 3-Column Learning Hub Mount."""

import json
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication

from desktop_app.ui.components.sign_card_viewer import SignCardViewer
from desktop_app.ui.academy_dashboard import AcademyDashboard
from desktop_app.ui.main_window import IsharaMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_visual_cards_exist_for_all_63_signs():
    """Verify dataset/visual_cards/ contains visual assets for all 63 canonical signs."""
    labels_file = Path("dataset/labels.json")
    assert labels_file.exists(), "dataset/labels.json must exist"

    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    signs = data.get("signs", [])
    assert len(signs) >= 60

    missing = []
    for sign in signs:
        slug = sign.get("slug")
        svg_file = Path(f"dataset/visual_cards/{slug}.svg")
        png_file = Path(f"dataset/visual_cards/{slug}.png")
        if not svg_file.exists() and not png_file.exists():
            missing.append(slug)

    assert len(missing) == 0, f"Missing visual cards for: {missing}"


def test_sign_card_viewer_loads_all_63_signs(qapp):
    """Verify SignCardViewer successfully resolves and loads every sign in labels.json."""
    labels_file = Path("dataset/labels.json")
    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    viewer = SignCardViewer()
    for sign in data.get("signs", []):
        slug = sign.get("slug")
        bn = sign.get("label_bn", "")
        en = sign.get("label_en", "")
        loaded = viewer.load_sign(slug, bn, en)
        assert loaded is True, f"Failed loading card for sign: {slug}"


def test_main_window_learning_mode_mounts_3column_academy(qapp):
    """Verify IsharaMainWindow switches to AcademyDashboard on learning mode."""
    win = IsharaMainWindow()
    win._set_mode("learning")
    assert win.stacked_widget.currentWidget() == win.academy_view
    assert isinstance(win.stacked_widget.currentWidget(), AcademyDashboard)
