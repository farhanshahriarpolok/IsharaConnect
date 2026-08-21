"""Unit tests for SignCardViewer Component."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPainter, QImage

from desktop_app.ui.components.sign_card_viewer import SignCardViewer


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_sign_card_viewer_init(qapp):
    """Verify SignCardViewer initializes with correct dimensions and properties."""
    viewer = SignCardViewer("dhonnobad", "ধন্যবাদ", "Thank you")
    assert viewer.slug == "dhonnobad"
    assert viewer.label_bn == "ধন্যবাদ"
    assert viewer.label_en == "Thank you"
    assert viewer.minimumWidth() >= 280
    assert viewer.minimumHeight() >= 220


def test_sign_card_viewer_load_existing_and_missing_sign(qapp):
    """Verify load_sign handles existing SVG files and unknown sign fallbacks."""
    viewer = SignCardViewer()

    # Load known sign
    res_known = viewer.load_sign("dhonnobad", "ধন্যবাদ", "Thank you")
    assert viewer.slug == "dhonnobad"

    # Load non-existent sign
    res_unknown = viewer.load_sign("unknown_xyz_sign_999", "অজ্ঞাত", "Unknown")
    assert viewer.slug == "unknown_xyz_sign_999"
    assert viewer.svg_renderer is None or not viewer.svg_renderer.isValid()


def test_sign_card_viewer_paint_event(qapp):
    """Verify paintEvent executes and renders onto canvas without exception."""
    viewer = SignCardViewer("dhonnobad", "ধন্যবাদ", "Thank you")
    viewer.resize(300, 240)

    img = QImage(300, 240, QImage.Format.Format_ARGB32)
    painter = QPainter(img)
    try:
        viewer._paint_fallback_graphic(painter, 300.0, 240.0)
    finally:
        painter.end()
