"""Desktop UI widgets, views, and styling for IsharaConnect."""

from desktop_app.ui.theme import ThemeStyles, ThemeColors, BORDER_RADIUS


def get_main_window_cls():
    from desktop_app.ui.main_window import IsharaMainWindow
    return IsharaMainWindow


__all__ = [
    "ThemeStyles",
    "ThemeColors",
    "BORDER_RADIUS",
    "get_main_window_cls"
]

