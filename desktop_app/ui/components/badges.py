"""Pulsing Status Badge Widget for IsharaConnect."""

from typing import Union
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer
from desktop_app.ui.theme import ThemeColors


class NeonDot(QWidget):
    """Animated glowing status dot with customizable pulse."""

    def __init__(self, color_hex: str = ThemeColors.EMERALD_SUCCESS, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.color = QColor(color_hex)
        self._opacity = 255
        self.pulsing = True

        # Setup pulse animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.toggle_pulse)
        self.timer.start(800)
        self.pulse_up = False

    def set_color(self, color_hex: str):
        """Update neon dot color."""
        self.color = QColor(color_hex)
        self.update()

    def toggle_pulse(self):
        """Toggle alpha between 120 and 255 for pulsing effect."""
        if not self.pulsing:
            self._opacity = 255
            self.update()
            return
        self.pulse_up = not self.pulse_up
        self._opacity = 120 if self.pulse_up else 255
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Outer glow
        glow_color = QColor(self.color)
        glow_color.setAlpha(int(self._opacity * 0.3))
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 16, 16)

        # Inner core
        core_color = QColor(self.color)
        core_color.setAlpha(self._opacity)
        painter.setBrush(QBrush(core_color))
        painter.drawEllipse(4, 4, 8, 8)


class PulsingStatusBadge(QWidget):
    """Badge widget combining NeonDot and QLabel with full text and status control."""

    def __init__(self, text: str = "Online", color: str = ThemeColors.EMERALD_SUCCESS, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.dot = NeonDot(color)
        self.label = QLabel(text)
        self.label.setStyleSheet(f"color: {color}; font-weight: bold;")

        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        layout.addStretch()

    def text(self) -> str:
        """Return the current label text."""
        return self.label.text()

    def setText(self, text: str):
        """Set badge text and automatically infer indicator color and state."""
        self.label.setText(text)
        text_lower = text.lower()

        if any(w in text_lower for w in ["offline", "disconnected", "stopped", "🔴"]):
            color = ThemeColors.CORAL_ERROR
            self.dot.pulsing = False
        elif any(w in text_lower for w in ["reconnecting", "connecting", "initializing", "🟡"]):
            color = "#F59E0B"
            self.dot.pulsing = True
        elif any(w in text_lower for w in ["cam error", "error", "warning", "⚠️"]):
            color = "#F59E0B"
            self.dot.pulsing = True
        elif any(w in text_lower for w in ["online", "connected", "ready", "active", "🟢"]):
            color = ThemeColors.EMERALD_SUCCESS
            self.dot.pulsing = True
        else:
            color = ThemeColors.CYAN_ACCENT
            self.dot.pulsing = True

        self.label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.dot.set_color(color)

    def set_status(self, text: str, is_connected: Union[bool, str] = True):
        """Update both badge text and connection status/color directly."""
        self.label.setText(text)
        if isinstance(is_connected, bool):
            color = ThemeColors.EMERALD_SUCCESS if is_connected else ThemeColors.CORAL_ERROR
            self.dot.pulsing = is_connected
        else:
            color = is_connected
            self.dot.pulsing = True

        self.label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.dot.set_color(color)
