"""Pulsing Status Badge Widget."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, pyqtProperty
from desktop_app.ui.theme import ThemeColors

class NeonDot(QWidget):
    def __init__(self, color_hex: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.color = QColor(color_hex)
        self._opacity = 255
        
        # Setup pulse animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.toggle_pulse)
        self.timer.start(800)
        self.pulse_up = False

    def toggle_pulse(self):
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
    def __init__(self, text: str = "Online", color: str = ThemeColors.EMERALD_SUCCESS, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.dot = NeonDot(color)
        self.label = QLabel(text)
        self.label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        layout.addStretch()
        
    def set_status(self, text: str, color: str):
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.dot.color = QColor(color)
        self.dot.update()
