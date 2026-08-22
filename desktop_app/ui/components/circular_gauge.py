"""Circular Accuracy Gauge Widget."""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRectF
from desktop_app.ui.theme import ThemeColors


class CircularAccuracyGauge(QWidget):
    def __init__(self, parent=None, radius: int = 40, thickness: int = 8):
        super().__init__(parent)
        self.radius = radius
        self.thickness = thickness
        self.setFixedSize(self.radius * 2, self.radius * 2)
        self.value = 0.0

    def set_value(self, val: float):
        if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            self.value = 0.0
        else:
            self.value = max(0.0, min(100.0, float(val)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.thickness / 2, self.thickness / 2,
                      self.width() - self.thickness, self.height() - self.thickness)

        # Draw background track
        pen_bg = QPen(QColor(ThemeColors.SURFACE_DARK))
        pen_bg.setWidth(self.thickness)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16)

        safe_val = 0.0 if (math.isnan(self.value) or math.isinf(self.value)) else self.value

        # Draw progress arc
        if safe_val > 0:
            if safe_val > 80:
                color = QColor(ThemeColors.EMERALD_SUCCESS)
            elif safe_val > 50:
                color = QColor(ThemeColors.CYAN_ACCENT)
            else:
                color = QColor(ThemeColors.CORAL_ERROR)
                
            pen_fg = QPen(color)
            pen_fg.setWidth(self.thickness)
            pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_fg)
            span_angle = int((safe_val / 100.0) * -360 * 16)
            painter.drawArc(rect, 90 * 16, span_angle)

        # Draw text
        painter.setPen(QColor(ThemeColors.TEXT_PRIMARY))
        font = QFont("Inter", int(self.radius * 0.35), QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(safe_val)}%")
