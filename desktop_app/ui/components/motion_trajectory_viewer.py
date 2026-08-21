"""Interactive Visual Motion Trajectory Viewer."""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QPointF
from collections import deque
import numpy as np
from desktop_app.ui.theme import ThemeColors

class MotionTrajectoryViewer(QWidget):
    """Draws trailing bezier curves over the video feed to visualize motion paths."""
    
    def __init__(self, parent=None, max_trail: int = 15):
        super().__init__(parent)
        self.max_trail = max_trail
        self.left_trail = deque(maxlen=self.max_trail)
        self.right_trail = deque(maxlen=self.max_trail)
        
        # We overlay this widget on top of the camera feed, so it must be transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        
    def update_trajectory(self, left_point: tuple = None, right_point: tuple = None):
        """Update trails with latest screen coordinates (x, y)."""
        if left_point:
            self.left_trail.append(QPointF(left_point[0], left_point[1]))
        
        if right_point:
            self.right_trail.append(QPointF(right_point[0], right_point[1]))
            
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self._draw_trail(painter, list(self.left_trail), QColor(ThemeColors.CYAN_ACCENT))
        self._draw_trail(painter, list(self.right_trail), QColor(ThemeColors.EMERALD_SUCCESS))
        
    def _draw_trail(self, painter: QPainter, trail: list, color: QColor):
        if len(trail) < 2:
            return
            
        path = QPainterPath()
        path.moveTo(trail[0])
        
        # Create a smoothed path using quadratic bezier curves
        for i in range(1, len(trail) - 1):
            p0 = trail[i]
            p1 = trail[i+1]
            midpoint = QPointF((p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2)
            path.quadTo(p0, midpoint)
            
        path.lineTo(trail[-1])
        
        # Neon glow effect
        for width, alpha in [(12, 40), (6, 120), (2, 255)]:
            pen_color = QColor(color)
            pen_color.setAlpha(alpha)
            pen = QPen(pen_color)
            pen.setWidth(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)
            
        # Draw leading dot (head of the trail)
        head_color = QColor(color)
        head_color.setAlpha(255)
        painter.setBrush(head_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(trail[-1], 6, 6)
