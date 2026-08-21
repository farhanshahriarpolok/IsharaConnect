"""Interactive Visual Motion Trajectory Viewer for BdSL Dynamic Gestures.

Maintains rolling queues of hand landmark coordinates and paints glowing neon
bezier trails directly on the live camera stream overlay.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QBrush
from PyQt6.QtCore import Qt, QPointF
from collections import deque
from typing import Optional, Tuple
from desktop_app.ui.theme import ThemeColors


class MotionTrajectoryViewer(QWidget):
    """Draws vibrant neon trailing bezier motion paths over video feeds for dynamic gestures."""

    def __init__(self, parent=None, max_trail: int = 15):
        super().__init__(parent)
        self.max_trail = max_trail
        self.left_wrist_trail = deque(maxlen=self.max_trail)
        self.right_wrist_trail = deque(maxlen=self.max_trail)
        self.left_index_trail = deque(maxlen=self.max_trail)
        self.right_index_trail = deque(maxlen=self.max_trail)

        # Ensure transparency and mouse event pass-through
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")

    def update_trajectory(
        self,
        left_wrist: Optional[Tuple[float, float]] = None,
        right_wrist: Optional[Tuple[float, float]] = None,
        left_index: Optional[Tuple[float, float]] = None,
        right_index: Optional[Tuple[float, float]] = None,
    ):
        """Update motion trails with latest pixel coordinates (x, y)."""
        if left_wrist is not None and left_wrist[0] > 0 and left_wrist[1] > 0:
            self.left_wrist_trail.append(QPointF(float(left_wrist[0]), float(left_wrist[1])))
        elif left_wrist is None and len(self.left_wrist_trail) > 0:
            # Decay trail if hand is not visible
            self.left_wrist_trail.popleft()

        if right_wrist is not None and right_wrist[0] > 0 and right_wrist[1] > 0:
            self.right_wrist_trail.append(QPointF(float(right_wrist[0]), float(right_wrist[1])))
        elif right_wrist is None and len(self.right_wrist_trail) > 0:
            self.right_wrist_trail.popleft()

        if left_index is not None and left_index[0] > 0 and left_index[1] > 0:
            self.left_index_trail.append(QPointF(float(left_index[0]), float(left_index[1])))
        elif left_index is None and len(self.left_index_trail) > 0:
            self.left_index_trail.popleft()

        if right_index is not None and right_index[0] > 0 and right_index[1] > 0:
            self.right_index_trail.append(QPointF(float(right_index[0]), float(right_index[1])))
        elif right_index is None and len(self.right_index_trail) > 0:
            self.right_index_trail.popleft()

        self.update()

    def clear_trails(self):
        """Clear all active motion trajectories."""
        self.left_wrist_trail.clear()
        self.right_wrist_trail.clear()
        self.left_index_trail.clear()
        self.right_index_trail.clear()
        self.update()

    def paintEvent(self, event):
        """Paints anti-aliased fading glowing bezier trails on top of the parent feed."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Left Hand - Cyan / Electric Blue Trails
        cyan_color = QColor(ThemeColors.CYAN_ACCENT)
        self._draw_glowing_trail(painter, list(self.left_wrist_trail), cyan_color, base_width=10)
        self._draw_glowing_trail(painter, list(self.left_index_trail), cyan_color.lighter(130), base_width=6)

        # 2. Right Hand - Emerald / Electric Green Trails
        emerald_color = QColor(ThemeColors.EMERALD_SUCCESS)
        self._draw_glowing_trail(painter, list(self.right_wrist_trail), emerald_color, base_width=10)
        self._draw_glowing_trail(painter, list(self.right_index_trail), emerald_color.lighter(130), base_width=6)

    def _draw_glowing_trail(self, painter: QPainter, trail: list, color: QColor, base_width: int = 8):
        """Draws multi-pass smoothed bezier curves with decaying alpha trails and leading particle head."""
        if len(trail) < 2:
            return

        # Build smoothed bezier curve path
        path = QPainterPath()
        path.moveTo(trail[0])

        for i in range(1, len(trail) - 1):
            p0 = trail[i]
            p1 = trail[i + 1]
            midpoint = QPointF((p0.x() + p1.x()) / 2.0, (p0.y() + p1.y()) / 2.0)
            path.quadTo(p0, midpoint)

        path.lineTo(trail[-1])

        # Multi-layer neon aura glow (Outer glow -> Mid glow -> Core laser)
        layers = [
            (base_width * 2.0, 35),   # Outer soft diffused aura
            (base_width * 1.2, 90),   # Mid glow
            (base_width * 0.4, 220),  # Inner vibrant neon
            (1.5, 255)                # Core white-hot laser center
        ]

        for width, alpha in layers:
            pen_color = QColor(color)
            if width <= 2.0:
                pen_color = QColor(255, 255, 255, alpha)
            else:
                pen_color.setAlpha(alpha)

            pen = QPen(pen_color)
            pen.setWidthF(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        # Draw particle pulse head at leading tip
        head_pt = trail[-1]
        
        # Outer pulsating halo
        halo_color = QColor(color)
        halo_color.setAlpha(80)
        painter.setBrush(QBrush(halo_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(head_pt, base_width * 1.2, base_width * 1.2)

        # Core bead
        core_color = QColor(255, 255, 255, 240)
        painter.setBrush(QBrush(core_color))
        painter.drawEllipse(head_pt, base_width * 0.4, base_width * 0.4)
