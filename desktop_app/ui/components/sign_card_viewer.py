"""Bulletproof SignCardViewer Widget for rendering SVG Visual Cards and Geometric Fallbacks."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

try:
    from PyQt6.QtSvg import QSvgRenderer
    SVG_AVAILABLE = True
except ImportError:
    SVG_AVAILABLE = False

logger = logging.getLogger(__name__)

COLOR_BG = QColor("#181825")
COLOR_BORDER = QColor("#06B6D4")
COLOR_TEXT_PRIMARY = QColor("#CDD6F4")
COLOR_ACCENT = QColor("#89B4FA")
COLOR_EMERALD = QColor("#10B981")


class SignCardViewer(QWidget):
    """Renders visual BdSL sign cards via SVG with high-contrast canvas fallback."""

    def __init__(
        self,
        slug: str = "dhonnobad",
        label_bn: str = "ধন্যবাদ",
        label_en: str = "Thank you",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.slug = slug
        self.label_bn = label_bn
        self.label_en = label_en
        self.svg_renderer: Optional[QSvgRenderer] = None
        self.svg_path: Optional[Path] = None

        self.setMinimumSize(280, 220)
        self.load_sign(slug, label_bn, label_en)

    def load_sign(self, slug: str, label_bn: str = "", label_en: str = "") -> bool:
        """Resolves and loads the visual SVG card for the given sign."""
        self.slug = slug
        if label_bn:
            self.label_bn = label_bn
        if label_en:
            self.label_en = label_en

        self.svg_renderer = None
        self.svg_path = self._resolve_svg_path(slug)

        if self.svg_path and SVG_AVAILABLE:
            try:
                renderer = QSvgRenderer(str(self.svg_path))
                if renderer.isValid():
                    self.svg_renderer = renderer
            except Exception as e:
                logger.debug(f"Failed loading SVG {self.svg_path}: {e}")

        self.update()
        return self.svg_renderer is not None

    def _resolve_svg_path(self, slug: str) -> Optional[Path]:
        """Resolves the absolute path to dataset/visual_cards/{slug}.svg."""
        if not slug:
            return None

        clean_slug = slug.strip().lower()
        candidates = [
            Path(__file__).resolve().parents[3] / "dataset" / "visual_cards" / f"{clean_slug}.svg",
            Path.cwd() / "dataset" / "visual_cards" / f"{clean_slug}.svg",
            Path("dataset/visual_cards") / f"{clean_slug}.svg"
        ]

        for p in candidates:
            if p.exists():
                return p

        # Check in visual_cards directory with exact stem match ignoring case
        vc_dir = Path.cwd() / "dataset" / "visual_cards"
        if vc_dir.exists():
            for f in vc_dir.glob("*.svg"):
                if f.stem.lower() == clean_slug:
                    return f

        return None

    def paintEvent(self, event):
        """Paints SVG or cyber high-contrast geometric fallback."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = float(self.width()), float(self.height())
        rect = QRectF(2, 2, w - 4, h - 4)

        # 1. Base Glassmorphic Background Card
        path = QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        painter.fillPath(path, QBrush(COLOR_BG))
        painter.setPen(QPen(COLOR_BORDER, 1.5))
        painter.drawPath(path)

        # 2. Render SVG Card if available
        if self.svg_renderer and self.svg_renderer.isValid():
            # Inset SVG to keep rounded card border
            inner_rect = QRectF(10, 10, w - 20, h - 20)
            self.svg_renderer.render(painter, inner_rect)
        else:
            # 3. Fallback High-Contrast Cyber Graphic
            self._paint_fallback_graphic(painter, w, h)

        painter.end()

    def _paint_fallback_graphic(self, painter: QPainter, w: float, h: float):
        """Renders stylized high-contrast geometric representation on canvas."""
        # Top Sign Title
        painter.setPen(COLOR_ACCENT)
        painter.setFont(QFont("SolaimanLipi", 22, QFont.Weight.Bold))
        painter.drawText(QRectF(10, 15, w - 20, 35), Qt.AlignmentFlag.AlignCenter, self.label_bn)

        # Subtitle
        painter.setPen(COLOR_TEXT_PRIMARY)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(QRectF(10, 50, w - 20, 25), Qt.AlignmentFlag.AlignCenter, self.label_en)

        # Central Stylized Hand Graphic
        cx, cy = w / 2.0, h / 2.0 + 20.0
        painter.setPen(QPen(COLOR_BORDER, 2))
        painter.setBrush(QBrush(QColor("#313244")))
        painter.drawEllipse(QRectF(cx - 35, cy - 35, 70, 70))

        # Finger rays / points
        painter.setPen(QPen(COLOR_EMERALD, 3))
        for angle_deg in [-40, -20, 0, 20, 40]:
            import math
            rad = math.radians(angle_deg - 90)
            x1 = cx + 25 * math.cos(rad)
            y1 = cy + 25 * math.sin(rad)
            x2 = cx + 45 * math.cos(rad)
            y2 = cy + 45 * math.sin(rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            painter.setBrush(QBrush(COLOR_ACCENT))
            painter.drawEllipse(QRectF(x2 - 3, y2 - 3, 6, 6))

        # Footnote
        painter.setPen(QColor("#94A3B8"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRectF(10, h - 25, w - 20, 20), Qt.AlignmentFlag.AlignCenter, "[BdSL Vector Card]")
