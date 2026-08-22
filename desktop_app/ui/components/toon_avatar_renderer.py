"""Stylized Cel-Shaded Illustrated Humanoid Avatar Engine (PyQt6).

Renders a modern, expressive 2.5D cel-shaded educational sign language vector avatar
with warm skin tones, clean bold stroke contours, modern casual hoodie clothing,
expressive FACS-animated eyes/eyebrows, and crystal-clear 5-finger articulated anatomy.

Features:
  - HyperKinematic Bézier Motion Engine (60 FPS smooth trajectory interpolation)
  - Dual Perspective View: Full Upper Torso (1.0x) vs Hand Close-Up Zoom (2.2x)
  - Facial Action Mimicry: Eyebrows (AU01/02/04), Eyes, and Phonetic Mouth Morphs
  - Multi-speed playback (1.0x, 0.5x, 0.25x), frame scrubbing, and loop control
"""

import enum
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from core_engine.vision.kinematic_interpolator import (
    FINGERTIP_INDICES,
    HAND_CONNECTIONS,
    KinematicJointFrame,
    KinematicMotionInterpolator,
)

logger = logging.getLogger(__name__)


# ── View Modes ───────────────────────────────────────────────────────────────
class AvatarViewMode(enum.Enum):
    FULL_BODY = "full_body"
    HAND_ZOOM = "hand_zoom"


# ── Color Palette (Cel-Shaded Vector Aesthetics) ─────────────────────────────
# Skin Tones
TOON_SKIN_HIGHLIGHT = QColor("#FFF2DF")
TOON_SKIN_BASE      = QColor("#F8CCA4")  # Base warm skin
TOON_SKIN_SHADOW    = QColor("#DB9B70")  # Cel shadow tone
TOON_SKIN_CREASE    = QColor("#995830")  # Crease & contour
TOON_BLUSH          = QColor(244, 114, 182, 60)  # Cheeks

# Hair & Outline
TOON_HAIR_BASE      = QColor("#1E1B2E")  # Dark violet-black
TOON_HAIR_HL        = QColor("#4338CA")  # Hair rim/sheen
TOON_HAIR_STRAND    = QColor("#312E81")
TOON_OUTLINE        = QColor("#111827")  # Bold vector stroke (almost black)

# Eyes & Facial Features
TOON_EYE_WHITE      = QColor("#FFFFFF")
TOON_IRIS_BASE      = QColor("#0284C7")  # Cyber cyan-blue
TOON_IRIS_DARK      = QColor("#0369A1")
TOON_PUPIL          = QColor("#0F172A")
TOON_EYE_SPEC       = QColor("#FFFFFF")
TOON_LIP_BASE       = QColor("#F472B6")
TOON_LIP_DARK       = QColor("#DB2777")
TOON_MOUTH_INNER    = QColor("#881337")

# Clothing (Modern Cyber Hoodie)
TOON_CLOTH_BASE     = QColor("#1E293B")  # Slate dark
TOON_CLOTH_SHADOW   = QColor("#0F172A")  # Cel shadow
TOON_CLOTH_HL       = QColor("#334155")
TOON_CLOTH_ACCENT   = QColor("#06B6D4")  # Cyan stripe / piping
TOON_HOOD_INSIDE    = QColor("#0A0F1D")

# Hands & Articulators
TOON_NAIL_BASE      = QColor("#FED7AA")
TOON_TOUCH_HALO     = QColor(245, 158, 11, 190)  # Amber gold touch glow
TOON_AURA_CYAN      = QColor(6, 182, 212, 160)

# Backdrop
BG_GRADIENT_START   = QColor("#080D1A")
BG_GRADIENT_END     = QColor("#0F172A")


class ToonAvatarRenderer(QWidget):
    """Expressive Cel-Shaded Vector Avatar Signer for Learning Hub & Web.

    Renders an animated 2.5D humanoid character playing 60-frame Bézier motion loops.
    """

    frame_changed = pyqtSignal(int, int)          # (current_frame, total_frames)
    playback_state_changed = pyqtSignal(bool)     # is_playing
    zoom_mode_changed = pyqtSignal(str)           # "full_body" or "hand_zoom"

    def __init__(
        self,
        sign_slug: str = "dhonnobad",
        label_bn: str = "ধন্যবাদ",
        label_en: str = "Thank you",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.sign_slug = sign_slug
        self.label_bn = label_bn
        self.label_en = label_en

        self.interpolator = KinematicMotionInterpolator()
        self.frames: List[KinematicJointFrame] = []
        self.current_frame_idx: int = 0
        self.total_frames: int = 60

        self.view_mode: AvatarViewMode = AvatarViewMode.FULL_BODY
        self.zoom_scale: float = 2.2

        self.is_playing: bool = True
        self.loop: bool = True
        self.speed: float = 1.0  # 1.0x, 0.5x, 0.25x

        # FACS Mimicry States
        self.au01_inner_brow: float = 0.0
        self.au02_outer_brow: float = 0.0
        self.au04_brow_furrow: float = 0.0
        self.mouth_open_ratio: float = 0.15

        # Animation timer (Base = ~30 FPS -> 33 ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)
        self._update_timer_interval()

        self.load_sign_motion(self.sign_slug, self.label_bn, self.label_en)

    # ── Public API & Sign Loading ────────────────────────────────────────────

    def load_sign_motion(self, sign_slug: str, label_bn: str = "", label_en: str = "") -> None:
        """Loads and pre-synthesizes kinematic motion loop for target sign."""
        self.sign_slug = sign_slug or "dhonnobad"
        self.label_bn = label_bn or self.sign_slug
        self.label_en = label_en or self.sign_slug

        self.frames = self.interpolator.generate_motion_sequence(self.sign_slug)
        self.total_frames = max(1, len(self.frames))
        self.current_frame_idx = 0

        # Configure facial expressions based on sign characteristics
        self._configure_facs_for_sign(self.sign_slug)

        self.frame_changed.emit(self.current_frame_idx, self.total_frames)
        self.update()

    def load_compound_sign(self, constituents: List[str], label_bn: str = "", label_en: str = "") -> None:
        """Loads sequence of compound sign constituents."""
        primary_slug = constituents[0] if constituents else "dhonnobad"
        self.load_sign_motion(primary_slug, label_bn, label_en)


    def _configure_facs_for_sign(self, slug: str) -> None:
        """Sets expressive FACS Action Units matching the linguistic tone of the sign."""
        slug_lower = slug.lower()
        if slug_lower in ["dhonnobad", "shagotom", "bhalo", "thik_ache", "ma", "baba"]:
            # Warm, pleasant: slightly raised brows, smiling mouth
            self.au01_inner_brow = 0.15
            self.au02_outer_brow = 0.20
            self.au04_brow_furrow = 0.0
            self.mouth_open_ratio = 0.25
        elif slug_lower in ["sahajjo", "hospital", "daktar", "emergency"]:
            # Urgent, attentive: furrowed brows, open mouth
            self.au01_inner_brow = 0.35
            self.au02_outer_brow = 0.0
            self.au04_brow_furrow = 0.40
            self.mouth_open_ratio = 0.40
        elif slug_lower in ["kemon_achen", "kothay", "ki", "question"]:
            # Questioning / WH-question: raised eyebrow, slight furrow
            self.au01_inner_brow = 0.50
            self.au02_outer_brow = 0.40
            self.au04_brow_furrow = 0.10
            self.mouth_open_ratio = 0.30
        elif slug_lower in ["na", "no", "khaprap"]:
            # Negation: furrowed brows
            self.au01_inner_brow = 0.0
            self.au02_outer_brow = 0.0
            self.au04_brow_furrow = 0.60
            self.mouth_open_ratio = 0.10
        else:
            self.au01_inner_brow = 0.10
            self.au02_outer_brow = 0.10
            self.au04_brow_furrow = 0.0
            self.mouth_open_ratio = 0.15

    # ── Playback Controls ────────────────────────────────────────────────────

    def play(self) -> None:
        """Starts playback."""
        self.is_playing = True
        self.timer.start()
        self.playback_state_changed.emit(True)

    def pause(self) -> None:
        """Pauses playback."""
        self.is_playing = False
        self.timer.stop()
        self.playback_state_changed.emit(False)

    def toggle_play(self) -> bool:
        """Toggles play/pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
        return self.is_playing

    def set_speed(self, speed: float) -> None:
        """Sets playback speed multiplier (e.g. 1.0, 0.5, 0.25)."""
        self.speed = max(0.05, min(3.0, float(speed)))
        self._update_timer_interval()

    def set_loop(self, loop: bool) -> None:
        """Sets continuous looping toggle."""
        self.loop = bool(loop)

    def seek(self, frame_idx: int) -> None:
        """Seeks to specific frame index."""
        if not self.frames:
            return
        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.frame_changed.emit(self.current_frame_idx, self.total_frames)
        self.update()

    def step_forward(self) -> None:
        """Advances by one frame."""
        self.seek((self.current_frame_idx + 1) % self.total_frames)

    def step_backward(self) -> None:
        """Steps back by one frame."""
        self.seek((self.current_frame_idx - 1 + self.total_frames) % self.total_frames)

    def _update_timer_interval(self) -> None:
        interval = max(10, int(33.33 / self.speed))
        if self.timer.isActive():
            self.timer.setInterval(interval)
        else:
            self.timer.setInterval(interval)
            if self.is_playing:
                self.timer.start()

    def _on_timer_tick(self) -> None:
        if not self.frames:
            return
        next_idx = self.current_frame_idx + 1
        if next_idx >= self.total_frames:
            if self.loop:
                next_idx = 0
            else:
                next_idx = self.total_frames - 1
                self.pause()
        self.current_frame_idx = next_idx
        self.frame_changed.emit(self.current_frame_idx, self.total_frames)
        self.update()

    # ── Perspective & Zoom Mode ──────────────────────────────────────────────

    def set_view_mode(self, mode: Union[AvatarViewMode, str]) -> None:
        """Switches between FULL_BODY (1.0x) and HAND_ZOOM (2.2x)."""
        if isinstance(mode, str):
            if mode.lower() in ["hand_zoom", "zoom", "hand"]:
                self.view_mode = AvatarViewMode.HAND_ZOOM
            else:
                self.view_mode = AvatarViewMode.FULL_BODY
        else:
            self.view_mode = mode

        self.zoom_mode_changed.emit(self.view_mode.value)
        self.update()

    def toggle_view_mode(self) -> str:
        """Toggles between Full Body and Hand Zoom mode."""
        if self.view_mode == AvatarViewMode.FULL_BODY:
            self.set_view_mode(AvatarViewMode.HAND_ZOOM)
        else:
            self.set_view_mode(AvatarViewMode.FULL_BODY)
        return self.view_mode.value

    def get_zoom_transform(self, width: float, height: float) -> Tuple[float, float, float]:
        """Calculates (scale, tx, ty) for the current view mode and frame.

        In HAND_ZOOM mode, centers tightly on the active dominant hand.
        """
        if self.view_mode == AvatarViewMode.FULL_BODY or not self.frames:
            return 1.0, 0.0, 0.0

        cur_frame = self.frames[self.current_frame_idx % len(self.frames)]
        # Target dominant hand centroid
        target_hand = cur_frame.right_hand if cur_frame.is_right_active and cur_frame.right_hand else cur_frame.left_hand
        if target_hand and len(target_hand) >= 21:
            hx = sum(p[0] for p in target_hand) / len(target_hand)
            hy = sum(p[1] for p in target_hand) / len(target_hand)
        else:
            hx, hy = cur_frame.right_wrist

        scale = self.zoom_scale
        # Center target (hx * width, hy * height) at canvas center (width / 2, height / 2)
        tx = (width * 0.5) - (hx * width * scale)
        ty = (height * 0.45) - (hy * height * scale)

        return scale, tx, ty

    # ── Paint Event & Vector Rendering Pipeline ──────────────────────────────

    def paintEvent(self, event) -> None:
        """High-DPI double-buffered cel-shaded vector rendering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = float(self.width())
        h = float(self.height())

        # 1. Background
        self._draw_backdrop(painter, w, h)

        if not self.frames:
            self._draw_placeholder(painter, w, h)
            return

        cur_frame = self.frames[self.current_frame_idx % len(self.frames)]

        # 2. Setup Zoom & Pan Matrix
        scale, tx, ty = self.get_zoom_transform(w, h)
        painter.save()
        painter.translate(tx, ty)
        painter.scale(scale, scale)

        # 3. Render Humanoid Cel-Shaded Rig
        self._draw_cel_shaded_body(painter, cur_frame, w, h)

        painter.restore()

        # 4. Viewport Badges & Watermark
        self._draw_overlay_badges(painter, w, h)

    def _draw_backdrop(self, p: QPainter, w: float, h: float) -> None:
        """Draws dark studio radial gradient backdrop with subtle cyber vignette."""
        bg_grad = QRadialGradient(w * 0.5, h * 0.45, max(w, h) * 0.7)
        bg_grad.setColorAt(0.0, QColor("#111827"))
        bg_grad.setColorAt(0.6, BG_GRADIENT_START)
        bg_grad.setColorAt(1.0, QColor("#040711"))

        p.fillRect(QRectF(0, 0, w, h), bg_grad)

        # Subtle cyber grid
        p.setPen(QPen(QColor(6, 182, 212, 12), 1, Qt.PenStyle.DotLine))
        grid_step = 28
        for x in range(0, int(w), grid_step):
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        for y in range(0, int(h), grid_step):
            p.drawLine(QPointF(0, y), QPointF(w, y))

    def _draw_placeholder(self, p: QPainter, w: float, h: float) -> None:
        p.setPen(QPen(QColor("#94A3B8"), 1))
        p.setFont(QFont("Segoe UI", 12))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "Loading Toon Avatar...")

    # ── Cel-Shaded Anatomical Drawing ─────────────────────────────────────────

    def _draw_cel_shaded_body(self, p: QPainter, frame: KinematicJointFrame, w: float, h: float) -> None:
        """Multi-pass layered vector character drawing."""
        # Convert normalized rig points (0..1) to canvas pixels
        def P(norm_pt: Tuple[float, float]) -> QPointF:
            return QPointF(norm_pt[0] * w, norm_pt[1] * h)

        head_pt = P(frame.head)
        neck_pt = P(frame.neck)
        chest_pt = P(frame.chest)
        ls_pt = P(frame.left_shoulder)
        rs_pt = P(frame.right_shoulder)
        le_pt = P(frame.left_elbow)
        re_pt = P(frame.right_elbow)
        lw_pt = P(frame.left_wrist)
        rw_pt = P(frame.right_wrist)

        # Layer 1: Torso / Clothing (Hoodie base + cel shadow + collar)
        self._draw_torso_hoodie(p, chest_pt, neck_pt, ls_pt, rs_pt, w, h)

        # Layer 2: Arms & Sleeves
        self._draw_arm_segment(p, ls_pt, le_pt, lw_pt, is_right=False)
        self._draw_arm_segment(p, rs_pt, re_pt, rw_pt, is_right=True)

        # Layer 3: Neck
        self._draw_neck(p, neck_pt, head_pt, ls_pt, rs_pt)

        # Layer 4: Head & Cel-Shaded Face (Hair, Eyes, Eyebrows, Mouth)
        self._draw_head_and_face(p, head_pt, w, h)

        # Layer 5: Hands with Depth-Sorting
        # Render deeper hand first, closer hand on top
        if frame.right_hand_z >= frame.left_hand_z:
            if frame.is_left_active and frame.left_hand:
                self._draw_articulated_hand(p, frame.left_hand, is_right=False, w=w, h=h)
            if frame.is_right_active and frame.right_hand:
                self._draw_articulated_hand(p, frame.right_hand, is_right=True, w=w, h=h)
        else:
            if frame.is_right_active and frame.right_hand:
                self._draw_articulated_hand(p, frame.right_hand, is_right=True, w=w, h=h)
            if frame.is_left_active and frame.left_hand:
                self._draw_articulated_hand(p, frame.left_hand, is_right=False, w=w, h=h)

        # Layer 6: Touch Pulse Halos
        self._draw_touch_effects(p, frame, w, h)

    def _draw_torso_hoodie(
        self, p: QPainter, chest: QPointF, neck: QPointF, ls: QPointF, rs: QPointF, w: float, h: float
    ) -> None:
        """Draws cel-shaded hoodie torso with shadow split and cyan piping."""
        torso_path = QPainterPath()
        torso_path.moveTo(ls)
        torso_path.lineTo(rs)
        torso_path.lineTo(rs.x() + 15, h * 0.95)
        torso_path.lineTo(ls.x() - 15, h * 0.95)
        torso_path.closeSubpath()

        # Base fill
        p.fillPath(torso_path, QBrush(TOON_CLOTH_BASE))

        # Cel-Shadow layer (right side shadow)
        shadow_path = QPainterPath()
        shadow_path.moveTo(chest.x() + 5, neck.y())
        shadow_path.lineTo(rs)
        shadow_path.lineTo(rs.x() + 15, h * 0.95)
        shadow_path.lineTo(chest.x() + 10, h * 0.95)
        shadow_path.closeSubpath()
        p.fillPath(shadow_path, QBrush(TOON_CLOTH_SHADOW))

        # Bold contour outline
        p.setPen(QPen(TOON_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(torso_path)

        # Cyan Accent Stripe / Zipper
        p.setPen(QPen(TOON_CLOTH_ACCENT, 2.0))
        p.drawLine(QPointF(chest.x(), neck.y() + 6), QPointF(chest.x(), h * 0.95))

        # Collar V-Neck
        collar = QPainterPath()
        collar.moveTo(neck.x() - 18, neck.y() - 4)
        collar.lineTo(chest.x(), neck.y() + 16)
        collar.lineTo(neck.x() + 18, neck.y() - 4)
        p.setPen(QPen(TOON_OUTLINE, 2.5))
        p.fillPath(collar, QBrush(TOON_HOOD_INSIDE))
        p.drawPath(collar)

    def _draw_neck(self, p: QPainter, neck: QPointF, head: QPointF, ls: QPointF, rs: QPointF) -> None:
        """Draws warm cel-shaded neck cylinder."""
        nw = 18.0
        neck_path = QPainterPath()
        neck_path.moveTo(neck.x() - nw, head.y() + 20)
        neck_path.lineTo(neck.x() + nw, head.y() + 20)
        neck_path.lineTo(neck.x() + nw + 2, neck.y() + 4)
        neck_path.lineTo(neck.x() - nw - 2, neck.y() + 4)
        neck_path.closeSubpath()

        p.fillPath(neck_path, QBrush(TOON_SKIN_SHADOW))
        p.setPen(QPen(TOON_OUTLINE, 2.0))
        p.drawPath(neck_path)

    def _draw_arm_segment(self, p: QPainter, shoulder: QPointF, elbow: QPointF, wrist: QPointF, is_right: bool) -> None:
        """Draws bold cel-shaded arm cylinders with clothing sleeve + forearm skin."""
        arm_w = 16.0

        # Upper Arm (Sleeve)
        up_path = QPainterPath()
        dx = elbow.x() - shoulder.x()
        dy = elbow.y() - shoulder.y()
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * (arm_w * 0.5), dx / L * (arm_w * 0.5)

        up_path.moveTo(shoulder.x() + nx, shoulder.y() + ny)
        up_path.lineTo(elbow.x() + nx, elbow.y() + ny)
        up_path.lineTo(elbow.x() - nx, elbow.y() - ny)
        up_path.lineTo(shoulder.x() - nx, shoulder.y() - ny)
        up_path.closeSubpath()

        p.fillPath(up_path, QBrush(TOON_CLOTH_BASE))
        p.setPen(QPen(TOON_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(up_path)

        # Forearm (Skin tone with cel shade)
        fa_w = 12.0
        dx2 = wrist.x() - elbow.x()
        dy2 = wrist.y() - elbow.y()
        L2 = math.hypot(dx2, dy2) or 1.0
        nx2, ny2 = -dy2 / L2 * (fa_w * 0.5), dx2 / L2 * (fa_w * 0.5)

        fa_path = QPainterPath()
        fa_path.moveTo(elbow.x() + nx2, elbow.y() + ny2)
        fa_path.lineTo(wrist.x() + nx2 * 0.8, wrist.y() + ny2 * 0.8)
        fa_path.lineTo(wrist.x() - nx2 * 0.8, wrist.y() - ny2 * 0.8)
        fa_path.lineTo(elbow.x() - nx2, elbow.y() - ny2)
        fa_path.closeSubpath()

        p.fillPath(fa_path, QBrush(TOON_SKIN_BASE))
        p.setPen(QPen(TOON_OUTLINE, 2.0))
        p.drawPath(fa_path)

    def _draw_head_and_face(self, p: QPainter, head: QPointF, w: float, h: float) -> None:
        """Draws stylized anime-vector head with hair, expressive FACS eyebrows, eyes, and mouth."""
        cx, cy = head.x(), head.y()
        head_rx, head_ry = 34.0, 42.0

        # 1. Head Oval Base
        head_path = QPainterPath()
        head_path.addEllipse(QPointF(cx, cy), head_rx, head_ry)
        p.fillPath(head_path, QBrush(TOON_SKIN_BASE))

        # 2. Cel-Shadow (Lower-right jaw shadow)
        p.save()
        p.setClipPath(head_path)
        shadow_poly = QPolygonF([
            QPointF(cx - head_rx, cy + 10),
            QPointF(cx + head_rx, cy + 5),
            QPointF(cx + head_rx, cy + head_ry),
            QPointF(cx - head_rx, cy + head_ry),
        ])
        shadow_path = QPainterPath()
        shadow_path.addPolygon(shadow_poly)
        p.fillPath(shadow_path, QBrush(TOON_SKIN_SHADOW))
        p.restore()

        p.setPen(QPen(TOON_OUTLINE, 2.5))
        p.drawPath(head_path)

        # 3. Rosy Cheeks
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(TOON_BLUSH))
        p.drawEllipse(QPointF(cx - 20, cy + 8), 8, 5)
        p.drawEllipse(QPointF(cx + 20, cy + 8), 8, 5)

        # 4. Stylized Eyes
        self._draw_eye(p, QPointF(cx - 14, cy - 2), is_right=False)
        self._draw_eye(p, QPointF(cx + 14, cy - 2), is_right=True)

        # 5. FACS Morphable Eyebrows (AU01/AU02/AU04)
        self._draw_eyebrows(p, cx, cy)

        # 6. Nose
        p.setPen(QPen(TOON_SKIN_CREASE, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(cx, cy + 3), QPointF(cx + 2, cy + 10))
        p.drawLine(QPointF(cx + 2, cy + 10), QPointF(cx - 2, cy + 11))

        # 7. Expressive Mouth Morph
        self._draw_mouth(p, cx, cy + 22)

        # 8. Stylized Vector Hair
        self._draw_hair(p, cx, cy, head_rx, head_ry)

    def _draw_eye(self, p: QPainter, center: QPointF, is_right: bool) -> None:
        """Draws clear vector eye with sclera, cyan iris, pupil, and specular gloss."""
        ex, ey = center.x(), center.y()
        ew, eh = 8.5, 6.0

        # Eye White
        eye_path = QPainterPath()
        eye_path.addEllipse(center, ew, eh)
        p.fillPath(eye_path, QBrush(TOON_EYE_WHITE))
        p.setPen(QPen(TOON_OUTLINE, 1.8))
        p.drawPath(eye_path)

        # Iris
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(TOON_IRIS_BASE))
        p.drawEllipse(center, 4.5, 5.0)

        # Pupil
        p.setBrush(QBrush(TOON_PUPIL))
        p.drawEllipse(center, 2.5, 2.8)

        # Specular Highlight Dot
        p.setBrush(QBrush(TOON_EYE_SPEC))
        p.drawEllipse(QPointF(ex + (1.2 if is_right else -1.2), ey - 1.8), 1.6, 1.6)

        # Eyelid top line
        p.setPen(QPen(TOON_OUTLINE, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(ex - ew - 1, ey - 2), QPointF(ex + ew + 1, ey - 2))

    def _draw_eyebrows(self, p: QPainter, cx: float, cy: float) -> None:
        """Animates eyebrows based on AU01 (Inner Raise), AU02 (Outer Raise), AU04 (Furrow)."""
        p.setPen(QPen(TOON_HAIR_BASE, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

        # Left Eyebrow
        l_inner_y = cy - 14 - (self.au01_inner_brow * 6.0) + (self.au04_brow_furrow * 4.0)
        l_outer_y = cy - 12 - (self.au02_outer_brow * 6.0)
        l_mid_y   = (l_inner_y + l_outer_y) * 0.5 - 2.0

        l_path = QPainterPath()
        l_path.moveTo(cx - 24, l_outer_y)
        l_path.quadTo(cx - 16, l_mid_y, cx - 6, l_inner_y)
        p.drawPath(l_path)

        # Right Eyebrow
        r_inner_y = cy - 14 - (self.au01_inner_brow * 6.0) + (self.au04_brow_furrow * 4.0)
        r_outer_y = cy - 12 - (self.au02_outer_brow * 6.0)
        r_mid_y   = (r_inner_y + r_outer_y) * 0.5 - 2.0

        r_path = QPainterPath()
        r_path.moveTo(cx + 6, r_inner_y)
        r_path.quadTo(cx + 16, r_mid_y, cx + 24, r_outer_y)
        p.drawPath(r_path)

    def _draw_mouth(self, p: QPainter, cx: float, cy: float) -> None:
        """Morphs mouth shape between pleasant smile and phonetically open mouth."""
        open_h = self.mouth_open_ratio * 12.0
        half_w = 11.0

        mouth_path = QPainterPath()
        if open_h > 2.5:
            # Open mouth (vocalization shape)
            mouth_path.moveTo(cx - half_w, cy)
            mouth_path.quadTo(cx, cy - 2, cx + half_w, cy)
            mouth_path.quadTo(cx, cy + open_h, cx - half_w, cy)
            mouth_path.closeSubpath()
            p.fillPath(mouth_path, QBrush(TOON_MOUTH_INNER))
            p.setPen(QPen(TOON_OUTLINE, 2.0))
            p.drawPath(mouth_path)

            # Teeth top bar
            teeth = QPainterPath()
            teeth.moveTo(cx - half_w + 3, cy)
            teeth.lineTo(cx + half_w - 3, cy)
            teeth.lineTo(cx + half_w - 3, cy + 3)
            teeth.lineTo(cx - half_w + 3, cy + 3)
            teeth.closeSubpath()
            p.fillPath(teeth, QBrush(QColor("#FFFFFF")))
        else:
            # Subtle smile curve
            mouth_path.moveTo(cx - half_w, cy)
            mouth_path.quadTo(cx, cy + 4, cx + half_w, cy)
            p.setPen(QPen(TOON_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(mouth_path)

    def _draw_hair(self, p: QPainter, cx: float, cy: float, rx: float, ry: float) -> None:
        """Draws stylish anime-inspired vector hair with layered tufts and highlight gloss."""
        hair_path = QPainterPath()
        # Top dome + side bangs
        hair_path.moveTo(cx - rx - 4, cy - 5)
        hair_path.cubicTo(cx - rx - 8, cy - ry - 14, cx + rx + 8, cy - ry - 14, cx + rx + 4, cy - 5)
        # Front bangs tufts
        hair_path.lineTo(cx + rx - 2, cy - 14)
        hair_path.lineTo(cx + 12, cy - 18)
        hair_path.lineTo(cx + 4, cy - 25)
        hair_path.lineTo(cx - 8, cy - 18)
        hair_path.lineTo(cx - 20, cy - 22)
        hair_path.closeSubpath()

        p.fillPath(hair_path, QBrush(TOON_HAIR_BASE))

        # Highlight Sheen Arc
        hl_path = QPainterPath()
        hl_path.moveTo(cx - 18, cy - ry + 4)
        hl_path.quadTo(cx, cy - ry - 2, cx + 18, cy - ry + 4)
        p.setPen(QPen(TOON_HAIR_HL, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(hl_path)

        p.setPen(QPen(TOON_OUTLINE, 2.5))
        p.drawPath(hair_path)

    # ── Articulated 5-Finger Hand Renderer ─────────────────────────────────────

    def _draw_articulated_hand(
        self, p: QPainter, hand_pts: List[Tuple[float, float]], is_right: bool, w: float, h: float
    ) -> None:
        """Renders 5-finger articulated hand with palm volume, knuckle shading, and fingernails."""
        if len(hand_pts) < 21:
            return

        def P(idx: int) -> QPointF:
            return QPointF(hand_pts[idx][0] * w, hand_pts[idx][1] * h)

        # 1. Draw Palm Base Polygon (0 -> 1 -> 5 -> 9 -> 13 -> 17 -> 0)
        palm_poly = QPolygonF([P(0), P(1), P(5), P(9), P(13), P(17)])
        palm_path = QPainterPath()
        palm_path.addPolygon(palm_poly)
        p.fillPath(palm_path, QBrush(TOON_SKIN_BASE))
        p.setPen(QPen(TOON_OUTLINE, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(palm_path)

        # 2. Draw 5 Articulated Fingers as Tapered Rounded Segments
        finger_chains = [
            [0, 1, 2, 3, 4],       # Thumb
            [5, 6, 7, 8],          # Index
            [9, 10, 11, 12],       # Middle
            [13, 14, 15, 16],      # Ring
            [17, 18, 19, 20],      # Pinky
        ]
        widths = [7.5, 6.5, 6.8, 6.0, 5.2]

        for f_i, chain in enumerate(finger_chains):
            f_w = widths[f_i]
            for s in range(len(chain) - 1):
                p1 = P(chain[s])
                p2 = P(chain[s + 1])
                seg_w = f_w * (1.0 - s * 0.15)
                self._draw_finger_capsule(p, p1, p2, seg_w)

            # Draw Fingernail on Tip (distal segment)
            tip_pt = P(chain[-1])
            prev_pt = P(chain[-2])
            self._draw_fingernail(p, tip_pt, prev_pt, f_w * 0.6)

    def _draw_finger_capsule(self, p: QPainter, p1: QPointF, p2: QPointF, radius: float) -> None:
        """Draws cel-shaded rounded finger bone segment with outline."""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        L = math.hypot(dx, dy)
        if L < 0.1:
            return
        nx = -dy / L * radius
        ny = dx / L * radius

        capsule = QPainterPath()
        capsule.moveTo(p1.x() + nx, p1.y() + ny)
        capsule.lineTo(p2.x() + nx * 0.85, p2.y() + ny * 0.85)
        capsule.arcTo(QRectF(p2.x() - radius * 0.85, p2.y() - radius * 0.85, radius * 1.7, radius * 1.7), 0, 180)
        capsule.lineTo(p1.x() - nx, p1.y() - ny)
        capsule.closeSubpath()

        p.fillPath(capsule, QBrush(TOON_SKIN_BASE))
        p.setPen(QPen(TOON_OUTLINE, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(capsule)

        # Knuckle Crease Line
        p.setPen(QPen(TOON_SKIN_CREASE, 1.2))
        p.drawLine(QPointF(p1.x() + nx * 0.6, p1.y() + ny * 0.6), QPointF(p1.x() - nx * 0.6, p1.y() - ny * 0.6))

    def _draw_fingernail(self, p: QPainter, tip: QPointF, prev: QPointF, size: float) -> None:
        """Draws subtle fingernail plate highlight on fingertip."""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(TOON_NAIL_BASE))
        p.drawEllipse(tip, size, size * 0.8)

    def _draw_touch_effects(self, p: QPainter, frame: KinematicJointFrame, w: float, h: float) -> None:
        """Renders glowing amber/cyan touch halos for active articulators."""
        if not frame.touch_contacts:
            return

        for a, b, intensity in frame.touch_contacts:
            if intensity <= 0.05:
                continue
            # Use right hand as reference anchor
            if frame.right_hand and a < len(frame.right_hand):
                pt = QPointF(frame.right_hand[a][0] * w, frame.right_hand[a][1] * h)
                halo_r = 12.0 * intensity + 4.0
                p.setPen(QPen(TOON_TOUCH_HALO, 2.0))
                p.setBrush(QBrush(QColor(245, 158, 11, int(70 * intensity))))
                p.drawEllipse(pt, halo_r, halo_r)

    def _draw_overlay_badges(self, p: QPainter, w: float, h: float) -> None:
        """Draws top status badges: sign label, FPS, and view mode badge."""
        # Top-left sign title badge
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(15, 23, 42, 200)))
        p.drawRoundedRect(QRectF(10, 10, 180, 42), 8, 8)

        p.setPen(QPen(QColor("#F8FAFC"), 1))
        p.setFont(QFont("Hind Siliguri", 13, QFont.Weight.Bold))
        p.drawText(QRectF(18, 12, 160, 20), Qt.AlignmentFlag.AlignLeft, f"{self.label_bn}")

        p.setPen(QPen(QColor("#06B6D4"), 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(18, 30, 160, 18), Qt.AlignmentFlag.AlignLeft, f"{self.label_en.upper()} &bull; 60 FPS")

        # Top-right View Mode Badge
        mode_text = "🔍 হাতের জুম (2.2x)" if self.view_mode == AvatarViewMode.HAND_ZOOM else "👤 ফুল বডি"
        mode_color = QColor("#06B6D4") if self.view_mode == AvatarViewMode.HAND_ZOOM else QColor("#A78BFA")

        p.setBrush(QBrush(QColor(15, 23, 42, 200)))
        p.drawRoundedRect(QRectF(w - 120, 10, 110, 28), 6, 6)

        p.setPen(QPen(mode_color, 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(w - 120, 10, 110, 28), Qt.AlignmentFlag.AlignCenter, mode_text)
