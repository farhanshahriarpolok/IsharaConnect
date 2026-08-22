"""Stylized Cel-Vector 2D Humanoid Signer Illustration Engine (PyQt6).

Renders a pristine, high-DPI 2D cel-shaded educational sign language vector avatar:
  - Dark bob-cut hair with layered bangs (#1E1E24)
  - Warm natural skin tones with clean contours (#E5A882 / #D48B68)
  - Expressive FACS-animated vector eyes, brows (AU01/AU02/AU04), and defined lips (#D97757)
  - Slate-blue collared button-down shirt (#5B7C99 / #4A6984) with line-art collar flaps (#1A2530)
  - Segmented 5-finger articulated vector hands with knuckle creases and bold stroke isolation

Features:
  - HyperKinematic Bézier Motion Engine (60 FPS smooth trajectory interpolation)
  - Dual Perspective View: Full Upper Torso (1.0x) vs Hand Close-Up Zoom (2.2x)
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


# ── Color Palette (Pristine 2D Vector Cel Aesthetics) ────────────────────────
# Studio Background
COLOR_BG_STUDIO_TOP = QColor("#2B394E")
COLOR_BG_STUDIO_BTM = QColor("#1E293B")
COLOR_BG_GRID       = QColor(56, 189, 248, 12)

# Skin Tones
COLOR_SKIN_HIGHLIGHT = QColor("#FCE4D0")  # Soft specular
COLOR_SKIN_BASE      = QColor("#E5A882")  # Warm natural skin fill
COLOR_SKIN_SHADOW    = QColor("#D48B68")  # Cel shadow tone
COLOR_SKIN_CREASE    = QColor("#995830")  # Crease & contour
COLOR_CHEEK_BLUSH    = QColor(244, 114, 182, 55)

# Hair & Contours
COLOR_HAIR_BASE      = QColor("#1E1E24")  # Dark bob-cut black-brown
COLOR_HAIR_HL        = QColor("#3E3E4C")  # Layered sheen / highlight
COLOR_OUTLINE        = QColor("#1A2530")  # Bold line-art vector stroke (2.5px)

# Eyes & Facial Features
COLOR_EYE_WHITE      = QColor("#FFFFFF")
COLOR_IRIS_BASE      = QColor("#0284C7")  # Expressive cyan-blue
COLOR_IRIS_DARK      = QColor("#0369A1")
COLOR_PUPIL          = QColor("#0F172A")
COLOR_EYE_SPEC       = QColor("#FFFFFF")
COLOR_LIP_BASE       = QColor("#D97757")  # Well-defined terracotta lips
COLOR_LIP_DARK       = QColor("#B45309")
COLOR_MOUTH_INNER    = QColor("#881337")

# Costume: Slate-Blue Collared Button-Down Shirt
COLOR_SHIRT_BASE     = QColor("#5B7C99")  # Slate-blue shirt
COLOR_SHIRT_SHADOW   = QColor("#4A6984")  # Cel shadow fold
COLOR_SHIRT_PLACKET  = QColor("#3D5970")  # Button placket
COLOR_SHIRT_BUTTON   = QColor("#F8FAFC")  # Clean white buttons
COLOR_COLLAR_BASE    = QColor("#6688A6")  # Crisp collar flaps

# Hands & Articulators
COLOR_NAIL_BASE      = QColor("#FED7AA")
COLOR_TOUCH_HALO     = QColor(245, 158, 11, 190)  # Amber touch aura
COLOR_AURA_CYAN      = QColor(6, 182, 212, 160)


class ToonAvatarRenderer(QWidget):
    """Ultra-Clean 2D Cel-Vector Humanoid Signer Widget.

    Renders a high-DPI vector character playing 60 FPS Bézier motion sequences.
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

        # FACS Mimicry Parameters
        self.au01_inner_brow: float = 0.0
        self.au02_outer_brow: float = 0.0
        self.au04_brow_furrow: float = 0.0
        self.mouth_open_ratio: float = 0.15

        # Animation timer (Base = ~30 FPS -> 33.3 ms)
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

        self._configure_facs_for_sign(self.sign_slug)

        self.frame_changed.emit(self.current_frame_idx, self.total_frames)
        self.update()

    def load_compound_sign(self, constituents: List[str], label_bn: str = "", label_en: str = "") -> None:
        """Loads sequence of compound sign constituents."""
        primary_slug = constituents[0] if constituents else "dhonnobad"
        self.load_sign_motion(primary_slug, label_bn, label_en)

    def _configure_facs_for_sign(self, slug: str) -> None:
        """Configures expressive FACS Action Units matching sign phonetics and tone."""
        slug_lower = slug.lower()
        if slug_lower in ["dhonnobad", "shagotom", "bhalo", "thik_ache", "ma", "baba"]:
            self.au01_inner_brow = 0.15
            self.au02_outer_brow = 0.20
            self.au04_brow_furrow = 0.0
            self.mouth_open_ratio = 0.25
        elif slug_lower in ["sahajjo", "hospital", "daktar", "emergency"]:
            self.au01_inner_brow = 0.35
            self.au02_outer_brow = 0.0
            self.au04_brow_furrow = 0.40
            self.mouth_open_ratio = 0.40
        elif slug_lower in ["kemon_achen", "kothay", "ki", "question"]:
            self.au01_inner_brow = 0.50
            self.au02_outer_brow = 0.40
            self.au04_brow_furrow = 0.10
            self.mouth_open_ratio = 0.30
        elif slug_lower in ["na", "no", "khaprap"]:
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
        self.is_playing = True
        self.timer.start()
        self.playback_state_changed.emit(True)

    def pause(self) -> None:
        self.is_playing = False
        self.timer.stop()
        self.playback_state_changed.emit(False)

    def toggle_play(self) -> bool:
        if self.is_playing:
            self.pause()
        else:
            self.play()
        return self.is_playing

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.05, min(3.0, float(speed)))
        self._update_timer_interval()

    def set_loop(self, loop: bool) -> None:
        self.loop = bool(loop)

    def seek(self, frame_idx: int) -> None:
        if not self.frames:
            return
        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.frame_changed.emit(self.current_frame_idx, self.total_frames)
        self.update()

    def step_forward(self) -> None:
        self.seek((self.current_frame_idx + 1) % self.total_frames)

    def step_backward(self) -> None:
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
        if self.view_mode == AvatarViewMode.FULL_BODY:
            self.set_view_mode(AvatarViewMode.HAND_ZOOM)
        else:
            self.set_view_mode(AvatarViewMode.FULL_BODY)
        return self.view_mode.value

    def get_zoom_transform(self, width: float, height: float) -> Tuple[float, float, float]:
        """Calculates (scale, tx, ty) for the current view mode and frame."""
        if self.view_mode == AvatarViewMode.FULL_BODY or not self.frames:
            return 1.0, 0.0, 0.0

        cur_frame = self.frames[self.current_frame_idx % len(self.frames)]
        target_hand = cur_frame.right_hand if cur_frame.is_right_active and cur_frame.right_hand else cur_frame.left_hand
        if target_hand and len(target_hand) >= 21:
            hx = sum(p[0] for p in target_hand) / len(target_hand)
            hy = sum(p[1] for p in target_hand) / len(target_hand)
        else:
            hx, hy = cur_frame.right_wrist

        scale = self.zoom_scale
        tx = (width * 0.5) - (hx * width * scale)
        ty = (height * 0.45) - (hy * height * scale)

        return scale, tx, ty

    # ── Vector Path Helper Methods (Exposed for Testing) ──────────────────────

    @staticmethod
    def get_head_path(cx: float, cy: float, rx: float = 34.0, ry: float = 42.0) -> QPainterPath:
        """Returns the facial oval vector path."""
        path = QPainterPath()
        path.addEllipse(QPointF(cx, cy), rx, ry)
        return path

    @staticmethod
    def get_hair_path(cx: float, cy: float, rx: float = 34.0, ry: float = 42.0) -> QPainterPath:
        """Returns the dark bob-cut hair vector path with bangs."""
        path = QPainterPath()
        path.moveTo(cx - rx - 4, cy - 5)
        path.cubicTo(cx - rx - 8, cy - ry - 14, cx + rx + 8, cy - ry - 14, cx + rx + 4, cy - 5)
        path.lineTo(cx + rx - 2, cy - 14)
        path.lineTo(cx + 12, cy - 18)
        path.lineTo(cx + 4, cy - 25)
        path.lineTo(cx - 8, cy - 18)
        path.lineTo(cx - 20, cy - 22)
        path.closeSubpath()
        return path

    @staticmethod
    def get_collar_path(neck: QPointF, chest: QPointF) -> Tuple[QPainterPath, QPainterPath]:
        """Returns (left_collar_flap, right_collar_flap) vector paths for collared shirt."""
        l_flap = QPainterPath()
        l_flap.moveTo(neck.x() - 4, neck.y() + 4)
        l_flap.lineTo(neck.x() - 22, neck.y() + 2)
        l_flap.lineTo(neck.x() - 14, neck.y() + 20)
        l_flap.lineTo(chest.x(), neck.y() + 14)
        l_flap.closeSubpath()

        r_flap = QPainterPath()
        r_flap.moveTo(neck.x() + 4, neck.y() + 4)
        r_flap.lineTo(neck.x() + 22, neck.y() + 2)
        r_flap.lineTo(neck.x() + 14, neck.y() + 20)
        r_flap.lineTo(chest.x(), neck.y() + 14)
        r_flap.closeSubpath()

        return l_flap, r_flap

    @staticmethod
    def get_eye_path(center: QPointF, ew: float = 8.5, eh: float = 6.0) -> QPainterPath:
        """Returns almond-shaped eye vector path."""
        path = QPainterPath()
        path.addEllipse(center, ew, eh)
        return path

    @staticmethod
    def get_finger_capsule_path(p1: QPointF, p2: QPointF, radius: float) -> QPainterPath:
        """Returns rounded bone capsule vector path between two joint points."""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        L = math.hypot(dx, dy)
        if L < 0.1:
            path = QPainterPath()
            path.addEllipse(p1, radius, radius)
            return path

        nx = -dy / L * radius
        ny = dx / L * radius

        capsule = QPainterPath()
        capsule.moveTo(p1.x() + nx, p1.y() + ny)
        capsule.lineTo(p2.x() + nx * 0.85, p2.y() + ny * 0.85)
        capsule.arcTo(QRectF(p2.x() - radius * 0.85, p2.y() - radius * 0.85, radius * 1.7, radius * 1.7), 0, 180)
        capsule.lineTo(p1.x() - nx, p1.y() - ny)
        capsule.closeSubpath()
        return capsule

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

        # 3. Render Modular Cel-Vector Humanoid
        self._draw_cel_shaded_body(painter, cur_frame, w, h)

        painter.restore()

        # 4. Viewport Badges & Overlays
        self._draw_overlay_badges(painter, w, h)

    def _draw_backdrop(self, p: QPainter, w: float, h: float) -> None:
        """Draws rich navy-slate studio backdrop with subtle grid."""
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, COLOR_BG_STUDIO_TOP)
        bg_grad.setColorAt(1.0, COLOR_BG_STUDIO_BTM)
        p.fillRect(QRectF(0, 0, w, h), bg_grad)

        # Subtle cyber grid
        p.setPen(QPen(COLOR_BG_GRID, 1, Qt.PenStyle.DotLine))
        grid_step = 28
        for x in range(0, int(w), grid_step):
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        for y in range(0, int(h), grid_step):
            p.drawLine(QPointF(0, y), QPointF(w, y))

    def _draw_placeholder(self, p: QPainter, w: float, h: float) -> None:
        p.setPen(QPen(QColor("#94A3B8"), 1))
        p.setFont(QFont("Segoe UI", 12))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "Loading Toon Avatar...")

    # ── Modular Cel-Vector Body Drawing ───────────────────────────────────────

    def _draw_cel_shaded_body(self, p: QPainter, frame: KinematicJointFrame, w: float, h: float) -> None:
        """Multi-pass layered vector character drawing."""
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

        # 1. Torso & Collared Shirt
        self._draw_torso_and_clothing(p, chest_pt, neck_pt, ls_pt, rs_pt, w, h)

        # 2. Arms & Sleeves
        self._draw_vector_arm(p, ls_pt, le_pt, lw_pt, is_right=False)
        self._draw_vector_arm(p, rs_pt, re_pt, rw_pt, is_right=True)

        # 3. Neck
        self._draw_neck(p, neck_pt, head_pt)

        # 4. Head & Face Features
        self._draw_head_and_hair(p, head_pt, w, h)
        self._draw_face_features(p, head_pt, w, h)

        # 5. Segmented Articulated Hands (Depth-Sorted)
        if frame.right_hand_z >= frame.left_hand_z:
            if frame.is_left_active and frame.left_hand:
                self._draw_vector_hand(p, frame.left_hand, is_right=False, w=w, h=h)
            if frame.is_right_active and frame.right_hand:
                self._draw_vector_hand(p, frame.right_hand, is_right=True, w=w, h=h)
        else:
            if frame.is_right_active and frame.right_hand:
                self._draw_vector_hand(p, frame.right_hand, is_right=True, w=w, h=h)
            if frame.is_left_active and frame.left_hand:
                self._draw_vector_hand(p, frame.left_hand, is_right=False, w=w, h=h)

        # 6. Touch Halos
        self._draw_touch_effects(p, frame, w, h)

    def _draw_torso_and_clothing(
        self, p: QPainter, chest: QPointF, neck: QPointF, ls: QPointF, rs: QPointF, w: float, h: float
    ) -> None:
        """Draws slate-blue collared shirt with line-art collar flaps and button placket."""
        # 1. Shirt Torso Body
        shirt_path = QPainterPath()
        shirt_path.moveTo(ls)
        shirt_path.lineTo(rs)
        shirt_path.lineTo(rs.x() + 15, h * 0.95)
        shirt_path.lineTo(ls.x() - 15, h * 0.95)
        shirt_path.closeSubpath()

        p.fillPath(shirt_path, QBrush(COLOR_SHIRT_BASE))

        # 2. Cel-Shadow Fold (Right flank)
        shadow_path = QPainterPath()
        shadow_path.moveTo(chest.x() + 5, neck.y())
        shadow_path.lineTo(rs)
        shadow_path.lineTo(rs.x() + 15, h * 0.95)
        shadow_path.lineTo(chest.x() + 12, h * 0.95)
        shadow_path.closeSubpath()
        p.fillPath(shadow_path, QBrush(COLOR_SHIRT_SHADOW))

        # 3. Contour Outline
        p.setPen(QPen(COLOR_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(shirt_path)

        # 4. Center Button Placket
        p.setPen(QPen(COLOR_SHIRT_PLACKET, 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        p.drawLine(QPointF(chest.x(), neck.y() + 10), QPointF(chest.x(), h * 0.95))

        # Placket Line-Art Seams
        p.setPen(QPen(COLOR_OUTLINE, 1.5))
        p.drawLine(QPointF(chest.x() - 4, neck.y() + 10), QPointF(chest.x() - 4, h * 0.95))
        p.drawLine(QPointF(chest.x() + 4, neck.y() + 10), QPointF(chest.x() + 4, h * 0.95))

        # White Buttons
        p.setPen(QPen(COLOR_OUTLINE, 1.0))
        p.setBrush(QBrush(COLOR_SHIRT_BUTTON))
        for btn_y in [neck.y() + 24, neck.y() + 44, neck.y() + 64, neck.y() + 84]:
            if btn_y < h * 0.92:
                p.drawEllipse(QPointF(chest.x(), btn_y), 2.2, 2.2)

        # 5. Crisp Line-Art Collar Flaps
        l_flap, r_flap = self.get_collar_path(neck, chest)
        p.fillPath(l_flap, QBrush(COLOR_COLLAR_BASE))
        p.fillPath(r_flap, QBrush(COLOR_COLLAR_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(l_flap)
        p.drawPath(r_flap)

    def _draw_neck(self, p: QPainter, neck: QPointF, head: QPointF) -> None:
        """Draws warm natural neck cylinder."""
        nw = 18.0
        neck_path = QPainterPath()
        neck_path.moveTo(neck.x() - nw, head.y() + 20)
        neck_path.lineTo(neck.x() + nw, head.y() + 20)
        neck_path.lineTo(neck.x() + nw + 2, neck.y() + 4)
        neck_path.lineTo(neck.x() - nw - 2, neck.y() + 4)
        neck_path.closeSubpath()

        p.fillPath(neck_path, QBrush(COLOR_SKIN_SHADOW))
        p.setPen(QPen(COLOR_OUTLINE, 2.0))
        p.drawPath(neck_path)

    def _draw_vector_arm(self, p: QPainter, shoulder: QPointF, elbow: QPointF, wrist: QPointF, is_right: bool) -> None:
        """Draws collared shirt sleeve + forearm cylinder."""
        arm_w = 16.0

        # Sleeve Segment
        dx = elbow.x() - shoulder.x()
        dy = elbow.y() - shoulder.y()
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * (arm_w * 0.5), dx / L * (arm_w * 0.5)

        up_path = QPainterPath()
        up_path.moveTo(shoulder.x() + nx, shoulder.y() + ny)
        up_path.lineTo(elbow.x() + nx, elbow.y() + ny)
        up_path.lineTo(elbow.x() - nx, elbow.y() - ny)
        up_path.lineTo(shoulder.x() - nx, shoulder.y() - ny)
        up_path.closeSubpath()

        p.fillPath(up_path, QBrush(COLOR_SHIRT_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(up_path)

        # Forearm Segment (Warm Skin)
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

        p.fillPath(fa_path, QBrush(COLOR_SKIN_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.0))
        p.drawPath(fa_path)

    def _draw_head_and_hair(self, p: QPainter, head: QPointF, w: float, h: float) -> None:
        """Draws facial oval and dark bob-cut hair."""
        cx, cy = head.x(), head.y()
        rx, ry = 34.0, 42.0

        # Facial Oval
        head_path = self.get_head_path(cx, cy, rx, ry)
        p.fillPath(head_path, QBrush(COLOR_SKIN_BASE))

        # Cel Shadow
        p.save()
        p.setClipPath(head_path)
        shadow_poly = QPolygonF([
            QPointF(cx - rx, cy + 10),
            QPointF(cx + rx, cy + 5),
            QPointF(cx + rx, cy + ry),
            QPointF(cx - rx, cy + ry),
        ])
        shadow_path = QPainterPath()
        shadow_path.addPolygon(shadow_poly)
        p.fillPath(shadow_path, QBrush(COLOR_SKIN_SHADOW))
        p.restore()

        p.setPen(QPen(COLOR_OUTLINE, 2.5))
        p.drawPath(head_path)

        # Hair
        hair_path = self.get_hair_path(cx, cy, rx, ry)
        p.fillPath(hair_path, QBrush(COLOR_HAIR_BASE))

        # Hair Highlight Sheen
        hl_path = QPainterPath()
        hl_path.moveTo(cx - 18, cy - ry + 4)
        hl_path.quadTo(cx, cy - ry - 2, cx + 18, cy - ry + 4)
        p.setPen(QPen(COLOR_HAIR_HL, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(hl_path)

        p.setPen(QPen(COLOR_OUTLINE, 2.5))
        p.drawPath(hair_path)

    def _draw_face_features(self, p: QPainter, head: QPointF, w: float, h: float) -> None:
        """Draws cheeks, almond eyes, dynamic FACS brows, nose ridge, and terracotta lips."""
        cx, cy = head.x(), head.y()

        # Cheeks
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(COLOR_CHEEK_BLUSH))
        p.drawEllipse(QPointF(cx - 20, cy + 8), 8, 5)
        p.drawEllipse(QPointF(cx + 20, cy + 8), 8, 5)

        # Almond Eyes
        self._draw_eye(p, QPointF(cx - 14, cy - 2), is_right=False)
        self._draw_eye(p, QPointF(cx + 14, cy - 2), is_right=True)

        # FACS Eyebrows
        self._draw_eyebrows(p, cx, cy)

        # Nose Ridge
        p.setPen(QPen(COLOR_SKIN_CREASE, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(cx, cy + 3), QPointF(cx + 2, cy + 10))
        p.drawLine(QPointF(cx + 2, cy + 10), QPointF(cx - 2, cy + 11))

        # Terracotta Lips
        self._draw_lips(p, cx, cy + 22)

    def _draw_eye(self, p: QPainter, center: QPointF, is_right: bool) -> None:
        """Draws crisp vector eye with sclera, cyan iris, pupil, and specular dot."""
        ex, ey = center.x(), center.y()
        ew, eh = 8.5, 6.0

        eye_path = self.get_eye_path(center, ew, eh)
        p.fillPath(eye_path, QBrush(COLOR_EYE_WHITE))
        p.setPen(QPen(COLOR_OUTLINE, 1.8))
        p.drawPath(eye_path)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(COLOR_IRIS_BASE))
        p.drawEllipse(center, 4.5, 5.0)

        p.setBrush(QBrush(COLOR_PUPIL))
        p.drawEllipse(center, 2.5, 2.8)

        p.setBrush(QBrush(COLOR_EYE_SPEC))
        p.drawEllipse(QPointF(ex + (1.2 if is_right else -1.2), ey - 1.8), 1.6, 1.6)

        p.setPen(QPen(COLOR_OUTLINE, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(ex - ew - 1, ey - 2), QPointF(ex + ew + 1, ey - 2))

    def _draw_eyebrows(self, p: QPainter, cx: float, cy: float) -> None:
        p.setPen(QPen(COLOR_HAIR_BASE, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

        l_inner_y = cy - 14 - (self.au01_inner_brow * 6.0) + (self.au04_brow_furrow * 4.0)
        l_outer_y = cy - 12 - (self.au02_outer_brow * 6.0)
        l_mid_y   = (l_inner_y + l_outer_y) * 0.5 - 2.0

        l_path = QPainterPath()
        l_path.moveTo(cx - 24, l_outer_y)
        l_path.quadTo(cx - 16, l_mid_y, cx - 6, l_inner_y)
        p.drawPath(l_path)

        r_inner_y = cy - 14 - (self.au01_inner_brow * 6.0) + (self.au04_brow_furrow * 4.0)
        r_outer_y = cy - 12 - (self.au02_outer_brow * 6.0)
        r_mid_y   = (r_inner_y + r_outer_y) * 0.5 - 2.0

        r_path = QPainterPath()
        r_path.moveTo(cx + 6, r_inner_y)
        r_path.quadTo(cx + 16, r_mid_y, cx + 24, r_outer_y)
        p.drawPath(r_path)

    def _draw_lips(self, p: QPainter, cx: float, cy: float) -> None:
        open_h = self.mouth_open_ratio * 12.0
        half_w = 11.0

        mouth_path = QPainterPath()
        if open_h > 2.5:
            mouth_path.moveTo(cx - half_w, cy)
            mouth_path.quadTo(cx, cy - 2, cx + half_w, cy)
            mouth_path.quadTo(cx, cy + open_h, cx - half_w, cy)
            mouth_path.closeSubpath()
            p.fillPath(mouth_path, QBrush(COLOR_MOUTH_INNER))
            p.setPen(QPen(COLOR_OUTLINE, 2.0))
            p.drawPath(mouth_path)

            teeth = QPainterPath()
            teeth.moveTo(cx - half_w + 3, cy)
            teeth.lineTo(cx + half_w - 3, cy)
            teeth.lineTo(cx + half_w - 3, cy + 3)
            teeth.lineTo(cx - half_w + 3, cy + 3)
            teeth.closeSubpath()
            p.fillPath(teeth, QBrush(QColor("#FFFFFF")))
        else:
            mouth_path.moveTo(cx - half_w, cy)
            mouth_path.quadTo(cx, cy + 4, cx + half_w, cy)
            p.setPen(QPen(COLOR_LIP_BASE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(mouth_path)

    # ── Segmented 5-Finger Articulated Vector Hand ────────────────────────────

    def _draw_vector_hand(
        self, p: QPainter, hand_pts: List[Tuple[float, float]], is_right: bool, w: float, h: float
    ) -> None:
        """Renders 5-finger articulated vector hand with line-art stroke separation."""
        if len(hand_pts) < 21:
            return

        def P(idx: int) -> QPointF:
            return QPointF(hand_pts[idx][0] * w, hand_pts[idx][1] * h)

        # 1. Palm Base Polygon
        palm_poly = QPolygonF([P(0), P(1), P(5), P(9), P(13), P(17)])
        palm_path = QPainterPath()
        palm_path.addPolygon(palm_poly)
        p.fillPath(palm_path, QBrush(COLOR_SKIN_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(palm_path)

        # 2. 5 Articulated Fingers (Thumb, Index, Middle, Ring, Pinky)
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
                self._draw_finger_bone_capsule(p, p1, p2, seg_w)

            # Fingernail Highlight
            tip_pt = P(chain[-1])
            prev_pt = P(chain[-2])
            self._draw_fingernail(p, tip_pt, prev_pt, f_w * 0.6)

    def _draw_finger_bone_capsule(self, p: QPainter, p1: QPointF, p2: QPointF, radius: float) -> None:
        """Draws discrete finger bone capsule with bold outline to eliminate silhouette blending."""
        capsule = self.get_finger_capsule_path(p1, p2, radius)
        p.fillPath(capsule, QBrush(COLOR_SKIN_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(capsule)

        # Knuckle Crease Line
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        L = math.hypot(dx, dy) or 1.0
        nx = -dy / L * radius
        ny = dx / L * radius
        p.setPen(QPen(COLOR_SKIN_CREASE, 1.3))
        p.drawLine(QPointF(p1.x() + nx * 0.6, p1.y() + ny * 0.6), QPointF(p1.x() - nx * 0.6, p1.y() - ny * 0.6))

    def _draw_fingernail(self, p: QPainter, tip: QPointF, prev: QPointF, size: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(COLOR_NAIL_BASE))
        p.drawEllipse(tip, size, size * 0.8)

    def _draw_touch_effects(self, p: QPainter, frame: KinematicJointFrame, w: float, h: float) -> None:
        if not frame.touch_contacts:
            return

        for a, b, intensity in frame.touch_contacts:
            if intensity <= 0.05:
                continue
            if frame.right_hand and a < len(frame.right_hand):
                pt = QPointF(frame.right_hand[a][0] * w, frame.right_hand[a][1] * h)
                halo_r = 12.0 * intensity + 4.0
                p.setPen(QPen(COLOR_TOUCH_HALO, 2.0))
                p.setBrush(QBrush(QColor(245, 158, 11, int(70 * intensity))))
                p.drawEllipse(pt, halo_r, halo_r)

    def _draw_overlay_badges(self, p: QPainter, w: float, h: float) -> None:
        # Top-left sign title badge
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(15, 23, 42, 210)))
        p.drawRoundedRect(QRectF(10, 10, 180, 42), 8, 8)

        p.setPen(QPen(QColor("#F8FAFC"), 1))
        p.setFont(QFont("Hind Siliguri", 13, QFont.Weight.Bold))
        p.drawText(QRectF(18, 12, 160, 20), Qt.AlignmentFlag.AlignLeft, f"{self.label_bn}")

        p.setPen(QPen(QColor("#38BDF8"), 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(18, 30, 160, 18), Qt.AlignmentFlag.AlignLeft, f"{self.label_en.upper()} &bull; 60 FPS")

        # Top-right View Mode Badge
        mode_text = "🔍 হাতের জুম (2.2x)" if self.view_mode == AvatarViewMode.HAND_ZOOM else "👤 ফুল বডি"
        mode_color = QColor("#38BDF8") if self.view_mode == AvatarViewMode.HAND_ZOOM else QColor("#A78BFA")

        p.setBrush(QBrush(QColor(15, 23, 42, 210)))
        p.drawRoundedRect(QRectF(w - 120, 10, 110, 28), 6, 6)

        p.setPen(QPen(mode_color, 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(w - 120, 10, 110, 28), Qt.AlignmentFlag.AlignCenter, mode_text)
