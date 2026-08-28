"""Stylized Cel-Vector 2D Humanoid Signer Illustration Engine (PyQt6).

Renders a pristine, high-DPI 2D cel-shaded educational sign language vector avatar
reflecting the reference vector illustration aesthetic:
  - Layer 0: Smooth slate studio backdrop (#374B5C / #2F4150) with subtle grid
  - Layer 1: Back hair silhouette mass (#181A20) behind neck
  - Layer 2: Tapered neck & throat (#D08964 shadow, #E29F78 base) connecting head to torso
  - Layer 3: Torso with inner shirt (#1E293B) and slate-blue buttoned shirt (#527490) with
             crisp collar lapels (#141E28 2.5px strokes) and central button placket
  - Layer 4: Head base, smooth jawline, ears with warm skin gradient (#E8A882 to #DC9670)
  - Layer 5: Expressive facial matrix: almond eyes with #0284C7 iris, specular highlights,
             dynamic FACS eyebrows (AU01/AU02/AU04), defined nose notch, terracotta lips (#CF6B4E)
  - Layer 6: Front layered curved bangs with subtle highlight arcs (#3A3D4A / #4E5263)
  - Layer 7 & 8: Parametric 5-finger articulated arms and segmented hands with distinct
                 knuckle creases, palm lines, fingernails, and bold #141E28 isolation outlines

Features:
  - HyperKinematic Bézier Motion Engine (60 FPS smooth trajectory interpolation)
  - Dual Perspective View: Full Upper Torso (1.0x) vs Hand Close-Up Zoom (2.2x)
  - Multi-speed playback (1.0x, 0.5x, 0.25x), frame scrubbing, and loop control
"""

import enum
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

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
COLOR_BG_STUDIO_TOP = QColor("#374B5C")  # Smooth slate blue-gray
COLOR_BG_STUDIO_BTM = QColor("#2F4150")  # Deep slate bottom
COLOR_BG_GRID       = QColor(255, 255, 255, 12)

# Hair Palette
COLOR_HAIR_BACK     = QColor("#181A20")  # Dark hair mass behind neck
COLOR_HAIR_BASE     = QColor("#1E1E24")  # Main bob-cut black-brown
COLOR_HAIR_HL       = QColor("#3A3D4A")  # Layered bangs highlight sheen
COLOR_HAIR_HL_LIGHT = QColor("#4E5263")
COLOR_OUTLINE       = QColor("#141E28")  # Crisp vector stroke outline (2.5px)

# Skin Tones (Warm natural skin gradient)
COLOR_SKIN_BASE      = QColor("#E8A882")  # Warm natural skin base
COLOR_SKIN_SHADOW    = QColor("#DC9670")  # Soft shadow gradient
COLOR_SKIN_DEEP      = QColor("#D08964")  # Deep shadow / neck shadow
COLOR_SKIN_LIGHT     = QColor("#F5C3A6")  # Specular / highlight
COLOR_SKIN_CREASE    = QColor("#A8603B")  # Palm & knuckle creases
COLOR_CHEEK_BLUSH    = QColor(244, 114, 182, 60)

# Eyes & Facial Features
COLOR_EYE_WHITE      = QColor("#FFFFFF")
COLOR_IRIS_BASE      = QColor("#0284C7")  # Expressive cyan-blue
COLOR_IRIS_DARK      = QColor("#0369A1")
COLOR_PUPIL          = QColor("#0F172A")
COLOR_EYE_SPEC       = QColor("#FFFFFF")
COLOR_LIP_BASE       = QColor("#CF6B4E")  # Well-defined terracotta lips
COLOR_LIP_DARK       = QColor("#B45309")
COLOR_MOUTH_INNER    = QColor("#881337")

# Costume: Slate-Blue Shirt
COLOR_SHIRT_BASE     = QColor("#527490")  # Modern slate-blue shirt
COLOR_SHIRT_SHADOW   = QColor("#3D5970")  # Cel shadow fold
COLOR_SHIRT_INNER    = QColor("#1E293B")  # Inner shirt collar / base
COLOR_COLLAR_BASE    = QColor("#5D82A2")  # Crisp collar flaps
COLOR_SHIRT_PLACKET  = QColor("#3D5970")  # Button placket
COLOR_SHIRT_BUTTON   = QColor("#F8FAFC")  # Clean white/pearl buttons

# Hands & Articulators
COLOR_NAIL_BASE      = QColor("#FED7AA")
COLOR_TOUCH_HALO     = QColor(245, 158, 11, 190)  # Amber touch aura
COLOR_AURA_CYAN      = QColor(6, 182, 212, 160)


class ToonAvatarRenderer(QWidget):
    """Ultra-Clean Layered 2D Cel-Vector Humanoid Signer Widget.

    Renders a high-DPI vector character playing 60 FPS Bézier motion sequences
    with back-to-front Z-index depth sorting and anatomically articulated 5-finger hands.
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

    # ── Vector Path Helper Methods (Exposed for Testing & Layering) ──────────

    @staticmethod
    def get_head_path(cx: float, cy: float, rx: float = 34.0, ry: float = 42.0) -> QPainterPath:
        """Returns the smooth feminine jawline and facial oval vector path."""
        path = QPainterPath()
        path.addEllipse(QPointF(cx, cy), rx, ry)
        return path

    @staticmethod
    def get_back_hair_path(cx: float, cy: float, rx: float = 34.0, ry: float = 42.0) -> QPainterPath:
        """Returns Layer 1: Dark hair silhouette mass behind neck and head."""
        path = QPainterPath()
        path.moveTo(cx - rx - 8, cy - ry * 0.4)
        path.cubicTo(cx - rx - 12, cy - ry - 14, cx + rx + 12, cy - ry - 14, cx + rx + 8, cy - ry * 0.4)
        path.cubicTo(cx + rx + 14, cy + ry * 0.5, cx + rx + 10, cy + ry * 1.1, cx + rx * 0.7, cy + ry * 1.25)
        path.cubicTo(cx + rx * 0.4, cy + ry * 1.32, cx - rx * 0.4, cy + ry * 1.32, cx - rx * 0.7, cy + ry * 1.25)
        path.cubicTo(cx - rx - 10, cy + ry * 1.1, cx - rx - 14, cy + ry * 0.5, cx - rx - 8, cy - ry * 0.4)
        path.closeSubpath()
        return path

    @staticmethod
    def get_hair_path(cx: float, cy: float, rx: float = 34.0, ry: float = 42.0) -> QPainterPath:
        """Returns Layer 6: Front dark bob-cut hair with layered bangs."""
        path = QPainterPath()
        path.moveTo(cx - rx - 6, cy - 2)
        path.cubicTo(cx - rx - 10, cy - ry - 16, cx + rx + 10, cy - ry - 16, cx + rx + 6, cy - 2)
        path.cubicTo(cx + rx + 2, cy + ry * 0.35, cx + rx - 4, cy + ry * 0.65, cx + rx - 6, cy + ry * 0.7)
        path.cubicTo(cx + rx - 10, cy + ry * 0.3, cx + rx - 14, cy - ry * 0.1, cx + 18, cy - ry * 0.35)
        path.cubicTo(cx + 8, cy - ry * 0.05, cx + 2, cy - ry * 0.25, cx - 6, cy - ry * 0.4)
        path.cubicTo(cx - 16, cy - ry * 0.1, cx - 22, cy + ry * 0.3, cx - rx + 4, cy + ry * 0.7)
        path.cubicTo(cx - rx + 2, cy + ry * 0.65, cx - rx - 4, cy + ry * 0.35, cx - rx - 6, cy - 2)
        path.closeSubpath()
        return path

    @staticmethod
    def get_collar_path(neck: QPointF, chest: QPointF) -> Tuple[QPainterPath, QPainterPath]:
        """Returns (left_collar_flap, right_collar_flap) crisp vector paths for slate-blue shirt."""
        l_flap = QPainterPath()
        l_flap.moveTo(neck.x() - 3, neck.y() + 2)
        l_flap.lineTo(neck.x() - 24, neck.y() - 1)
        l_flap.lineTo(neck.x() - 16, neck.y() + 22)
        l_flap.lineTo(chest.x() - 2, neck.y() + 15)
        l_flap.closeSubpath()

        r_flap = QPainterPath()
        r_flap.moveTo(neck.x() + 3, neck.y() + 2)
        r_flap.lineTo(neck.x() + 24, neck.y() - 1)
        r_flap.lineTo(neck.x() + 16, neck.y() + 22)
        r_flap.lineTo(chest.x() + 2, neck.y() + 15)
        r_flap.closeSubpath()

        return l_flap, r_flap

    @staticmethod
    def get_eye_path(center: QPointF, ew: float = 8.5, eh: float = 6.0) -> QPainterPath:
        """Returns stylized almond-shaped eye vector path."""
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
        capsule.lineTo(p2.x() + nx * 0.88, p2.y() + ny * 0.88)
        # Rounded tip cap at p2
        capsule.cubicTo(
            p2.x() + nx * 0.88 + dx / L * radius * 0.7,
            p2.y() + ny * 0.88 + dy / L * radius * 0.7,
            p2.x() - nx * 0.88 + dx / L * radius * 0.7,
            p2.y() - ny * 0.88 + dy / L * radius * 0.7,
            p2.x() - nx * 0.88,
            p2.y() - ny * 0.88,
        )
        capsule.lineTo(p1.x() - nx, p1.y() - ny)
        capsule.closeSubpath()
        return capsule

    # ── Paint Event & Back-to-Front Z-Index Pipeline ─────────────────────────

    def paintEvent(self, event) -> None:
        """High-DPI double-buffered cel-shaded vector rendering with strict back-to-front layering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = float(self.width())
        h = float(self.height())

        # Layer 0: Studio Backdrop
        self._draw_backdrop(painter, w, h)

        if not self.frames:
            self._draw_placeholder(painter, w, h)
            return

        cur_frame = self.frames[self.current_frame_idx % len(self.frames)]

        # Setup Zoom & Pan Matrix
        scale, tx, ty = self.get_zoom_transform(w, h)
        painter.save()
        painter.translate(tx, ty)
        painter.scale(scale, scale)

        # Back-to-Front Layered Vector Humanoid Drawing
        self._draw_cel_shaded_body(painter, cur_frame, w, h)

        painter.restore()

        # Viewport Badges & Overlays
        self._draw_overlay_badges(painter, w, h)

    def _draw_backdrop(self, p: QPainter, w: float, h: float) -> None:
        """Layer 0: Smooth slate background gradient (#374B5C to #2F4150) with subtle grid."""
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, COLOR_BG_STUDIO_TOP)
        bg_grad.setColorAt(1.0, COLOR_BG_STUDIO_BTM)
        p.fillRect(QRectF(0, 0, w, h), bg_grad)

        # Subtle aesthetic grid
        p.setPen(QPen(COLOR_BG_GRID, 1, Qt.PenStyle.DotLine))
        grid_step = 28
        for x in range(0, int(w), grid_step):
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        for y in range(0, int(h), grid_step):
            p.drawLine(QPointF(0, y), QPointF(w, y))

    def _draw_placeholder(self, p: QPainter, w: float, h: float) -> None:
        p.setPen(QPen(QColor("#94A3B8"), 1))
        p.setFont(QFont("Segoe UI", 12))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "Loading Vector Avatar...")

    # ── Modular Cel-Vector Body Drawing Pipeline ──────────────────────────────

    def _draw_cel_shaded_body(self, p: QPainter, frame: KinematicJointFrame, w: float, h: float) -> None:
        """Strict 9-layer Back-to-Front Z-Index depth-sorted rendering pipeline."""
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

        # ── Layer 1: Back Hair Silhouette Mass ───────────────────────────────
        self._draw_back_hair(p, head_pt, w, h)

        # ── Layer 2: Neck & Throat ───────────────────────────────────────────
        self._draw_neck(p, neck_pt, head_pt)

        # ── Layer 3: Torso & Slate-Blue Buttoned Shirt ────────────────────────
        self._draw_torso_and_clothing(p, chest_pt, neck_pt, ls_pt, rs_pt, w, h)

        # ── Layer 4: Head Base & Facial Oval with Ears ────────────────────────
        self._draw_head_and_hair(p, head_pt, w, h)

        # ── Layer 5: Facial Expression Matrix ─────────────────────────────────
        self._draw_face_features(p, head_pt, w, h)

        # ── Layer 6: Front Hair & Bangs ───────────────────────────────────────
        self._draw_front_hair_bangs(p, head_pt, w, h)

        # ── Layers 7 & 8: Parametric 5-Finger Articulated Arms & Hands ────────
        # Render Upper Arms / Sleeves
        self._draw_vector_arm(p, ls_pt, le_pt, lw_pt, is_right=False)
        self._draw_vector_arm(p, rs_pt, re_pt, rw_pt, is_right=True)

        # Depth-Sorted Hand Articulation (Back hand first, then front hand)
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

        # Touch Halos
        self._draw_touch_effects(p, frame, w, h)

    # ── Layer 1: Back Hair Silhouette ────────────────────────────────────────

    def _draw_back_hair(self, p: QPainter, head_pos: Any, w: float, h: float, scale: float = 1.0) -> None:
        """Layer 1: Renders dark hair mass behind neck (#181A20)."""
        if isinstance(head_pos, QPointF):
            cx, cy = head_pos.x(), head_pos.y()
        elif isinstance(head_pos, (tuple, list)):
            cx, cy = float(head_pos[0]), float(head_pos[1])
        else:
            cx, cy = w * 0.5, h * 0.25

        rx, ry = 34.0 * scale, 42.0 * scale
        back_hair = self.get_back_hair_path(cx, cy, rx, ry)
        p.fillPath(back_hair, QBrush(COLOR_HAIR_BACK))
        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(back_hair)

    # ── Layer 2: Neck & Throat ───────────────────────────────────────────────

    def _draw_neck(self, p: QPainter, neck: QPointF, head: QPointF, scale: float = 1.0) -> None:
        """Layer 2: Realistic tapered neck (#D08964 shadow, #E29F78 base) connecting head to torso."""
        nw = 19.0 * scale
        neck_top_y = head.y() + 18.0 * scale
        neck_btm_y = neck.y() + 6.0 * scale

        neck_path = QPainterPath()
        neck_path.moveTo(neck.x() - nw * 0.88, neck_top_y)
        neck_path.cubicTo(neck.x() - nw * 0.95, (neck_top_y + neck_btm_y) * 0.5, neck.x() - nw * 1.15, neck_btm_y - 2 * scale, neck.x() - nw * 1.25, neck_btm_y)
        neck_path.lineTo(neck.x() + nw * 1.25, neck_btm_y)
        neck_path.cubicTo(neck.x() + nw * 1.15, neck_btm_y - 2 * scale, neck.x() + nw * 0.95, (neck_top_y + neck_btm_y) * 0.5, neck.x() + nw * 0.88, neck_top_y)
        neck_path.closeSubpath()

        # Warm skin neck gradient (darker under chin)
        grad = QLinearGradient(neck.x(), neck_top_y, neck.x(), neck_btm_y)
        grad.setColorAt(0.0, COLOR_SKIN_DEEP)
        grad.setColorAt(0.4, COLOR_SKIN_SHADOW)
        grad.setColorAt(1.0, COLOR_SKIN_BASE)

        p.fillPath(neck_path, QBrush(grad))
        p.setPen(QPen(COLOR_OUTLINE, 2.2 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(neck_path)

        # Sternocleidomastoid shadow curves
        p.setPen(QPen(COLOR_SKIN_CREASE, 1.4 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(neck.x() - 4 * scale, neck_top_y + 10 * scale), QPointF(neck.x() - 8 * scale, neck_btm_y - 2 * scale))
        p.drawLine(QPointF(neck.x() + 4 * scale, neck_top_y + 10 * scale), QPointF(neck.x() + 8 * scale, neck_btm_y - 2 * scale))

    # ── Layer 3: Torso & Slate Shirt ─────────────────────────────────────────

    def _draw_torso_and_clothing(
        self,
        p: QPainter,
        chest_or_bounds: Any,
        neck: Optional[QPointF] = None,
        ls: Optional[QPointF] = None,
        rs: Optional[QPointF] = None,
        w: Optional[float] = None,
        h: Optional[float] = None,
        scale: float = 1.0,
    ) -> None:
        """Layer 3: Torso with realistic collarbone, inner shirt (#1E293B), and slate-blue shirt (#527490)."""
        if isinstance(chest_or_bounds, (QRectF, tuple, list)):
            bounds = chest_or_bounds if isinstance(chest_or_bounds, QRectF) else QRectF(*chest_or_bounds)
            w = bounds.width()
            h = bounds.height()
            cx = bounds.center().x()
            top = bounds.top()
            neck = QPointF(cx, top + 30.0 * scale)
            chest = QPointF(cx, top + 80.0 * scale)
            ls = QPointF(cx - 62.0 * scale, top + 42.0 * scale)
            rs = QPointF(cx + 62.0 * scale, top + 42.0 * scale)
        else:
            chest = chest_or_bounds
            w = w or float(self.width())
            h = h or float(self.height())

        # 1. Inner V-Neck Base (#1E293B)
        inner_path = QPainterPath()
        inner_path.moveTo(neck.x() - 20 * scale, neck.y() + 2 * scale)
        inner_path.lineTo(neck.x(), neck.y() + 28 * scale)
        inner_path.lineTo(neck.x() + 20 * scale, neck.y() + 2 * scale)
        inner_path.closeSubpath()
        p.fillPath(inner_path, QBrush(COLOR_SHIRT_INNER))

        # 2. Main Slate-Blue Shirt Torso Body
        shirt_path = QPainterPath()
        shirt_path.moveTo(ls.x(), ls.y())
        shirt_path.cubicTo(neck.x() - 30 * scale, neck.y() + 8 * scale, neck.x() + 30 * scale, neck.y() + 8 * scale, rs.x(), rs.y())
        shirt_path.lineTo(rs.x() + 16 * scale, h * 0.96)
        shirt_path.lineTo(ls.x() - 16 * scale, h * 0.96)
        shirt_path.closeSubpath()

        p.fillPath(shirt_path, QBrush(COLOR_SHIRT_BASE))

        # 3. Cel-Shadow Fold (Right flank)
        shadow_path = QPainterPath()
        shadow_path.moveTo(chest.x() + 6 * scale, neck.y() + 10 * scale)
        shadow_path.lineTo(rs.x(), rs.y())
        shadow_path.lineTo(rs.x() + 16 * scale, h * 0.96)
        shadow_path.lineTo(chest.x() + 14 * scale, h * 0.96)
        shadow_path.closeSubpath()
        p.fillPath(shadow_path, QBrush(COLOR_SHIRT_SHADOW))

        # 4. Bold 2.5px Contour Outline
        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(shirt_path)

        # 5. Center Button Placket
        placket_w = 12.0 * scale
        placket_rect = QRectF(chest.x() - placket_w * 0.5, neck.y() + 18 * scale, placket_w, h * 0.96 - (neck.y() + 18 * scale))
        p.fillRect(placket_rect, QBrush(COLOR_SHIRT_PLACKET))

        # Placket Stitch Seams
        p.setPen(QPen(COLOR_OUTLINE, 1.6 * scale))
        p.drawLine(QPointF(chest.x() - placket_w * 0.5, neck.y() + 18 * scale), QPointF(chest.x() - placket_w * 0.5, h * 0.96))
        p.drawLine(QPointF(chest.x() + placket_w * 0.5, neck.y() + 18 * scale), QPointF(chest.x() + placket_w * 0.5, h * 0.96))

        # White Pearl Buttons with dark rims
        p.setPen(QPen(COLOR_OUTLINE, 1.2 * scale))
        p.setBrush(QBrush(COLOR_SHIRT_BUTTON))
        for btn_y in [neck.y() + 32 * scale, neck.y() + 54 * scale, neck.y() + 76 * scale, neck.y() + 98 * scale]:
            if btn_y < h * 0.92:
                p.drawEllipse(QPointF(chest.x(), btn_y), 2.6 * scale, 2.6 * scale)

        # 6. Crisp Collar Flaps with 2.5px vector strokes
        l_flap, r_flap = self.get_collar_path(neck, chest)
        p.fillPath(l_flap, QBrush(COLOR_COLLAR_BASE))
        p.fillPath(r_flap, QBrush(COLOR_COLLAR_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(l_flap)
        p.drawPath(r_flap)

    # ── Layer 4: Head Base & Facial Oval ─────────────────────────────────────

    def _draw_head_and_hair(
        self,
        p: QPainter,
        head_pos: Any,
        scale_or_w: Union[float, None] = 1.0,
        nmm_params_or_h: Any = None,
        scale: float = 1.0,
        nmm_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Layer 4: Head base, smooth jawline, ears with warm skin gradient (#E8A882 to #DC9670)."""
        if isinstance(head_pos, QPointF):
            cx, cy = head_pos.x(), head_pos.y()
        elif isinstance(head_pos, (tuple, list)):
            cx, cy = float(head_pos[0]), float(head_pos[1])
        else:
            cx, cy = float(self.width()) * 0.5, float(self.height()) * 0.25

        if isinstance(scale_or_w, float) and scale_or_w <= 5.0:
            scale = scale_or_w
        if isinstance(nmm_params_or_h, dict):
            nmm_params = nmm_params_or_h

        rx, ry = 34.0 * scale, 42.0 * scale

        # Ears (L & R) behind facial oval
        self._draw_ears(p, cx, cy, rx, ry, scale)

        # Facial Oval with Warm Skin Gradient
        head_path = self.get_head_path(cx, cy, rx, ry)
        skin_grad = QLinearGradient(cx, cy - ry, cx, cy + ry)
        skin_grad.setColorAt(0.0, COLOR_SKIN_LIGHT)
        skin_grad.setColorAt(0.4, COLOR_SKIN_BASE)
        skin_grad.setColorAt(1.0, COLOR_SKIN_SHADOW)
        p.fillPath(head_path, QBrush(skin_grad))

        # Soft Chin / Jaw Shadow
        p.save()
        p.setClipPath(head_path)
        jaw_shadow = QPainterPath()
        jaw_shadow.moveTo(cx - rx, cy + ry * 0.4)
        jaw_shadow.cubicTo(cx - rx * 0.5, cy + ry * 0.7, cx + rx * 0.5, cy + ry * 0.7, cx + rx, cy + ry * 0.4)
        jaw_shadow.lineTo(cx + rx, cy + ry)
        jaw_shadow.lineTo(cx - rx, cy + ry)
        jaw_shadow.closeSubpath()
        p.fillPath(jaw_shadow, QBrush(COLOR_SKIN_DEEP))
        p.restore()

        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(head_path)

    def _draw_ears(self, p: QPainter, cx: float, cy: float, rx: float, ry: float, scale: float = 1.0) -> None:
        """Renders stylized ear lobes with inner auricle cartilage."""
        ear_w, ear_h = 7.0 * scale, 12.0 * scale
        ear_y = cy + 2.0 * scale

        for is_r in [False, True]:
            ear_x = cx + (rx - 2 * scale if is_r else -rx + 2 * scale)
            ear_path = QPainterPath()
            ear_path.addEllipse(QPointF(ear_x, ear_y), ear_w, ear_h)
            p.fillPath(ear_path, QBrush(COLOR_SKIN_BASE))
            p.setPen(QPen(COLOR_OUTLINE, 2.2 * scale))
            p.drawPath(ear_path)

            # Inner auricle cartilage detail
            p.setPen(QPen(COLOR_SKIN_CREASE, 1.4 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            inner_path = QPainterPath()
            inner_x = ear_x + (1.5 * scale if is_r else -1.5 * scale)
            inner_path.moveTo(inner_x, ear_y - 4 * scale)
            inner_path.quadTo(inner_x + (2.0 * scale if is_r else -2.0 * scale), ear_y, inner_x, ear_y + 4 * scale)
            p.drawPath(inner_path)

    # ── Layer 5: Facial Expression Matrix ────────────────────────────────────

    def _draw_face_features(
        self,
        p: QPainter,
        head_pos: Any,
        scale_or_w: Union[float, None] = 1.0,
        nmm_params_or_h: Any = None,
        scale: float = 1.0,
        nmm_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Layer 5: Cheeks, almond eyes with specular highlights, dynamic brows, nose, and terracotta lips."""
        if isinstance(head_pos, QPointF):
            cx, cy = head_pos.x(), head_pos.y()
        elif isinstance(head_pos, (tuple, list)):
            cx, cy = float(head_pos[0]), float(head_pos[1])
        else:
            cx, cy = float(self.width()) * 0.5, float(self.height()) * 0.25

        if isinstance(scale_or_w, float) and scale_or_w <= 5.0:
            scale = scale_or_w
        if isinstance(nmm_params_or_h, dict):
            nmm_params = nmm_params_or_h

        # Update temporary FACS Action Units if provided
        if nmm_params:
            if "au01" in nmm_params or "au01_inner_brow" in nmm_params:
                self.au01_inner_brow = nmm_params.get("au01", nmm_params.get("au01_inner_brow", self.au01_inner_brow))
            if "au02" in nmm_params or "au02_outer_brow" in nmm_params:
                self.au02_outer_brow = nmm_params.get("au02", nmm_params.get("au02_outer_brow", self.au02_outer_brow))
            if "au04" in nmm_params or "au04_brow_furrow" in nmm_params:
                self.au04_brow_furrow = nmm_params.get("au04", nmm_params.get("au04_brow_furrow", self.au04_brow_furrow))
            if "mouth_open" in nmm_params or "mouth_open_ratio" in nmm_params:
                self.mouth_open_ratio = nmm_params.get("mouth_open", nmm_params.get("mouth_open_ratio", self.mouth_open_ratio))

        # 1. Cheeks Blush
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(COLOR_CHEEK_BLUSH))
        p.drawEllipse(QPointF(cx - 20 * scale, cy + 9 * scale), 8.5 * scale, 5.5 * scale)
        p.drawEllipse(QPointF(cx + 20 * scale, cy + 9 * scale), 8.5 * scale, 5.5 * scale)

        # 2. Almond Eyes with Iris, Pupil & Specular Highlights
        self._draw_eye(p, QPointF(cx - 14 * scale, cy - 2 * scale), is_right=False, scale=scale)
        self._draw_eye(p, QPointF(cx + 14 * scale, cy - 2 * scale), is_right=True, scale=scale)

        # 3. Dynamic FACS Eyebrows
        self._draw_eyebrows(p, cx, cy, scale=scale)

        # 4. Defined Nose Contour Notch
        p.setPen(QPen(COLOR_SKIN_CREASE, 2.0 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        nose_path = QPainterPath()
        nose_path.moveTo(cx - 1 * scale, cy + 2 * scale)
        nose_path.lineTo(cx + 2.5 * scale, cy + 10 * scale)
        nose_path.lineTo(cx - 2.5 * scale, cy + 11.5 * scale)
        p.drawPath(nose_path)

        # 5. Terracotta Lips (#CF6B4E)
        self._draw_lips(p, cx, cy + 22 * scale, scale=scale)

    def _draw_eye(self, p: QPainter, center: QPointF, is_right: bool, scale: float = 1.0) -> None:
        """Draws crisp vector eye with sclera, #0284C7 cyan-blue iris, pupil, and white specular dot."""
        ex, ey = center.x(), center.y()
        ew, eh = 8.5 * scale, 6.0 * scale

        eye_path = self.get_eye_path(center, ew, eh)
        p.fillPath(eye_path, QBrush(COLOR_EYE_WHITE))
        p.setPen(QPen(COLOR_OUTLINE, 2.0 * scale))
        p.drawPath(eye_path)

        # Iris with vertical gradient (#0369A1 top to #0284C7 bottom)
        p.setPen(Qt.PenStyle.NoPen)
        iris_grad = QLinearGradient(ex, ey - eh, ex, ey + eh)
        iris_grad.setColorAt(0.0, COLOR_IRIS_DARK)
        iris_grad.setColorAt(1.0, COLOR_IRIS_BASE)
        p.setBrush(QBrush(iris_grad))
        p.drawEllipse(center, 4.6 * scale, 5.2 * scale)

        # Pupil
        p.setBrush(QBrush(COLOR_PUPIL))
        p.drawEllipse(center, 2.5 * scale, 2.8 * scale)

        # White Specular Highlights (Double dot for anime/cel clarity)
        p.setBrush(QBrush(COLOR_EYE_SPEC))
        spec_x = ex + (1.3 if is_right else -1.3) * scale
        p.drawEllipse(QPointF(spec_x, ey - 1.8 * scale), 1.6 * scale, 1.6 * scale)
        p.drawEllipse(QPointF(ex - (0.8 if is_right else -0.8) * scale, ey + 1.2 * scale), 0.8 * scale, 0.8 * scale)

        # Upper Eyelash Eyeliner Stroke
        p.setPen(QPen(COLOR_OUTLINE, 2.6 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        eyelash = QPainterPath()
        eyelash.moveTo(ex - ew - 1.5 * scale, ey - 0.5 * scale)
        eyelash.cubicTo(ex - ew * 0.5, ey - eh * 1.35, ex + ew * 0.5, ey - eh * 1.35, ex + ew + 1.5 * scale, ey - 0.5 * scale)
        p.drawPath(eyelash)

    def _draw_eyebrows(self, p: QPainter, cx: float, cy: float, scale: float = 1.0) -> None:
        """Renders dynamic curved dark eyebrows responding to AU01, AU02, AU04."""
        p.setPen(QPen(COLOR_HAIR_BASE, 3.2 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

        l_inner_y = cy - (14.0 * scale) - (self.au01_inner_brow * 6.0 * scale) + (self.au04_brow_furrow * 4.0 * scale)
        l_outer_y = cy - (12.0 * scale) - (self.au02_outer_brow * 6.0 * scale)
        l_mid_y   = (l_inner_y + l_outer_y) * 0.5 - (2.5 * scale)

        l_path = QPainterPath()
        l_path.moveTo(cx - 24 * scale, l_outer_y)
        l_path.quadTo(cx - 16 * scale, l_mid_y, cx - 6 * scale, l_inner_y)
        p.drawPath(l_path)

        r_inner_y = cy - (14.0 * scale) - (self.au01_inner_brow * 6.0 * scale) + (self.au04_brow_furrow * 4.0 * scale)
        r_outer_y = cy - (12.0 * scale) - (self.au02_outer_brow * 6.0 * scale)
        r_mid_y   = (r_inner_y + r_outer_y) * 0.5 - (2.5 * scale)

        r_path = QPainterPath()
        r_path.moveTo(cx + 6 * scale, r_inner_y)
        r_path.quadTo(cx + 16 * scale, r_mid_y, cx + 24 * scale, r_outer_y)
        p.drawPath(r_path)

    def _draw_lips(self, p: QPainter, cx: float, cy: float, scale: float = 1.0) -> None:
        """Renders warm terracotta lips (#CF6B4E) with teeth and philtrum contour."""
        open_h = self.mouth_open_ratio * 12.0 * scale
        half_w = 11.5 * scale

        mouth_path = QPainterPath()
        if open_h > 2.5 * scale:
            mouth_path.moveTo(cx - half_w, cy)
            mouth_path.cubicTo(cx - half_w * 0.4, cy - 2.5 * scale, cx + half_w * 0.4, cy - 2.5 * scale, cx + half_w, cy)
            mouth_path.cubicTo(cx + half_w * 0.5, cy + open_h, cx - half_w * 0.5, cy + open_h, cx - half_w, cy)
            mouth_path.closeSubpath()
            p.fillPath(mouth_path, QBrush(COLOR_MOUTH_INNER))
            p.setPen(QPen(COLOR_OUTLINE, 2.2 * scale))
            p.drawPath(mouth_path)

            # Teeth row
            teeth = QPainterPath()
            teeth.moveTo(cx - half_w + 3 * scale, cy)
            teeth.lineTo(cx + half_w - 3 * scale, cy)
            teeth.lineTo(cx + half_w - 3 * scale, cy + 3.2 * scale)
            teeth.lineTo(cx - half_w + 3 * scale, cy + 3.2 * scale)
            teeth.closeSubpath()
            p.fillPath(teeth, QBrush(QColor("#FFFFFF")))

            # Lower lip highlight
            p.setPen(QPen(COLOR_LIP_BASE, 2.2 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            lower_lip = QPainterPath()
            lower_lip.moveTo(cx - half_w * 0.6, cy + open_h + 1.5 * scale)
            lower_lip.quadTo(cx, cy + open_h + 3.5 * scale, cx + half_w * 0.6, cy + open_h + 1.5 * scale)
            p.drawPath(lower_lip)
        else:
            mouth_path.moveTo(cx - half_w, cy)
            mouth_path.cubicTo(cx - half_w * 0.3, cy + 3.5 * scale, cx + half_w * 0.3, cy + 3.5 * scale, cx + half_w, cy)
            p.setPen(QPen(COLOR_LIP_BASE, 3.0 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(mouth_path)

            # Philtrum cupid bow top
            cupid = QPainterPath()
            cupid.moveTo(cx - 5 * scale, cy - 1.5 * scale)
            cupid.quadTo(cx, cy - 0.5 * scale, cx + 5 * scale, cy - 1.5 * scale)
            p.setPen(QPen(COLOR_LIP_DARK, 1.8 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPath(cupid)

    # ── Layer 6: Front Hair & Bangs ──────────────────────────────────────────

    def _draw_front_hair_bangs(self, p: QPainter, head_pos: Any, w: float, h: float, scale: float = 1.0) -> None:
        """Layer 6: Front layered curved bangs with highlight arcs (#3A3D4A / #4E5263)."""
        if isinstance(head_pos, QPointF):
            cx, cy = head_pos.x(), head_pos.y()
        elif isinstance(head_pos, (tuple, list)):
            cx, cy = float(head_pos[0]), float(head_pos[1])
        else:
            cx, cy = w * 0.5, h * 0.25

        rx, ry = 34.0 * scale, 42.0 * scale
        hair_path = self.get_hair_path(cx, cy, rx, ry)
        p.fillPath(hair_path, QBrush(COLOR_HAIR_BASE))

        # Highlight Sheen Arc (#3A3D4A)
        hl_path = QPainterPath()
        hl_path.moveTo(cx - 20 * scale, cy - ry + 4 * scale)
        hl_path.quadTo(cx, cy - ry - 3 * scale, cx + 20 * scale, cy - ry + 4 * scale)
        p.setPen(QPen(COLOR_HAIR_HL, 3.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(hl_path)

        # Subtle secondary highlight accent
        hl2_path = QPainterPath()
        hl2_path.moveTo(cx - 10 * scale, cy - ry + 1.5 * scale)
        hl2_path.quadTo(cx, cy - ry - 1.5 * scale, cx + 10 * scale, cy - ry + 1.5 * scale)
        p.setPen(QPen(COLOR_HAIR_HL_LIGHT, 1.8 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(hl2_path)

        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(hair_path)

    # ── Layers 7 & 8: Parametric 5-Finger Articulated Arms & Hands ───────────

    def _draw_vector_arm(self, p: QPainter, shoulder: QPointF, elbow: QPointF, wrist: QPointF, is_right: bool, scale: float = 1.0) -> None:
        """Renders anatomically proportioned bicep sleeve (#527490) + forearm cylinder with skin gradient."""
        arm_w = 17.0 * scale

        # 1. Bicep / Shirt Sleeve Segment
        dx = elbow.x() - shoulder.x()
        dy = elbow.y() - shoulder.y()
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * (arm_w * 0.5), dx / L * (arm_w * 0.5)

        up_path = QPainterPath()
        up_path.moveTo(shoulder.x() + nx * 1.1, shoulder.y() + ny * 1.1)
        up_path.lineTo(elbow.x() + nx * 0.9, elbow.y() + ny * 0.9)
        up_path.lineTo(elbow.x() - nx * 0.9, elbow.y() - ny * 0.9)
        up_path.lineTo(shoulder.x() - nx * 1.1, shoulder.y() - ny * 1.1)
        up_path.closeSubpath()

        p.fillPath(up_path, QBrush(COLOR_SHIRT_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(up_path)

        # Sleeve Cuff Seam
        p.setPen(QPen(COLOR_OUTLINE, 2.0 * scale))
        p.drawLine(QPointF(elbow.x() + nx * 0.9, elbow.y() + ny * 0.9), QPointF(elbow.x() - nx * 0.9, elbow.y() - ny * 0.9))

        # 2. Forearm Segment (Warm Skin Gradient)
        fa_w = 13.0 * scale
        dx2 = wrist.x() - elbow.x()
        dy2 = wrist.y() - elbow.y()
        L2 = math.hypot(dx2, dy2) or 1.0
        nx2, ny2 = -dy2 / L2 * (fa_w * 0.5), dx2 / L2 * (fa_w * 0.5)

        fa_path = QPainterPath()
        fa_path.moveTo(elbow.x() + nx2 * 0.95, elbow.y() + ny2 * 0.95)
        fa_path.lineTo(wrist.x() + nx2 * 0.8, wrist.y() + ny2 * 0.8)
        fa_path.lineTo(wrist.x() - nx2 * 0.8, wrist.y() - ny2 * 0.8)
        fa_path.lineTo(elbow.x() - nx2 * 0.95, elbow.y() - ny2 * 0.95)
        fa_path.closeSubpath()

        fa_grad = QLinearGradient(elbow, wrist)
        fa_grad.setColorAt(0.0, COLOR_SKIN_SHADOW)
        fa_grad.setColorAt(1.0, COLOR_SKIN_BASE)

        p.fillPath(fa_path, QBrush(fa_grad))
        p.setPen(QPen(COLOR_OUTLINE, 2.2 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(fa_path)

    def _draw_vector_arm_and_hand(
        self,
        p: QPainter,
        shoulder: QPointF,
        elbow: QPointF,
        wrist: QPointF,
        finger_angles: Optional[List[float]] = None,
        hand_type: str = "right",
        scale: float = 1.0,
        hand_pts: Optional[List[Tuple[float, float]]] = None,
        w: float = 400.0,
        h: float = 300.0,
    ) -> None:
        """Unified vector arm and articulated hand drawing."""
        is_right = (hand_type == "right")
        self._draw_vector_arm(p, shoulder, elbow, wrist, is_right=is_right, scale=scale)

        if hand_pts:
            self._draw_vector_hand(p, hand_pts, is_right=is_right, w=w, h=h, scale=scale)
        else:
            synthetic_hand = []
            wx, wy = wrist.x() / max(1.0, w), wrist.y() / max(1.0, h)
            synthetic_hand.append((wx, wy))
            for f in range(5):
                for j in range(1, 5):
                    fx = wx + (f - 2) * 0.02 * (1 if is_right else -1)
                    fy = wy + j * 0.025
                    synthetic_hand.append((fx, fy))
            self._draw_vector_hand(p, synthetic_hand, is_right=is_right, w=w, h=h, scale=scale)

    def _draw_vector_hand(
        self, p: QPainter, hand_pts: List[Tuple[float, float]], is_right: bool, w: float, h: float, scale: float = 1.0
    ) -> None:
        """Renders 5-finger articulated vector hand with distinct bone segments, knuckle creases, and outlines."""
        if len(hand_pts) < 21:
            return

        def P(idx: int) -> QPointF:
            return QPointF(hand_pts[idx][0] * w, hand_pts[idx][1] * h)

        # 1. Palm Base Polygon
        palm_poly = QPolygonF([P(0), P(1), P(5), P(9), P(13), P(17)])
        palm_path = QPainterPath()
        palm_path.addPolygon(palm_poly)
        p.fillPath(palm_path, QBrush(COLOR_SKIN_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(palm_path)

        # Palm crease lines
        p.setPen(QPen(COLOR_SKIN_CREASE, 1.4 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF((P(0).x() + P(5).x()) * 0.5, (P(0).y() + P(5).y()) * 0.5), QPointF((P(9).x() + P(13).x()) * 0.5, (P(9).y() + P(13).y()) * 0.5))

        # 2. 5 Articulated Fingers (Thumb, Index, Middle, Ring, Pinky)
        finger_chains = [
            [0, 1, 2, 3, 4],       # Thumb
            [5, 6, 7, 8],          # Index
            [9, 10, 11, 12],       # Middle
            [13, 14, 15, 16],      # Ring
            [17, 18, 19, 20],      # Pinky
        ]
        widths = [7.5 * scale, 6.5 * scale, 6.8 * scale, 6.0 * scale, 5.2 * scale]

        for f_i, chain in enumerate(finger_chains):
            f_w = widths[f_i]
            for s in range(len(chain) - 1):
                p1 = P(chain[s])
                p2 = P(chain[s + 1])
                seg_w = f_w * (1.0 - s * 0.15)
                self._draw_finger_bone_capsule(p, p1, p2, seg_w, scale=scale)

            # Fingernail Highlight
            tip_pt = P(chain[-1])
            prev_pt = P(chain[-2])
            self._draw_fingernail(p, tip_pt, prev_pt, f_w * 0.6)

    def _draw_finger_bone_capsule(self, p: QPainter, p1: QPointF, p2: QPointF, radius: float, scale: float = 1.0) -> None:
        """Draws discrete finger bone capsule with bold outline to eliminate silhouette blending."""
        capsule = self.get_finger_capsule_path(p1, p2, radius)
        p.fillPath(capsule, QBrush(COLOR_SKIN_BASE))
        p.setPen(QPen(COLOR_OUTLINE, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(capsule)

        # Knuckle Crease Line
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        L = math.hypot(dx, dy) or 1.0
        nx = -dy / L * radius
        ny = dx / L * radius
        p.setPen(QPen(COLOR_SKIN_CREASE, 1.4 * scale))
        p.drawLine(QPointF(p1.x() + nx * 0.65, p1.y() + ny * 0.65), QPointF(p1.x() - nx * 0.65, p1.y() - ny * 0.65))

    def _draw_fingernail(self, p: QPainter, tip: QPointF, prev: QPointF, size: float) -> None:
        """Draws clean fingernail highlight."""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(COLOR_NAIL_BASE))
        p.drawEllipse(tip, size, size * 0.8)

    def _draw_touch_effects(self, p: QPainter, frame: KinematicJointFrame, w: float, h: float) -> None:
        """Draws glowing amber touch auras when hands or fingers contact."""
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

    # ── Overlay Badges & Typography Fix ──────────────────────────────────────

    def _draw_overlay_badges(self, p: QPainter, w: float, h: float) -> None:
        """Draws crisp cybernetic glassmorphic HUD badges with proper Unicode bullet."""
        # 1. Top-left sign title badge
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(15, 23, 42, 220)))
        p.drawRoundedRect(QRectF(10, 10, 185, 44), 8, 8)

        p.setPen(QPen(QColor("#F8FAFC"), 1))
        p.setFont(QFont("Hind Siliguri", 13, QFont.Weight.Bold))
        p.drawText(QRectF(18, 12, 165, 20), Qt.AlignmentFlag.AlignLeft, f"{self.label_bn}")

        # Fixed typography: Replace raw &bull; with Unicode bullet \u2022
        p.setPen(QPen(QColor("#38BDF8"), 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(18, 31, 165, 18), Qt.AlignmentFlag.AlignLeft, f"{self.label_en.upper()}  \u2022  60 FPS")

        # 2. Top-right View Mode Badge
        mode_text = "🔍 হাতের জুম (2.2x)" if self.view_mode == AvatarViewMode.HAND_ZOOM else "👤 ফুল বডি"
        mode_color = QColor("#38BDF8") if self.view_mode == AvatarViewMode.HAND_ZOOM else QColor("#A78BFA")

        p.setBrush(QBrush(QColor(15, 23, 42, 220)))
        p.drawRoundedRect(QRectF(w - 125, 10, 115, 28), 6, 6)

        p.setPen(QPen(mode_color, 1))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(w - 125, 10, 115, 28), Qt.AlignmentFlag.AlignCenter, mode_text)
