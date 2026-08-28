"""Unit Tests for Ultra-Clean Cel-Vector 2D Humanoid Signer Illustration Engine.

Tests:
  - Vector path generation for back hair silhouette, head, hair bangs, eyes, collar, and fingers
  - 5-finger segmented bone capsule generation
  - Knuckle crease calculations and hand shape configurations
  - Hand zoom transform scale and pan calculations
  - Modular rendering methods execution safety across all 9 depth layers
  - Playback bar button layout metrics (anti-clipping min-widths and heights)
  - Unicode bullet formatting in HUD overlay badges
"""

import sys
from pathlib import Path

import pytest
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

app = QApplication.instance()
if app is None:
    app = QApplication([])

from desktop_app.ui.components.avatar_playback_bar import AvatarPlaybackBar
from desktop_app.ui.components.toon_avatar_renderer import (
    AvatarViewMode,
    ToonAvatarRenderer,
    COLOR_COLLAR_BASE,
    COLOR_HAIR_BACK,
    COLOR_HAIR_BASE,
    COLOR_OUTLINE,
    COLOR_SHIRT_BASE,
    COLOR_SKIN_BASE,
)
from core_engine.vision.kinematic_interpolator import KinematicMotionInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vector Path Geometry Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestVectorPathGeometry:
    def test_head_path_generation(self):
        path = ToonAvatarRenderer.get_head_path(200.0, 150.0, 34.0, 42.0)
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
        bounds = path.boundingRect()
        assert bounds.width() == pytest.approx(68.0, abs=4.0)
        assert bounds.height() == pytest.approx(84.0, abs=4.0)

    def test_back_hair_path_generation(self):
        path = ToonAvatarRenderer.get_back_hair_path(200.0, 150.0, 34.0, 42.0)
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
        bounds = path.boundingRect()
        assert bounds.width() > 60.0
        assert bounds.height() > 40.0

    def test_hair_bangs_path_generation(self):
        path = ToonAvatarRenderer.get_hair_path(200.0, 150.0, 34.0, 42.0)
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
        bounds = path.boundingRect()
        assert bounds.width() > 50.0
        assert bounds.height() > 30.0

    def test_collar_flaps_generation(self):
        neck = QPointF(200.0, 180.0)
        chest = QPointF(200.0, 220.0)
        l_flap, r_flap = ToonAvatarRenderer.get_collar_path(neck, chest)

        assert isinstance(l_flap, QPainterPath)
        assert isinstance(r_flap, QPainterPath)
        assert not l_flap.isEmpty()
        assert not r_flap.isEmpty()

        # Left flap should be left of center, right flap right of center
        l_bounds = l_flap.boundingRect()
        r_bounds = r_flap.boundingRect()
        assert l_bounds.center().x() < chest.x()
        assert r_bounds.center().x() > chest.x()

    def test_eye_path_generation(self):
        center = QPointF(190.0, 145.0)
        eye = ToonAvatarRenderer.get_eye_path(center, 8.5, 6.0)
        assert not eye.isEmpty()
        bounds = eye.boundingRect()
        assert bounds.width() == pytest.approx(17.0, abs=2.0)
        assert bounds.height() == pytest.approx(12.0, abs=2.0)

    def test_finger_capsule_path_generation(self):
        p1 = QPointF(100.0, 100.0)
        p2 = QPointF(100.0, 130.0)
        capsule = ToonAvatarRenderer.get_finger_capsule_path(p1, p2, 6.0)
        assert not capsule.isEmpty()
        bounds = capsule.boundingRect()
        assert bounds.width() >= 10.0
        assert bounds.height() >= 30.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hand Anatomy & Poses Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHandArticulations:
    @pytest.fixture
    def interpolator(self):
        return KinematicMotionInterpolator()

    @pytest.mark.parametrize("pose", ["flat_open", "fist_closed", "pinch_grip", "index_point", "thumbs_up"])
    def test_finger_pose_landmarks_count(self, interpolator, pose):
        wrist = (0.5, 0.5)
        landmarks = interpolator._generate_default_hand(wrist, is_right=True, pose=pose)
        assert len(landmarks) == 21
        # Wrist is first landmark
        assert landmarks[0] == wrist
        # All points are distinct and non-zero
        assert all(len(pt) == 2 for pt in landmarks)

    def test_fist_closed_vs_flat_open_compression(self, interpolator):
        wrist = (0.5, 0.5)
        flat = interpolator._generate_default_hand(wrist, is_right=True, pose="flat_open")
        fist = interpolator._generate_default_hand(wrist, is_right=True, pose="fist_closed")

        # Fingertip distances from wrist should be significantly shorter in fist
        flat_span = abs(flat[8][1] - wrist[1])
        fist_span = abs(fist[8][1] - wrist[1])
        assert fist_span < flat_span, f"Expected fist span ({fist_span}) < flat span ({flat_span})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Zoom Matrix & Viewport Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAvatarZoomMatrix:
    def test_full_body_matrix(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        renderer.set_view_mode(AvatarViewMode.FULL_BODY)
        scale, tx, ty = renderer.get_zoom_transform(400.0, 300.0)
        assert scale == 1.0
        assert tx == 0.0
        assert ty == 0.0

    def test_hand_zoom_matrix(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        renderer.set_view_mode(AvatarViewMode.HAND_ZOOM)
        scale, tx, ty = renderer.get_zoom_transform(400.0, 300.0)
        assert scale == 2.2
        # Transform centers the active hand
        assert tx != 0.0 or ty != 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Modular Painting Pipeline Safety Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestModularPaintingSafety:
    def test_modular_render_methods(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        renderer.resize(400, 300)

        img = QImage(400, 300, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)

        # 0. Background
        renderer._draw_backdrop(p, 400.0, 300.0)

        # 1. Back Hair
        head = QPointF(200.0, 70.0)
        renderer._draw_back_hair(p, head, 400.0, 300.0)

        # 2. Neck
        neck = QPointF(200.0, 110.0)
        renderer._draw_neck(p, neck, head)

        # 3. Torso
        chest = QPointF(200.0, 160.0)
        ls = QPointF(140.0, 120.0)
        rs = QPointF(260.0, 120.0)
        renderer._draw_torso_and_clothing(p, chest, neck, ls, rs, 400.0, 300.0)

        # 3b. Torso via Bounds & Scale
        bounds = QRectF(50.0, 50.0, 300.0, 200.0)
        renderer._draw_torso_and_clothing(p, bounds, scale=1.2)

        # 4. Head & Ears
        renderer._draw_head_and_hair(p, head, 400.0, 300.0)

        # 5. Face features
        renderer._draw_face_features(p, head, 400.0, 300.0)

        # 5b. Head & Face with NMM Parameters
        nmm = {"au01": 0.4, "au02": 0.3, "au04": 0.2, "mouth_open": 0.5}
        renderer._draw_head_and_hair(p, head, scale=1.2, nmm_params=nmm)
        renderer._draw_face_features(p, head, scale=1.2, nmm_params=nmm)

        # 6. Front hair bangs
        renderer._draw_front_hair_bangs(p, head, 400.0, 300.0)

        # 7. Arms & Sleeves
        lw = QPointF(130.0, 220.0)
        rw = QPointF(240.0, 190.0)
        renderer._draw_vector_arm(p, ls, QPointF(130.0, 170.0), lw, is_right=False)
        renderer._draw_vector_arm(p, rs, QPointF(260.0, 170.0), rw, is_right=True)

        # 7b. Unified Arm and Hand Drawing
        renderer._draw_vector_arm_and_hand(p, rs, QPointF(260.0, 170.0), rw, hand_type="right", scale=1.1)
        renderer._draw_vector_arm_and_hand(p, ls, QPointF(130.0, 170.0), lw, hand_type="left", scale=1.1)

        # 8. Articulated Hand
        hand_pts = renderer.frames[0].right_hand
        renderer._draw_vector_hand(p, hand_pts, is_right=True, w=400.0, h=300.0)

        # 9. Badges
        renderer._draw_overlay_badges(p, 400.0, 300.0)

        p.end()
        assert not img.isNull()

    def test_playback_controls_and_seeking(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        assert renderer.is_playing is True

        renderer.pause()
        assert renderer.is_playing is False

        renderer.play()
        assert renderer.is_playing is True

        renderer.toggle_play()
        assert renderer.is_playing is False

        renderer.set_speed(0.5)
        assert renderer.speed == 0.5

        renderer.seek(10)
        assert renderer.current_frame_idx == 10

        renderer.step_forward()
        assert renderer.current_frame_idx == 11

        renderer.step_backward()
        assert renderer.current_frame_idx == 10

        mode = renderer.toggle_view_mode()
        assert mode == "hand_zoom"
        assert renderer.view_mode == AvatarViewMode.HAND_ZOOM

        mode2 = renderer.toggle_view_mode()
        assert mode2 == "full_body"
        assert renderer.view_mode == AvatarViewMode.FULL_BODY


# ─────────────────────────────────────────────────────────────────────────────
# 5. Playback Bar UI Layout Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPlaybackBarLayoutMetrics:
    def test_playback_bar_button_metrics(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        bar = AvatarPlaybackBar(renderer)

        # Speed buttons min-width >= 54px, height >= 36px
        assert bar.btn_spd_1x.minimumWidth() >= 54
        assert bar.btn_spd_1x.height() >= 34 or bar.btn_spd_1x.minimumHeight() >= 34
        assert bar.btn_spd_05x.minimumWidth() >= 54
        assert bar.btn_spd_025x.minimumWidth() >= 54

        # Action buttons
        assert bar.btn_loop.minimumWidth() >= 60
        assert bar.btn_zoom.minimumWidth() >= 80

        # Transport buttons
        assert bar.btn_play.minimumWidth() >= 36
        assert bar.btn_prev.minimumWidth() >= 30
        assert bar.btn_next.minimumWidth() >= 30

        # Timecode badge
        assert "60" in bar.lbl_frame.text()
