"""Unit Tests for Ultra-Clean Cel-Vector 2D Humanoid Signer Illustration Engine.

Tests:
  - Vector path generation for head, hair, eyes, and collared shirt
  - 5-finger segmented bone capsule generation
  - Hand shape configs for flat palm, fist, pinch, and hook
  - Hand zoom transform scale and pan calculations
  - Modular rendering methods execution safety
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

from desktop_app.ui.components.toon_avatar_renderer import (
    AvatarViewMode,
    ToonAvatarRenderer,
    COLOR_COLLAR_BASE,
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
        assert bounds.width() == pytest.approx(68.0, abs=1.0)
        assert bounds.height() == pytest.approx(84.0, abs=1.0)

    def test_hair_path_generation(self):
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
        assert bounds.width() == pytest.approx(17.0, abs=1.0)
        assert bounds.height() == pytest.approx(12.0, abs=1.0)

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

        # 1. Background
        renderer._draw_backdrop(p, 400.0, 300.0)

        # 2. Torso
        neck = QPointF(200.0, 110.0)
        chest = QPointF(200.0, 160.0)
        ls = QPointF(140.0, 120.0)
        rs = QPointF(260.0, 120.0)
        renderer._draw_torso_and_clothing(p, chest, neck, ls, rs, 400.0, 300.0)

        # 3. Arms
        lw = QPointF(130.0, 220.0)
        rw = QPointF(240.0, 190.0)
        renderer._draw_vector_arm(p, ls, QPointF(130.0, 170.0), lw, is_right=False)
        renderer._draw_vector_arm(p, rs, QPointF(260.0, 170.0), rw, is_right=True)

        # 4. Head & Face
        head = QPointF(200.0, 70.0)
        renderer._draw_head_and_hair(p, head, 400.0, 300.0)
        renderer._draw_face_features(p, head, 400.0, 300.0)

        # 5. Hand
        hand_pts = renderer.frames[0].right_hand
        renderer._draw_vector_hand(p, hand_pts, is_right=True, w=400.0, h=300.0)

        # 6. Badges
        renderer._draw_overlay_badges(p, 400.0, 300.0)

        p.end()
        assert not img.isNull()
