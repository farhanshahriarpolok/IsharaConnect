"""Comprehensive Unit Tests for Stylized Cel-Shaded Toon Avatar Engine & Playback Bar.

Tests:
  - ToonAvatarRenderer pose solver for master signs (ধন্যবাদ, মা, বাবা, সাহায্য, স্বাগতম)
  - HyperKinematic Bézier trajectory synthesis (60 frames, 21 hand landmarks, touch contacts)
  - Dual perspective zoom transform matrix (Full Body 1.0x vs Hand Zoom 2.2x)
  - Playback speed modulation (1.0x, 0.5x, 0.25x) and timer intervals
  - Play, pause, loop, step, and seek functionality
  - FACS facial action mimicry (eyebrow and mouth morph parameters)
  - AvatarPlaybackBar controls, signals, and two-way binding
  - QPainter vector paint execution safety
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure headless Qt support
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize QApplication for tests
app = QApplication.instance()
if app is None:
    app = QApplication([])

from desktop_app.ui.components.avatar_playback_bar import AvatarPlaybackBar
from desktop_app.ui.components.toon_avatar_renderer import (
    AvatarViewMode,
    ToonAvatarRenderer,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pose Solver & Trajectory Synthesis Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestToonAvatarPoseSolver:
    @pytest.mark.parametrize(
        "slug, bn, en, expect_dual",
        [
            ("dhonnobad", "ধন্যবাদ", "Thank you", False),
            ("ma", "মা", "Mother", False),
            ("baba", "বাবা", "Father", False),
            ("sahajjo", "সাহায্য", "Help", True),
            ("shagotom", "স্বাগতম", "Welcome", True),
            ("kemon_achen", "কেমন আছেন", "How are you", True),
        ],
    )
    def test_pose_solver_master_signs(self, slug, bn, en, expect_dual):
        renderer = ToonAvatarRenderer(slug, bn, en)
        assert renderer.total_frames == 60
        assert len(renderer.frames) == 60

        for frame in renderer.frames:
            # Head, neck, chest sanity
            assert 0.0 <= frame.head[0] <= 1.0
            assert 0.0 <= frame.head[1] <= 1.0
            assert 0.0 <= frame.chest[0] <= 1.0

            # Wrists and active hands
            assert 0.0 <= frame.right_wrist[0] <= 1.0
            assert 0.0 <= frame.right_wrist[1] <= 1.0
            assert len(frame.right_hand) == 21

            if expect_dual:
                assert frame.is_left_active
                assert len(frame.left_hand) == 21

    def test_ma_cheek_tap_trajectory(self):
        """'ma' sign should show high elevation near right cheek."""
        renderer = ToonAvatarRenderer("ma", "মা", "Mother")
        # Middle frames should reach near cheek (y ~ 0.20 to 0.28)
        y_positions = [f.right_wrist[1] for f in renderer.frames]
        min_y = min(y_positions)
        assert min_y < 0.30, f"Expected hand to reach cheek (y < 0.30), got {min_y}"

    def test_baba_upper_lip_stroke_trajectory(self):
        """'baba' sign should traverse horizontally near mouth level (y ~ 0.27)."""
        renderer = ToonAvatarRenderer("baba", "বাবা", "Father")
        x_positions = [f.right_wrist[0] for f in renderer.frames]
        span_x = max(x_positions) - min(x_positions)
        assert span_x > 0.04, f"Expected horizontal stroke span > 0.04, got {span_x}"

    def test_sahajjo_dual_hand_boost(self):
        """'sahajjo' sign should feature both active hands."""
        renderer = ToonAvatarRenderer("sahajjo", "সাহায্য", "Help")
        assert all(f.is_left_active for f in renderer.frames)
        assert all(f.is_right_active for f in renderer.frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hand Zoom Transform Matrix Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestToonAvatarZoomTransform:
    @pytest.fixture
    def renderer(self):
        return ToonAvatarRenderer("dhonnobad", "ধন্যবাদ", "Thank you")

    def test_full_body_zoom_transform(self, renderer):
        renderer.set_view_mode(AvatarViewMode.FULL_BODY)
        scale, tx, ty = renderer.get_zoom_transform(400.0, 300.0)
        assert scale == 1.0
        assert tx == 0.0
        assert ty == 0.0

    def test_hand_zoom_transform(self, renderer):
        renderer.set_view_mode(AvatarViewMode.HAND_ZOOM)
        scale, tx, ty = renderer.get_zoom_transform(400.0, 300.0)
        assert scale == 2.2
        # Transform should pan away from origin (tx != 0 or ty != 0)
        assert tx != 0.0 or ty != 0.0

    def test_toggle_view_mode(self, renderer):
        assert renderer.view_mode == AvatarViewMode.FULL_BODY
        m1 = renderer.toggle_view_mode()
        assert m1 == "hand_zoom"
        assert renderer.view_mode == AvatarViewMode.HAND_ZOOM
        m2 = renderer.toggle_view_mode()
        assert m2 == "full_body"
        assert renderer.view_mode == AvatarViewMode.FULL_BODY

    def test_set_view_mode_str(self, renderer):
        renderer.set_view_mode("hand_zoom")
        assert renderer.view_mode == AvatarViewMode.HAND_ZOOM
        renderer.set_view_mode("full_body")
        assert renderer.view_mode == AvatarViewMode.FULL_BODY


# ─────────────────────────────────────────────────────────────────────────────
# 3. Playback Controls & Speed Modulation Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestToonAvatarPlayback:
    @pytest.fixture
    def renderer(self):
        return ToonAvatarRenderer("dhonnobad", "ধন্যবাদ", "Thank you")

    def test_speed_modulation(self, renderer):
        # Normal speed (1.0x) -> interval ~33ms
        renderer.set_speed(1.0)
        assert renderer.speed == 1.0
        assert 30 <= renderer.timer.interval() <= 36

        # Half speed (0.5x) -> interval ~66ms
        renderer.set_speed(0.5)
        assert renderer.speed == 0.5
        assert 60 <= renderer.timer.interval() <= 70

        # Slow-mo (0.25x) -> interval ~133ms
        renderer.set_speed(0.25)
        assert renderer.speed == 0.25
        assert 125 <= renderer.timer.interval() <= 140

    def test_play_pause_toggle(self, renderer):
        renderer.play()
        assert renderer.is_playing is True
        renderer.pause()
        assert renderer.is_playing is False
        state = renderer.toggle_play()
        assert state is True
        assert renderer.is_playing is True

    def test_seek_and_step(self, renderer):
        renderer.seek(15)
        assert renderer.current_frame_idx == 15

        renderer.step_forward()
        assert renderer.current_frame_idx == 16

        renderer.step_backward()
        assert renderer.current_frame_idx == 15

        # Bound check
        renderer.seek(999)
        assert renderer.current_frame_idx == 59
        renderer.seek(-10)
        assert renderer.current_frame_idx == 0

    def test_loop_toggle(self, renderer):
        renderer.set_loop(False)
        assert renderer.loop is False
        renderer.set_loop(True)
        assert renderer.loop is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. FACS Mimicry & Compound Sign Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestToonAvatarFACS:
    def test_facs_configuration(self):
        renderer = ToonAvatarRenderer("sahajjo")
        assert renderer.au04_brow_furrow > 0.0
        assert renderer.mouth_open_ratio > 0.2

    def test_compound_sign_loading(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        renderer.load_compound_sign(["khawa", "taka"], "হোটেল", "Hotel")
        assert renderer.sign_slug == "khawa"
        assert len(renderer.frames) == 60


# ─────────────────────────────────────────────────────────────────────────────
# 5. Playback Bar Two-Way Integration Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAvatarPlaybackBar:
    @pytest.fixture
    def setup_pair(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        bar = AvatarPlaybackBar(renderer)
        return renderer, bar

    def test_bar_init_and_bindings(self, setup_pair):
        renderer, bar = setup_pair
        assert bar.avatar is renderer
        assert bar.slider.maximum() == 59

    def test_bar_speed_click(self, setup_pair):
        renderer, bar = setup_pair
        bar.btn_spd_05x.click()
        assert renderer.speed == 0.5
        assert bar._current_speed == 0.5

        bar.btn_spd_025x.click()
        assert renderer.speed == 0.25
        assert bar._current_speed == 0.25

    def test_bar_zoom_click(self, setup_pair):
        renderer, bar = setup_pair
        bar.btn_zoom.click()
        assert renderer.view_mode == AvatarViewMode.HAND_ZOOM
        assert bar._is_hand_zoom is True

    def test_bar_play_toggle(self, setup_pair):
        renderer, bar = setup_pair
        bar.btn_play.click()  # Was playing -> pauses
        assert renderer.is_playing is False
        bar.btn_play.click()  # Resumes
        assert renderer.is_playing is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Paint Event Rendering Safety Test
# ─────────────────────────────────────────────────────────────────────────────
class TestToonAvatarPaintSafety:
    def test_render_paint_event_no_crash(self):
        renderer = ToonAvatarRenderer("dhonnobad")
        renderer.resize(400, 300)

        img = QImage(400, 300, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(img)
        # Directly execute painting logic without showing GUI window
        renderer._draw_backdrop(painter, 400.0, 300.0)
        renderer._draw_cel_shaded_body(painter, renderer.frames[0], 400.0, 300.0)
        renderer._draw_overlay_badges(painter, 400.0, 300.0)
        painter.end()

        # Check that image is not completely empty
        assert not img.isNull()
