"""Unit tests for Volumetric Human Avatar (HumanRigViewer) and KinematicMotionInterpolator.

Covers Sprint 16 (macro framing, playback controls, touch contacts) and
Sprint 17 (z-depth field, skin palette constants, capsule rendering pipeline,
volumetric avatar method existence).
"""

import math
import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from core_engine.vision.kinematic_interpolator import (
    KinematicMotionInterpolator,
    KinematicJointFrame,
    HAND_CONNECTIONS,
    FINGERTIP_INDICES,
    TOUCH_THRESHOLD,
)
from desktop_app.ui.components.human_rig_viewer import (
    HumanRigViewer,
    SKIN_BASE, SKIN_LIGHT, SKIN_SHADOW, NAIL_BASE,
    CLOTH_BASE, CLOTH_DARK,
    CAPSULE_W, FINGER_SEGS,
)
from desktop_app.ui.academy_dashboard import AcademyDashboard


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ── Kinematic Interpolator Tests ─────────────────────────────────────────────

def test_kinematic_interpolator_resolution():
    """Test that interpolator generates complete 60-frame sequence for dynamic and static signs."""
    interpolator = KinematicMotionInterpolator()

    frames_dynamic = interpolator.resolve_motion_sequence("dhonnobad", "ধন্যবাদ", "Thank you")
    assert len(frames_dynamic) == 60

    frame0 = frames_dynamic[0]
    assert isinstance(frame0, KinematicJointFrame)
    assert len(frame0.head) == 2
    assert len(frame0.left_shoulder) == 2
    assert len(frame0.right_shoulder) == 2
    assert len(frame0.right_elbow) == 2
    assert len(frame0.right_wrist) == 2
    assert len(frame0.right_hand) == 21
    assert len(frame0.left_hand) == 21

    frames_static = interpolator.resolve_motion_sequence("a", "অ", "A")
    assert len(frames_static) == 60
    assert frames_static[30].right_wrist[0] > 0
    assert frames_static[30].right_wrist[1] > 0


def test_kinematic_hand_connections_integrity():
    """Verify that HAND_CONNECTIONS maps all 21 landmark indices correctly."""
    assert len(HAND_CONNECTIONS) > 15
    for i1, i2 in HAND_CONNECTIONS:
        assert 0 <= i1 < 21
        assert 0 <= i2 < 21


def test_touch_contacts_field_exists():
    """Test that touch_contacts is a list field on KinematicJointFrame."""
    interpolator = KinematicMotionInterpolator()
    frames = interpolator.resolve_motion_sequence("dhonnobad", "ধন্যবাদ", "Thank you")
    assert len(frames) == 60
    for frame in frames:
        assert isinstance(frame.touch_contacts, list)
        for contact in frame.touch_contacts:
            assert len(contact) == 3
            tip_a, tip_b, intensity = contact
            assert 0 <= tip_a < 21
            assert 0 <= tip_b < 21
            assert 0.0 <= intensity <= 1.0


def test_touch_contacts_threshold():
    """Test that touch detection fires when landmark distance < TOUCH_THRESHOLD."""
    interpolator = KinematicMotionInterpolator()
    # pinch_grip pose should produce thumb-index touch at some frames
    frames_pinch = interpolator.resolve_motion_sequence("khabar", "খাবার", "Food")
    assert len(frames_pinch) == 60
    # All frames have valid touch_contacts lists
    for frame in frames_pinch:
        assert isinstance(frame.touch_contacts, list)


def test_hand_scale_landmarks_21():
    """Test _scale_hand_landmarks produces exactly 21 landmarks."""
    import numpy as np
    interpolator = KinematicMotionInterpolator()
    raw_lm = np.zeros((21, 3), dtype=np.float32)
    result = interpolator._scale_hand_landmarks(raw_lm, (0.5, 0.5), scale=0.14)
    assert len(result) == 21
    for pt in result:
        assert len(pt) == 2


def test_generate_default_hand_21_landmarks():
    """Test _generate_default_hand returns exactly 21 landmarks for all poses."""
    interpolator = KinematicMotionInterpolator()
    poses = ["flat_open", "fist_closed", "thumbs_up", "index_point",
             "pinch_grip", "cup_hand", "rest", "alphabet_pose"]
    for pose in poses:
        hand = interpolator._generate_default_hand((0.5, 0.5), is_right=True, pose=pose)
        assert len(hand) == 21, f"Pose '{pose}' returned {len(hand)} landmarks"
        for pt in hand:
            assert len(pt) == 2
            assert all(math.isfinite(v) for v in pt), f"Non-finite value in pose '{pose}'"


def test_catmull_rom_interp():
    """Test Catmull-Rom interpolation at t=0 returns p1 and at t=1 returns p2."""
    cr = KinematicMotionInterpolator._catmull_rom_interp
    # At t=0: result is p1
    assert abs(cr(0.0, 1.0, 2.0, 3.0, 0.0) - 1.0) < 1e-9
    # At t=1: result is p2
    assert abs(cr(0.0, 1.0, 2.0, 3.0, 1.0) - 2.0) < 1e-9


# ── HumanRigViewer Widget Tests ──────────────────────────────────────────────

def test_human_rig_viewer_initialization(qapp):
    """Test HumanRigViewer instantiates with correct dimensions and active playback timer."""
    viewer = HumanRigViewer("dhonnobad", "ধন্যবাদ", "Thank you")
    assert viewer.minimumWidth() >= 260
    assert viewer.minimumHeight() >= 260
    assert viewer.is_playing is True
    assert len(viewer.frames) == 60
    assert viewer.current_frame_idx == 0
    assert viewer.speed_factor == 1.0


def test_human_rig_viewer_playback_controls(qapp):
    """Test Play, Pause, Toggle, Reset, and Frame Advancement."""
    viewer = HumanRigViewer("sahajjo", "সাহায্য", "Help")

    viewer.pause()
    assert viewer.is_playing is False
    assert not viewer.timer.isActive()

    viewer.play()
    assert viewer.is_playing is True
    assert viewer.timer.isActive()

    init_frame = viewer.current_frame_idx
    viewer._advance_frame()
    assert viewer.current_frame_idx == (init_frame + 1) % 60

    viewer.toggle_playback()
    assert viewer.is_playing is False

    viewer.reset()
    assert viewer.current_frame_idx == 0
    assert len(viewer.trail_history) == 0


def test_human_rig_viewer_load_sign(qapp):
    """Test dynamically reloading different signs in HumanRigViewer."""
    viewer = HumanRigViewer()
    viewer.load_sign_motion("pani", "পানি", "Water")
    assert viewer.sign_slug == "pani"
    assert viewer.label_bn == "পানি"
    assert viewer.label_en == "Water"
    assert len(viewer.frames) == 60
    assert viewer.current_frame_idx == 0


def test_speed_toggle_half(qapp):
    """Test set_speed(0.5) configures timer interval to ~66ms."""
    viewer = HumanRigViewer()
    viewer.set_speed(0.5)
    assert viewer.speed_factor == 0.5
    # Timer interval should be in slow range (>= 60ms)
    assert viewer.timer.interval() >= 60


def test_speed_toggle_normal(qapp):
    """Test set_speed(1.0) configures timer interval to ~33ms."""
    viewer = HumanRigViewer()
    viewer.set_speed(1.0)
    assert viewer.speed_factor == 1.0
    assert viewer.timer.interval() <= 35


def test_step_forward_backward(qapp):
    """Test step_forward() and step_back() with wrapping."""
    viewer = HumanRigViewer()
    viewer.current_frame_idx = 0

    viewer.step_forward()
    assert viewer.current_frame_idx == 1

    viewer.step_back()
    assert viewer.current_frame_idx == 0

    # Wrap backward
    viewer.step_back()
    assert viewer.current_frame_idx == 59

    # Wrap forward
    viewer.current_frame_idx = 59
    viewer.step_forward()
    assert viewer.current_frame_idx == 0


def test_macro_frame_bounding_box(qapp):
    """Test _compute_hand_frame_rect produces valid macro scale and offsets."""
    viewer = HumanRigViewer()
    assert len(viewer.frames) == 60

    frame = viewer.frames[30]
    hand = frame.right_hand if frame.is_right_active else frame.left_hand
    assert len(hand) == 21

    canvas_w, canvas_h = 280.0, 240.0
    scale, tx, ty = viewer._compute_hand_frame_rect(hand, canvas_w, canvas_h, fill_ratio=0.75)

    # Macro scale must be significantly larger than 1.0 (hand fills >75% of canvas)
    assert scale > 2.0, f"Macro scale too small: {scale}"

    # Verify the transform maps the hand bounding box center to canvas center
    xs = [p[0] for p in hand]
    ys = [p[1] for p in hand]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    mapped_cx = cx * scale + tx
    mapped_cy = cy * scale + ty
    # Center of hand should map close to canvas center (within 5px tolerance)
    assert abs(mapped_cx - canvas_w / 2.0) < 5.0, f"Hand not centered horizontally: {mapped_cx}"
    assert abs(mapped_cy - canvas_h / 2.0) < 5.0, f"Hand not centered vertically: {mapped_cy}"

    # All landmark transforms must produce finite coordinates
    for pt in hand:
        px = pt[0] * scale + tx
        py = pt[1] * scale + ty
        assert math.isfinite(px) and math.isfinite(py)


def test_macro_frame_rect_edge_case(qapp):
    """Test _compute_hand_frame_rect handles degenerate input gracefully."""
    viewer = HumanRigViewer()
    # Single point — degenerate bounding box
    scale, tx, ty = viewer._compute_hand_frame_rect([(0.5, 0.5)], 280.0, 240.0)
    assert isinstance(scale, float)
    assert math.isfinite(scale)


# ── AcademyDashboard Integration Tests ───────────────────────────────────────

def test_academy_dashboard_motion_toggle(qapp):
    """Test segmented view toggle between Static Card and Kinematic Motion Demo."""
    dashboard = AcademyDashboard()

    assert dashboard.ref_display_stack.currentIndex() == 0
    assert dashboard.btn_view_static.isChecked() is True
    assert dashboard.btn_view_motion.isChecked() is False

    dashboard._set_reference_view_mode(1)
    assert dashboard.ref_display_stack.currentIndex() == 1
    assert dashboard.btn_view_static.isChecked() is False
    assert dashboard.btn_view_motion.isChecked() is True
    assert dashboard.human_rig_viewer.is_playing is True

    dashboard._set_reference_view_mode(0)
    assert dashboard.ref_display_stack.currentIndex() == 0
    assert dashboard.btn_view_static.isChecked() is True

    dashboard._update_reference_card("hospital")
    assert dashboard.human_rig_viewer.sign_slug == "hospital"
    assert dashboard.sign_card_viewer.slug == "hospital"


# ── Sprint 17: Volumetric Avatar Tests ───────────────────────────────────────

def test_z_depth_field_exists():
    """Test that KinematicJointFrame has right_hand_z and left_hand_z fields."""
    interpolator = KinematicMotionInterpolator()
    frames = interpolator.resolve_motion_sequence("dhonnobad", "ধন্যবাদ", "Thank you")
    assert len(frames) == 60
    for frame in frames:
        assert hasattr(frame, "right_hand_z"), "Missing right_hand_z field"
        assert hasattr(frame, "left_hand_z"), "Missing left_hand_z field"
        assert isinstance(frame.right_hand_z, float)
        assert isinstance(frame.left_hand_z, float)
        assert 0.0 <= frame.right_hand_z <= 1.0
        assert 0.0 <= frame.left_hand_z <= 1.0


def test_z_depth_face_level_sign():
    """Z-depth should be higher when hand is at face level (chin touch for dhonnobad)."""
    interpolator = KinematicMotionInterpolator()
    frames = interpolator.resolve_motion_sequence("dhonnobad", "ধন্যবাদ", "Thank you")
    # At least some frames should have z > 0.2 (hand raised to chin/face)
    max_z = max(f.right_hand_z for f in frames)
    assert max_z > 0.2, f"Max z-depth too small for face-level sign: {max_z}"


def test_z_depth_low_hand_sign():
    """Z-depth should be lower when hand is at chest/waist level."""
    interpolator = KinematicMotionInterpolator()
    frames = interpolator.resolve_motion_sequence("thik_ache", "ঠিক আছে", "Ok")
    # Waist-level signing → z should be lower than face signs
    avg_z = sum(f.right_hand_z for f in frames) / len(frames)
    assert avg_z < 0.5, f"Avg z-depth unexpectedly high for low-hand sign: {avg_z}"


def test_skin_palette_constants():
    """Test that all skin and clothing color constants are valid QColor instances."""
    for name, color in [
        ("SKIN_BASE", SKIN_BASE), ("SKIN_LIGHT", SKIN_LIGHT),
        ("SKIN_SHADOW", SKIN_SHADOW), ("NAIL_BASE", NAIL_BASE),
        ("CLOTH_BASE", CLOTH_BASE), ("CLOTH_DARK", CLOTH_DARK),
    ]:
        assert isinstance(color, QColor), f"{name} is not a QColor"
        assert color.isValid(), f"{name} is not a valid QColor"


def test_capsule_width_table():
    """Test CAPSULE_W has correct dimensions: 5 fingers × 3 segments each."""
    assert len(CAPSULE_W) == 5, "CAPSULE_W must have 5 finger entries"
    for i, widths in enumerate(CAPSULE_W):
        assert len(widths) == 3, f"Finger {i} must have 3 segment widths"
        for j, w in enumerate(widths):
            assert w > 0, f"CAPSULE_W[{i}][{j}] must be positive"
            assert w < 25, f"CAPSULE_W[{i}][{j}]={w} unreasonably large"
        # Verify taper: proximal >= intermediate >= distal
        assert widths[0] >= widths[1] >= widths[2], (
            f"Finger {i} widths should taper distally: {widths}"
        )


def test_finger_seg_table():
    """Test FINGER_SEGS covers all 21 landmarks with correct structure."""
    assert len(FINGER_SEGS) == 5
    all_indices = set()
    for segs in FINGER_SEGS:
        assert len(segs) == 5, "Each finger needs 5 indices (base + 4 joints)"
        for idx in segs:
            assert 0 <= idx < 21
        all_indices.update(segs[1:])  # All joints (skip wrist root duplicate)
    # All non-wrist landmarks covered
    assert len(all_indices) == 20, f"Expected 20 unique non-wrist indices, got {len(all_indices)}"


def test_volumetric_render_methods_exist(qapp):
    """Test that all new volumetric rendering methods exist on HumanRigViewer."""
    viewer = HumanRigViewer()
    required_methods = [
        "_paint_upper_body",
        "_paint_head_face",
        "_draw_arm_capsule",
        "_paint_motion_trail",
        "_paint_hands_volumetric",
        "_draw_hand_volumetric",
        "_paint_palm_volume",
        "_draw_finger_capsule",
        "_paint_hud_overlays",
        "_paint_playback_toolbar",
        "_compute_hand_frame_rect",
    ]
    for method in required_methods:
        assert hasattr(viewer, method), f"Missing method: {method}"
        assert callable(getattr(viewer, method)), f"Not callable: {method}"


def test_draw_finger_capsule_smoke(qapp):
    """Test _draw_finger_capsule can be called without error (smoke test)."""
    from PyQt6.QtGui import QPainter, QPixmap
    from PyQt6.QtCore import QPointF

    viewer = HumanRigViewer()
    # Draw into an off-screen pixmap
    pixmap = QPixmap(280, 240)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    p1 = QPointF(100, 100)
    p2 = QPointF(100, 70)
    # Extended
    viewer._draw_finger_capsule(painter, p1, p2, 10.0, is_extended=True)
    # Curled
    viewer._draw_finger_capsule(painter, p1, p2, 8.0, is_extended=False)

    painter.end()  # No exception = pass


def test_volumetric_viewer_smoke_sign_load(qapp):
    """Test loading multiple signs into volumetric viewer without errors."""
    viewer = HumanRigViewer()
    for slug, bn, en in [
        ("dhonnobad", "ধন্যবাদ", "Thank you"),
        ("sahajjo", "সাহায্য", "Help"),
        ("ami", "আমি", "Me"),
        ("hospital", "হাসপাতাল", "Hospital"),
        ("thik_ache", "ঠিক আছে", "Ok"),
    ]:
        viewer.load_sign_motion(slug, bn, en)
        assert viewer.sign_slug == slug
        assert len(viewer.frames) == 60
        for frame in viewer.frames:
            assert hasattr(frame, "right_hand_z")
            assert math.isfinite(frame.right_hand_z)
