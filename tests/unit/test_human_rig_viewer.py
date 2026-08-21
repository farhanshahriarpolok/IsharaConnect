"""Unit tests for 2D Kinematic Human Rig Viewer and Motion Interpolation Engine."""

import pytest
from PyQt6.QtWidgets import QApplication

from core_engine.vision.kinematic_interpolator import (
    KinematicMotionInterpolator,
    KinematicJointFrame,
    HAND_CONNECTIONS
)
from desktop_app.ui.components.human_rig_viewer import HumanRigViewer
from desktop_app.ui.academy_dashboard import AcademyDashboard


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_kinematic_interpolator_resolution():
    """Test that interpolator generates complete 60-frame sequence for dynamic and static signs."""
    interpolator = KinematicMotionInterpolator()
    
    # 1. Dynamic vocabulary sign
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

    # 2. Static alphabet sign
    frames_static = interpolator.resolve_motion_sequence("a", "অ", "A")
    assert len(frames_static) == 60
    assert frames_static[30].right_wrist[0] > 0
    assert frames_static[30].right_wrist[1] > 0


def test_kinematic_hand_connections_integrity():
    """Verify that HAND_CONNECTIONS maps all 21 landmark indices correctly without out-of-bounds."""
    assert len(HAND_CONNECTIONS) > 15
    for i1, i2 in HAND_CONNECTIONS:
        assert 0 <= i1 < 21
        assert 0 <= i2 < 21


def test_human_rig_viewer_initialization(qapp):
    """Test HumanRigViewer instantiates with correct dimensions and active playback timer."""
    viewer = HumanRigViewer("dhonnobad", "ধন্যবাদ", "Thank you")
    assert viewer.minimumWidth() >= 260
    assert viewer.minimumHeight() >= 220
    assert viewer.is_playing is True
    assert len(viewer.frames) == 60
    assert viewer.current_frame_idx == 0


def test_human_rig_viewer_playback_controls(qapp):
    """Test Play, Pause, Toggle, Reset, and Frame Advancement."""
    viewer = HumanRigViewer("sahajjo", "সাহায্য", "Help")
    
    # Pause
    viewer.pause()
    assert viewer.is_playing is False
    assert not viewer.timer.isActive()

    # Play
    viewer.play()
    assert viewer.is_playing is True
    assert viewer.timer.isActive()

    # Advance Frame
    init_frame = viewer.current_frame_idx
    viewer._advance_frame()
    assert viewer.current_frame_idx == (init_frame + 1) % 60
    assert len(viewer.trail_history) >= 1

    # Toggle Playback
    viewer.toggle_playback()
    assert viewer.is_playing is False

    # Reset
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


def test_academy_dashboard_motion_toggle(qapp):
    """Test segmented view toggle between Static Card and Kinematic Motion Demo in AcademyDashboard."""
    dashboard = AcademyDashboard()
    
    # Initial state should be Static Card (0)
    assert dashboard.ref_display_stack.currentIndex() == 0
    assert dashboard.btn_view_static.isChecked() is True
    assert dashboard.btn_view_motion.isChecked() is False

    # Switch to Motion Demo (1)
    dashboard._set_reference_view_mode(1)
    assert dashboard.ref_display_stack.currentIndex() == 1
    assert dashboard.btn_view_static.isChecked() is False
    assert dashboard.btn_view_motion.isChecked() is True
    assert dashboard.human_rig_viewer.is_playing is True

    # Switch back to Static Card (0)
    dashboard._set_reference_view_mode(0)
    assert dashboard.ref_display_stack.currentIndex() == 0
    assert dashboard.btn_view_static.isChecked() is True

    # Changing reference sign updates both card and rig
    dashboard._update_reference_card("hospital")
    assert dashboard.human_rig_viewer.sign_slug == "hospital"
    assert dashboard.sign_card_viewer.slug == "hospital"
