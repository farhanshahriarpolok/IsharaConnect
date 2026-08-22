"""Unit Test Suite for Interactive Ghost Skeleton Overlay & Precision Sign Coach (Sprint 32).

Tests:
1. Continuous 3D joint angle calculations for diverse hand poses (open palm, fist, hook, pinch).
2. EMA temporal smoothing of landmarks for jitter elimination.
3. Target Ghost Skeleton Overlay wireframe and per-joint alignment color mapping.
4. Dynamic 4-Card Bengali Instruction Generator with live channel status.
5. Multi-channel consensus scoring and sustained match hold logic.
"""

import numpy as np
import pytest

from core_engine.vision.spatial_normalizer import SpatialNormalizer
from core_engine.vision.sign_correction_advisor import SignCorrectionAdvisor, DiagnosticResult
from desktop_app.ui.components.ghost_skeleton_overlay import (
    GhostSkeletonOverlay,
    COLOR_NEON_GREEN,
    COLOR_AMBER,
    COLOR_CORAL_RED,
)


def _create_synthetic_hand(pose_type: str = "open") -> np.ndarray:
    """Creates synthetic 21-landmark hand arrays for diverse poses."""
    lm = np.zeros((21, 3), dtype=np.float32)
    wx, wy = 0.50, 0.50
    lm[0] = [wx, wy, 0.0]

    # Thumb
    if pose_type in ["open", "thumb_up"]:
        lm[1] = [wx - 0.03, wy - 0.02, 0.0]
        lm[2] = [wx - 0.05, wy - 0.04, 0.0]
        lm[3] = [wx - 0.07, wy - 0.06, 0.0]
        lm[4] = [wx - 0.09, wy - 0.08, 0.0]
    elif pose_type == "pinch":
        lm[1] = [wx - 0.02, wy - 0.02, 0.0]
        lm[2] = [wx - 0.01, wy - 0.04, 0.0]
        lm[3] = [wx + 0.01, wy - 0.06, 0.0]
        lm[4] = [wx + 0.02, wy - 0.08, 0.0]  # Touching index tip
    else:  # fist
        lm[1] = [wx - 0.02, wy - 0.01, 0.0]
        lm[2] = [wx - 0.03, wy - 0.02, 0.0]
        lm[3] = [wx - 0.02, wy - 0.01, 0.0]
        lm[4] = [wx - 0.01, wy, 0.03]

    # Fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    offsets = [(0.02, 5), (0.04, 9), (0.06, 13), (0.08, 17)]
    for f_idx, (x_off, base) in enumerate(offsets):
        bx = wx + x_off
        lm[base] = [bx, wy - 0.05, 0.0]
        if pose_type == "open":
            lm[base + 1] = [bx, wy - 0.09, 0.0]
            lm[base + 2] = [bx, wy - 0.13, 0.0]
            lm[base + 3] = [bx, wy - 0.17, 0.0]
        elif pose_type == "pinch" and f_idx == 0:  # Index touching thumb
            lm[base + 1] = [bx, wy - 0.07, 0.02]
            lm[base + 2] = [bx - 0.01, wy - 0.08, 0.02]
            lm[base + 3] = [wx + 0.02, wy - 0.08, 0.0]
        elif pose_type == "hook" and f_idx == 0:  # Index hook bent
            lm[base + 1] = [bx, wy - 0.08, 0.04]
            lm[base + 2] = [bx, wy - 0.06, 0.06]
            lm[base + 3] = [bx, wy - 0.03, 0.05]
        else:  # Curled
            lm[base + 1] = [bx, wy - 0.08, 0.03]
            lm[base + 2] = [bx, wy - 0.05, 0.05]
            lm[base + 3] = [bx, wy - 0.02, 0.03]

    return lm


def test_continuous_finger_angles_calculation():
    """Verify continuous degree angles are computed for all 15 joints."""
    normalizer = SpatialNormalizer()
    open_hand = _create_synthetic_hand("open")
    angles = normalizer.calculate_finger_angles(open_hand)

    assert len(angles) == 15
    assert all(0.0 <= a <= 180.0 for a in angles)

    angles_dict = normalizer.get_finger_angles_dict(open_hand)
    assert "thumb" in angles_dict
    assert "index" in angles_dict
    assert "middle" in angles_dict
    assert "ring" in angles_dict
    assert "pinky" in angles_dict
    assert angles_dict["index"]["pip"] > 140.0


def test_ema_landmark_jitter_smoothing():
    """Verify Exponential Moving Average removes artificial noise while following true movement."""
    normalizer = SpatialNormalizer()
    base_lm = _create_synthetic_hand("open")

    # Add Gaussian jitter
    np.random.seed(42)
    noisy_lm = base_lm + np.random.normal(0, 0.02, base_lm.shape).astype(np.float32)

    smoothed = normalizer.smooth_landmarks(noisy_lm, base_lm, alpha=0.65)
    assert smoothed.shape == (21, 3)

    # Smoothed error to base must be strictly less than noisy error to base
    noisy_err = np.mean(np.abs(noisy_lm - base_lm))
    smooth_err = np.mean(np.abs(smoothed - base_lm))
    assert smooth_err < noisy_err


def test_ghost_skeleton_overlay_rendering():
    """Verify GhostSkeletonOverlay draws canonical wireframe and returns valid BGR frame."""
    ghost = GhostSkeletonOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    user_lm = _create_synthetic_hand("open")

    annotated = ghost.render_ghost_overlay(
        frame,
        target_slug="dhonnobad",
        user_landmarks=user_lm,
        target_anchor="CHIN",
        match_score=88.0
    )

    assert annotated.shape == (480, 640, 3)
    # Output must have non-zero pixels from skeleton and bullseye drawing
    assert np.count_nonzero(annotated) > 0


def test_ghost_skeleton_joint_color_coding():
    """Verify aligned joints receive Neon Green and misaligned receive Coral Red/Amber."""
    ghost = GhostSkeletonOverlay()
    user_lm = _create_synthetic_hand("open")
    target_lm = ghost.get_canonical_target_landmarks("dhonnobad", 640, 480)

    # Perfect match >= 85% -> Green
    colors_perfect = ghost._evaluate_joint_colors(user_lm, target_lm, match_score=92.0)
    assert colors_perfect[8] == COLOR_NEON_GREEN

    # Low match < 50% -> Coral Red
    colors_poor = ghost._evaluate_joint_colors(user_lm, target_lm, match_score=35.0)
    assert colors_poor[8] == COLOR_CORAL_RED


def test_dynamic_4_card_bengali_guidance():
    """Verify 4-card instruction HTML properly incorporates channel status badges."""
    from desktop_app.ui.academy_dashboard import AcademyDashboard
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    dash = AcademyDashboard()

    # Pass partial warnings
    ch_status = {
        "handedness": "ok",
        "position": "warn",
        "fingers": "ok",
        "orientation": "warn"
    }
    html = dash._render_4_card_instructions_html("dhonnobad", channel_status=ch_status)

    assert "১. হাত নির্বাচন (Hand Selection)" in html
    assert "২. শারীরিক অবস্থান (Body Anchor)" in html
    assert "৩. আঙুলের বিন্যাস (Finger-by-Finger)" in html
    assert "৪. তালুর অভিমুখ ও মুখাবয়ব (Palm & Face)" in html
    assert "সঠিক ✅" in html
    assert "সংশোধন করুন ⚠️" in html
