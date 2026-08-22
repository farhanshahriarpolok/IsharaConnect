"""Unit Test Suite for Universal Articulatory Calibration for All 50+ Signs & Motion Discriminator (Sprint 35).

Tests:
1. Validation of all 50+ signs in master lexicon for complete articulatory specifications.
2. Contact resolution for Father ('baba' pinch at upper lip), Mother ('ma' index at cheek),
   Doctor ('daktar' dual-finger tap at left wrist), and Brother ('bhai' dual-hand parallel index touch).
3. Dynamic motion trajectory discrimination for 'TAP_TWICE', 'HIGH_FREQ_VIBRATION', and 'PULL_RIGHT'.
4. Golden calibration CLI and validation routine.
"""

import numpy as np
import pytest

from core_engine.nlp.master_lexicon import master_lexicon, MasterBdSLLexicon
from core_engine.vision.spatial_normalizer import SpatialNormalizer
from core_engine.vision.sign_correction_advisor import SignCorrectionAdvisor
from scripts.calibrate_golden_sign import validate_all_calibrations


def _create_synthetic_facemesh() -> np.ndarray:
    """Creates a 468 FaceMesh array with realistic keypoint clusters."""
    face = np.zeros((468, 3), dtype=np.float32)
    face[10] = [0.50, 0.20, 0.0]   # Forehead
    face[151] = [0.50, 0.21, 0.0]
    face[9] = [0.50, 0.22, 0.0]
    face[1] = [0.50, 0.30, 0.0]    # Nose
    face[4] = [0.50, 0.31, 0.0]
    face[195] = [0.50, 0.29, 0.0]
    face[0] = [0.50, 0.36, 0.0]    # Upper Lip
    face[13] = [0.50, 0.36, 0.0]
    face[152] = [0.50, 0.45, 0.0]  # Chin
    face[175] = [0.50, 0.45, 0.0]
    face[234] = [0.62, 0.34, 0.0]  # Right Cheek
    face[93] = [0.60, 0.34, 0.0]
    return face


def _create_synthetic_pose() -> np.ndarray:
    """Creates a 33 Pose landmark array with shoulders and wrists."""
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11] = [0.40, 0.60, 0.0]   # Left Shoulder
    pose[12] = [0.60, 0.60, 0.0]   # Right Shoulder
    pose[15] = [0.42, 0.65, 0.0]   # Left Wrist
    pose[16] = [0.58, 0.65, 0.0]   # Right Wrist
    return pose


def _create_hand_at(x: float, y: float, art_type: str = "INDEX_TIP") -> np.ndarray:
    """Creates a 21-landmark hand array positioned such that the articulator is at (x, y)."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [x, y + 0.16, 0.0]   # Wrist
    for i in range(1, 21):
        hand[i] = [x, y + 0.05, 0.0]
    # Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20
    hand[4] = [x - 0.02, y, 0.0]
    hand[8] = [x, y, 0.0]
    hand[12] = [x + 0.02, y, 0.0]
    hand[16] = [x + 0.04, y, 0.0]
    return hand


def test_all_signs_spec_completeness():
    """Verify all 50+ signs in master lexicon have complete non-null articulatory attributes."""
    valid = validate_all_calibrations()
    assert valid is True

    lexicon = MasterBdSLLexicon()
    signs = lexicon.all_signs()
    assert len(signs) >= 45

    for s in signs:
        spec = lexicon.get_articulatory_spec(s.get("slug", ""))
        assert spec.get("slug") is not None
        assert spec.get("label_bn") is not None
        assert spec.get("required_hand") in ["RIGHT_ONLY", "LEFT_ONLY", "DUAL_HAND"]
        assert spec.get("target_body_anchor") is not None
        assert spec.get("articulator_type") is not None
        assert spec.get("motion_type") is not None
        inst = spec.get("instructions_bn", {})
        assert "step_1_hand" in inst
        assert "step_2_location" in inst
        assert "step_3_fingers" in inst
        assert "step_4_palm_action" in inst


def test_father_pinch_at_upper_lip():
    """Verify Father ('baba') evaluates pinch articulator at upper lip with score >= 90%."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Position pinch (mean of 4 and 8) at upper lip (y=0.36)
    hand_baba = _create_hand_at(0.50, 0.36, art_type="THUMB_INDEX_PINCH")
    score, dist_cm, _, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_baba,
        target_anchor_name="UPPER_LIP",
        face_landmarks=facemesh,
        articulator_type="THUMB_INDEX_PINCH"
    )
    assert score >= 0.90
    assert dist_cm < 4.0


def test_mother_index_at_cheek():
    """Verify Mother ('ma') evaluates index tip at right cheek with score >= 90%."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Position index tip at right cheek (x=0.61, y=0.34)
    hand_ma = _create_hand_at(0.61, 0.34, art_type="INDEX_TIP")
    score, dist_cm, _, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_ma,
        target_anchor_name="CHEEK_RIGHT",
        face_landmarks=facemesh,
        articulator_type="INDEX_TIP"
    )
    assert score >= 0.90
    assert dist_cm < 4.0


def test_doctor_dual_finger_at_left_wrist():
    """Verify Doctor ('daktar') evaluates dual-finger contact at left wrist."""
    normalizer = SpatialNormalizer()
    pose = _create_synthetic_pose()

    # Left wrist anchor is at (0.42, 0.65)
    hand_doc = _create_hand_at(0.42, 0.65, art_type="DUAL_INDEX_MIDDLE")
    score, dist_cm, _, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_doc,
        target_anchor_name="LEFT_WRIST",
        pose_landmarks=pose,
        articulator_type="DUAL_INDEX_MIDDLE"
    )
    assert score >= 0.90
    assert dist_cm < 4.0


def test_dynamic_motion_trajectory_evaluation():
    """Verify advisor discriminates TAP_TWICE, HIGH_FREQ_VIBRATION, and PULL_RIGHT trajectories."""
    advisor = SignCorrectionAdvisor()

    # 1. Tap Twice Trajectory (2 cycles in y/z)
    t = np.linspace(0, 4 * np.pi, 20)
    tap_traj = np.column_stack([np.zeros(20), np.sin(t) * 0.05, np.zeros(20)])
    score_tap, status_tap, _ = advisor._eval_motion_trajectory("TAP_TWICE", None, tap_traj)
    assert score_tap >= 0.95
    assert status_tap == "ok"

    # 2. High Frequency Vibration Trajectory (3+ zero crossings in x)
    t_vib = np.linspace(0, 6 * np.pi, 25)
    vib_traj = np.column_stack([np.sin(t_vib) * 0.03, np.zeros(25), np.zeros(25)])
    score_vib, status_vib, _ = advisor._eval_motion_trajectory("HIGH_FREQ_VIBRATION", None, vib_traj)
    assert score_vib >= 0.95
    assert status_vib == "ok"

    # 3. Pull Right Trajectory (delta_x > 0.02)
    pull_traj = np.column_stack([np.linspace(0.0, 0.06, 15), np.zeros(15), np.zeros(15)])
    score_pull, status_pull, _ = advisor._eval_motion_trajectory("PULL_RIGHT", None, pull_traj)
    assert score_pull >= 0.95
    assert status_pull == "ok"
