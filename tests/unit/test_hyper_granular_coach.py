"""Unit Test Suite for Hyper-Granular Articulatory Coach & Scale-Invariant Normalizer (Sprint 31).

Tests:
1. Scale and translation invariance of SpatialNormalizer across far vs close camera distances.
2. 15-Joint 3D interior angle signature extraction.
3. Strict Hand Selection & Handedness Gate (RIGHT_ONLY, LEFT_ONLY, DUAL_HAND).
4. Finger-by-finger anatomical state classification and hyper-specific Bengali corrective hints.
5. 4-Row HUD diagnostic checklist rows generation.
6. Master BdSL Lexicon articulatory specifications query.
"""

import numpy as np
import pytest

from core_engine.nlp.master_lexicon import master_lexicon
from core_engine.vision.spatial_normalizer import SpatialNormalizer
from core_engine.vision.sign_correction_advisor import SignCorrectionAdvisor, DiagnosticResult


def _create_mock_hand_scale(
    wrist_xy: tuple = (0.50, 0.38),
    scale_factor: float = 1.0,
    fingers_up: list = [True, True, True, True, True]
) -> np.ndarray:
    """Creates synthetic hand landmarks with scalable distance from camera."""
    lm = np.zeros((21, 3), dtype=np.float32)
    wx, wy = wrist_xy
    lm[0] = [wx, wy, 0.0]

    # Thumb
    lm[1] = [wx - 0.03 * scale_factor, wy - 0.02 * scale_factor, 0.0]
    lm[2] = [wx - 0.05 * scale_factor, wy - 0.04 * scale_factor, 0.0]
    lm[3] = [wx - 0.07 * scale_factor, wy - 0.06 * scale_factor, 0.0]
    lm[4] = [wx - (0.09 if fingers_up[0] else 0.04) * scale_factor, wy - (0.08 if fingers_up[0] else 0.01) * scale_factor, (0.0 if fingers_up[0] else 0.04) * scale_factor]

    # Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    offsets = [(0.02, 5), (0.04, 9), (0.06, 13), (0.08, 17)]
    for f_idx, (x_off, base) in enumerate(offsets):
        ext = fingers_up[f_idx + 1]
        bx = wx + x_off * scale_factor
        lm[base] = [bx, wy - 0.05 * scale_factor, 0.0]
        if ext:
            lm[base + 1] = [bx, wy - 0.09 * scale_factor, 0.0]
            lm[base + 2] = [bx, wy - 0.13 * scale_factor, 0.0]
            lm[base + 3] = [bx, wy - 0.17 * scale_factor, 0.0]
        else:
            lm[base + 1] = [bx, wy - 0.08 * scale_factor, 0.03 * scale_factor]
            lm[base + 2] = [bx, wy - 0.05 * scale_factor, 0.05 * scale_factor]
            lm[base + 3] = [bx, wy - 0.02 * scale_factor, 0.03 * scale_factor]

    return lm


def test_spatial_normalizer_scale_invariance():
    """Verify landmarks normalized from close vs far camera positions produce matching 15-joint angles."""
    normalizer = SpatialNormalizer()
    
    # Hand close to camera (scale = 1.8) vs far from camera (scale = 0.6)
    hand_close = _create_mock_hand_scale((0.40, 0.30), scale_factor=1.8, fingers_up=[True, True, False, False, False])
    hand_far = _create_mock_hand_scale((0.60, 0.50), scale_factor=0.6, fingers_up=[True, True, False, False, False])

    norm_close = normalizer.normalize_landmarks(hand_close)
    norm_far = normalizer.normalize_landmarks(hand_far)

    # Wrist must be at origin for both
    np.testing.assert_allclose(norm_close[0], [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(norm_far[0], [0, 0, 0], atol=1e-5)

    angles_close = normalizer.calculate_15_joint_angles(norm_close)
    angles_far = normalizer.calculate_15_joint_angles(norm_far)

    assert len(angles_close) == 15
    assert len(angles_far) == 15
    # 15 joint angles should be identical regardless of distance/scale
    np.testing.assert_allclose(angles_close, angles_far, atol=1.0)


def test_handedness_enforcement_left_vs_right():
    """Verify using Left hand for Right-only sign triggers error and explicit guidance."""
    advisor = SignCorrectionAdvisor()
    hand_left = _create_mock_hand_scale((0.50, 0.38), scale_factor=1.0, fingers_up=[True, True, True, True, True])

    # Provide left hand for 'dhonnobad' (Right hand sign)
    diag = advisor.evaluate_user_posture("dhonnobad", right_landmarks=None, left_landmarks=hand_left)
    assert diag.is_match is False
    assert diag.channel_status["handedness"] == "error"
    assert any("ভুল হাত" in h and "ডান হাত" in h for h in diag.corrective_hints)
    assert any("ডান হাত ব্যবহার করুন" in row["text"] for row in diag.checklist_rows if row["row"] == 1)


def test_finger_by_finger_specific_corrective_hints():
    """Verify specific individual finger errors produce exact localized Bengali guidance."""
    advisor = SignCorrectionAdvisor()

    # Create posture with curled index and extended middle for 'baba' (where index should be extended/hook)
    hand_wrong_fingers = _create_mock_hand_scale((0.50, 0.32), scale_factor=1.0, fingers_up=[False, False, True, False, False])

    diag = advisor.evaluate_user_posture("baba", right_landmarks=hand_wrong_fingers)
    assert diag.channel_status["fingers"] != "ok"
    # Should specifically advise extending index or curling middle
    assert any("তর্জনী" in h for h in diag.corrective_hints)


def test_15_joint_angle_signature_extraction():
    """Verify 15-joint angles extracted correctly for fully extended hand vs fist."""
    normalizer = SpatialNormalizer()
    hand_open = _create_mock_hand_scale((0.50, 0.50), scale_factor=1.0, fingers_up=[True, True, True, True, True])
    hand_fist = _create_mock_hand_scale((0.50, 0.50), scale_factor=1.0, fingers_up=[False, False, False, False, False])

    angles_open = normalizer.calculate_15_joint_angles(hand_open)
    angles_fist = normalizer.calculate_15_joint_angles(hand_fist)

    # Open hand has straight joints (~180 degrees)
    assert angles_open[4] > 150.0  # Index PIP
    assert angles_open[7] > 150.0  # Middle PIP

    # Fist has bent/curled joints
    assert angles_fist[4] < 120.0  # Index PIP curled


def test_checklist_rows_4_line_structure():
    """Verify 4-row live HUD diagnostic checklist card format."""
    advisor = SignCorrectionAdvisor()
    hand_right = _create_mock_hand_scale((0.50, 0.38), scale_factor=1.0, fingers_up=[True, True, True, True, True])

    diag = advisor.evaluate_user_posture("dhonnobad", right_landmarks=hand_right)
    rows = diag.checklist_rows
    assert len(rows) == 4
    assert rows[0]["title"] == "হাত"
    assert rows[1]["title"] == "অবস্থান"
    assert rows[2]["title"] == "আঙুল"
    assert rows[3]["title"] == "তালু"
    assert rows[0]["status"] == "ok"


def test_master_lexicon_articulatory_spec_query():
    """Verify master lexicon articulatory spec retrieval returns 4-step instructions."""
    spec = master_lexicon.get_articulatory_spec("baba")
    assert spec is not None
    assert spec["slug"] == "baba"
    assert spec["required_hand"] == "RIGHT_ONLY"
    assert spec["target_body_anchor"] in ["UPPER_LIP", "LIP_UPPER", "PHILTRUM"]
    assert "step_1_hand" in spec["instructions_bn"]
    assert "step_2_location" in spec["instructions_bn"]
    assert "step_3_fingers" in spec["instructions_bn"]
    assert "step_4_palm_action" in spec["instructions_bn"]
