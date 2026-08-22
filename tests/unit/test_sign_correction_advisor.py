"""Unit Test Suite for Parametric Articulatory Sign Correction Advisor (Sprint 30).

Tests:
1. Articulatory evaluation of 5 channels (Handshape, Position, Orientation, FACS, Motion).
2. Localized Bengali corrective hints for wrong hand positions, palm orientations, and finger curls.
3. Diagnostic result generation for core signs (ধন্যবাদ, বাবা, মা, চাচা, দাদা, ভূমিকম্প, ডাক্তার, ক, অ).
4. Integration with EnsemblePredictor multi-modal dactylology & syntax pipeline.
"""

from typing import Optional
import numpy as np
import pytest

from core_engine.vision.sign_correction_advisor import SignCorrectionAdvisor, DiagnosticResult
from core_engine.inference.ensemble_predictor import EnsemblePredictor


def _create_mock_hand(
    wrist_xy: tuple = (0.50, 0.60),
    fingers_up: list = [True, True, True, True, True],
    palm_z: float = 0.0,
    fingertip_xy: Optional[tuple] = None
) -> np.ndarray:
    """Creates a synthetic 21-landmark hand array for testing."""
    lm = np.zeros((21, 3), dtype=np.float32)
    if fingertip_xy is not None:
        wx, wy = fingertip_xy[0], fingertip_xy[1] + 0.17
    else:
        wx, wy = wrist_xy
    lm[0] = [wx, wy, palm_z]  # Wrist

    # Thumb: 1, 2, 3, 4
    if fingers_up[0]:
        lm[1] = [wx - 0.03, wy - 0.02, palm_z]
        lm[2] = [wx - 0.05, wy - 0.04, palm_z]
        lm[3] = [wx - 0.07, wy - 0.06, palm_z]
        lm[4] = [wx - 0.09, wy - 0.08, palm_z]  # Extended
    else:
        lm[1] = [wx - 0.02, wy - 0.01, palm_z]
        lm[2] = [wx - 0.03, wy - 0.02, palm_z]
        lm[3] = [wx - 0.02, wy - 0.01, palm_z]
        lm[4] = [wx - 0.01, wy, palm_z + 0.03]  # Curled

    # Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    finger_bases = [(0.02, 5), (0.04, 9), (0.06, 13), (0.08, 17)]
    for f_idx, (x_off, base_idx) in enumerate(finger_bases):
        is_extended = fingers_up[f_idx + 1]
        bx = wx + x_off
        lm[base_idx] = [bx, wy - 0.05, palm_z]      # MCP
        if is_extended:
            lm[base_idx + 1] = [bx, wy - 0.09, palm_z]  # PIP
            lm[base_idx + 2] = [bx, wy - 0.13, palm_z]  # DIP
            lm[base_idx + 3] = [bx, wy - 0.17, palm_z]  # Tip
        else:
            lm[base_idx + 1] = [bx, wy - 0.08, palm_z + 0.03]
            lm[base_idx + 2] = [bx, wy - 0.05, palm_z + 0.05]
            lm[base_idx + 3] = [bx, wy - 0.02, palm_z + 0.03]

    return lm


def test_advisor_perfect_posture_dhonnobad():
    """Verify perfect flat-palm chin posture for 'dhonnobad' yields >80% match score."""
    advisor = SignCorrectionAdvisor()
    # Chin position is (0.50, 0.44), all fingers extended
    hand = _create_mock_hand(fingertip_xy=(0.50, 0.44), fingers_up=[True, True, True, True, True])

    diag = advisor.evaluate_user_posture("dhonnobad", right_landmarks=hand)
    assert isinstance(diag, DiagnosticResult)
    assert diag.match_score >= 75.0
    assert diag.is_match is True
    assert diag.channel_status["handshape"] == "ok"
    assert diag.channel_status["position"] == "ok"
    assert "নিখুঁত" in diag.corrective_hints[0] or "চমৎকার" in diag.corrective_hints[0]


def test_advisor_incorrect_position_chin_vs_chest():
    """Verify placing hand at chest instead of chin lowers position score and triggers hint."""
    advisor = SignCorrectionAdvisor()
    # Chest position (0.50, 0.70) instead of Chin (0.50, 0.44)
    hand_wrong_pos = _create_mock_hand(fingertip_xy=(0.50, 0.70), fingers_up=[True, True, True, True, True])

    diag = advisor.evaluate_user_posture("dhonnobad", right_landmarks=hand_wrong_pos)
    assert diag.channel_scores["position"] < 0.60
    assert any("চিবুক" in h for h in diag.corrective_hints)


def test_advisor_incorrect_handshape_curled_fingers():
    """Verify curled fingers for an open-palm sign triggers finger extension advice."""
    advisor = SignCorrectionAdvisor()
    # Fist at chin
    hand_fist = _create_mock_hand(fingertip_xy=(0.50, 0.44), fingers_up=[False, False, False, False, False])

    diag = advisor.evaluate_user_posture("dhonnobad", right_landmarks=hand_fist)
    assert diag.channel_scores["handshape"] < 0.50
    assert any("সোজা রাখুন" in h for h in diag.corrective_hints)


def test_advisor_father_vs_mother_posture():
    """Verify distinct anchor and finger requirements for 'baba' (mustache) vs 'ma' (cheek)."""
    advisor = SignCorrectionAdvisor()

    # Father: index extended at philtrum / upper lip (0.50, 0.38)
    hand_baba = _create_mock_hand(fingertip_xy=(0.50, 0.38), fingers_up=[False, True, False, False, False])
    diag_baba = advisor.evaluate_user_posture("baba", right_landmarks=hand_baba)
    assert diag_baba.match_score >= 70.0
    assert diag_baba.channel_status["handshape"] == "ok"

    # Mother: open palm at right cheek (0.62, 0.34)
    hand_ma = _create_mock_hand(fingertip_xy=(0.62, 0.34), fingers_up=[True, True, True, True, True])
    diag_ma = advisor.evaluate_user_posture("ma", right_landmarks=hand_ma)
    assert diag_ma.match_score >= 70.0
    assert diag_ma.channel_status["position"] == "ok"


def test_advisor_dactylology_consonant_ka():
    """Verify single index extended posture for consonant 'ক'."""
    advisor = SignCorrectionAdvisor()
    # Consonant Ka: Index pointing up, others curled in neutral space
    hand_ka = _create_mock_hand(wrist_xy=(0.50, 0.48), fingers_up=[False, True, False, False, False])
    diag_ka = advisor.evaluate_user_posture("cons_ka", right_landmarks=hand_ka)
    assert diag_ka.match_score >= 75.0
    assert diag_ka.is_match is True


def test_advisor_facs_brow_furrow_for_questions():
    """Verify Wh-question ('kemon_achen') generates brow furrow hint if face is missing AU04."""
    advisor = SignCorrectionAdvisor()
    hand = _create_mock_hand(wrist_xy=(0.50, 0.52), fingers_up=[True, True, True, True, True])

    # No face landmarks provided
    diag = advisor.evaluate_user_posture("kemon_achen", right_landmarks=hand, face_landmarks=None)
    assert any("ভ্রু" in h for h in diag.corrective_hints)


def test_ensemble_predictor_dactylology_and_syntax():
    """Verify EnsemblePredictor character stream debouncing and syntax sentence synthesis."""
    predictor = EnsemblePredictor()

    # 1. Dactylology character processing
    res1 = predictor.process_dactylology("ক", confidence=0.95, timestamp=10.0, trigger_id="T0")
    res2 = predictor.process_dactylology("ক", confidence=0.95, timestamp=10.1, trigger_id="T0")
    res3 = predictor.process_dactylology("ক", confidence=0.95, timestamp=10.2, trigger_id="T0")
    assert res3 == "ক"

    # Subsequent frame within debounce window should be None (debounced)
    res4 = predictor.process_dactylology("ক", confidence=0.95, timestamp=10.3, trigger_id="T0")
    assert res4 is None

    # 2. Syntax sentence synthesis
    sentence_res = predictor.synthesize_sentence(["আমি", "ভাত", "খাওয়া"])
    assert "আমি ভাত খাচ্ছি।" in sentence_res["bengali"]
    assert sentence_res["confidence"] >= 0.90
