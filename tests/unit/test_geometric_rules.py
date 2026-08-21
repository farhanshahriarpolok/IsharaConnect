"""Unit tests for BdSLGeometricRuleEngine."""

import pytest
import numpy as np
from core_engine.vision.geometric_rule_engine import BdSLGeometricRuleEngine


def _create_hand_landmarks(finger_extensions: dict) -> np.ndarray:
    """Helper to generate synthetic (21, 3) landmarks with specified finger extensions."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.8, 0.0]  # Wrist base

    # Finger MCP, PIP, DIP, TIP indices
    fingers = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    for finger, (mcp, pip, dip, tip) in fingers.items():
        is_extended = finger_extensions.get(finger, False)
        base_x = 0.5 + (mcp - 10) * 0.03
        
        lm[mcp] = [base_x, 0.6, 0.0]
        lm[pip] = [base_x, 0.45, 0.0]
        lm[dip] = [base_x, 0.35, 0.0]
        
        if is_extended:
            # Pointing high upward
            lm[tip] = [base_x, 0.20, 0.0]
        else:
            # Curled downward toward wrist
            lm[tip] = [base_x, 0.65, 0.0]

    # Handle thumb geometry
    lm[1] = [0.45, 0.75, 0.0]  # CMC
    lm[2] = [0.40, 0.70, 0.0]  # MCP
    lm[3] = [0.35, 0.65, 0.0]  # IP
    if finger_extensions.get("thumb", False):
        lm[4] = [0.15, 0.45, 0.0]  # Extended outward
    else:
        lm[4] = [0.36, 0.60, 0.0]  # Tucked across palm near index MCP

    return lm


def test_hand_analysis_open_palm():
    """All 5 fingers extended should be flagged as open palm."""
    engine = BdSLGeometricRuleEngine()
    lm = _create_hand_landmarks({
        "thumb": True, "index": True, "middle": True, "ring": True, "pinky": True
    })

    res = engine.analyze_hand(lm)
    assert res["present"] is True
    assert res["is_open_palm"] is True
    assert res["is_fist"] is False
    assert res["extended_count"] == 5


def test_hand_analysis_fist():
    """All fingers curled should be flagged as fist."""
    engine = BdSLGeometricRuleEngine()
    lm = _create_hand_landmarks({
        "thumb": False, "index": False, "middle": False, "ring": False, "pinky": False
    })

    res = engine.analyze_hand(lm)
    assert res["present"] is True
    assert res["is_fist"] is True
    assert res["is_open_palm"] is False
    assert res["extended_count"] == 0


def test_digit_1_rule():
    """Only index finger extended should evaluate to digit '১'."""
    engine = BdSLGeometricRuleEngine()
    lm = _create_hand_landmarks({
        "thumb": False, "index": True, "middle": False, "ring": False, "pinky": False
    })

    sign, conf, status = engine.evaluate_rules(None, lm)
    assert sign == "১"
    assert conf >= 0.90
    assert len(status["checklist"]) > 0


def test_digit_2_rule():
    """Index and middle extended should evaluate to digit '২'."""
    engine = BdSLGeometricRuleEngine()
    lm = _create_hand_landmarks({
        "thumb": False, "index": True, "middle": True, "ring": False, "pinky": False
    })

    sign, conf, status = engine.evaluate_rules(None, lm)
    assert sign == "২"
    assert conf >= 0.90


def test_digit_3_rule():
    """Index, middle, and ring extended should evaluate to digit '৩'."""
    engine = BdSLGeometricRuleEngine()
    lm = _create_hand_landmarks({
        "thumb": False, "index": True, "middle": True, "ring": True, "pinky": False
    })

    sign, conf, status = engine.evaluate_rules(None, lm)
    assert sign == "৩"
    assert conf >= 0.90


def test_dual_hand_sahajjo_rule():
    """Left open palm + Right fist evaluates to 'sahajjo' (Help)."""
    engine = BdSLGeometricRuleEngine()
    left_lm = _create_hand_landmarks({
        "thumb": True, "index": True, "middle": True, "ring": True, "pinky": True
    })
    right_lm = _create_hand_landmarks({
        "thumb": False, "index": False, "middle": False, "ring": False, "pinky": False
    })

    sign, conf, status = engine.evaluate_rules(left_lm, right_lm)
    assert sign == "sahajjo"
    assert conf >= 0.88


def test_empty_landmarks_safety():
    """Empty or None landmarks should not crash."""
    engine = BdSLGeometricRuleEngine()
    sign, conf, status = engine.evaluate_rules(None, None)
    assert sign is None
    assert conf == 0.0
    assert status["posture_summary"] == "কোনো হাত শনাক্ত হয়নি"
