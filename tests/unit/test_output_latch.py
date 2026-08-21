"""Unit tests for Stateful Gesture Output Latch & Debouncer."""

import time
import pytest
from core_engine.inference.ensemble_predictor import PredictionLatch, LatchState


def test_prediction_latch_single_trigger_cycle():
    """Verify a held sign triggers exactly once and remains in HOLDING state."""
    latch = PredictionLatch(confirmation_frames=3, rest_duration_sec=0.1, drop_threshold=0.40)
    
    mock_sign = {"label_bn": "ধন্যবাদ", "label_en": "Thank you", "confidence": 0.92}

    # Frame 1: Initial detection -> CONFIRMING
    out1, trigger1 = latch.process(mock_sign)
    assert latch.state == LatchState.CONFIRMING
    assert trigger1 is False
    assert out1["is_new_trigger"] is False

    # Frame 2: Second frame -> CONFIRMING
    out2, trigger2 = latch.process(mock_sign)
    assert latch.state == LatchState.CONFIRMING
    assert trigger2 is False

    # Frame 3: Third frame -> EMITTED (Single Trigger!)
    out3, trigger3 = latch.process(mock_sign)
    assert latch.state == LatchState.EMITTED
    assert trigger3 is True
    assert out3["is_new_trigger"] is True

    # Frame 4..10: Sustaining the same posture -> HOLDING (No repeated triggers)
    for _ in range(7):
        out_held, trigger_held = latch.process(mock_sign)
        assert latch.state == LatchState.HOLDING
        assert trigger_held is False
        assert out_held["is_new_trigger"] is False


def test_prediction_latch_rest_and_retrigger():
    """Verify dropping below confidence threshold resets the latch after rest duration."""
    latch = PredictionLatch(confirmation_frames=2, rest_duration_sec=0.05, drop_threshold=0.40)
    mock_sign1 = {"label_bn": "১", "label_en": "One", "confidence": 0.85}

    # Trigger sign 1
    latch.process(mock_sign1)
    out, triggered = latch.process(mock_sign1)
    assert triggered is True

    # Hand drops (confidence 0.10)
    low_conf = {"label_bn": "১", "label_en": "One", "confidence": 0.10}
    latch.process(low_conf)
    assert latch.state == LatchState.REST

    # Wait for rest duration
    time.sleep(0.06)
    latch.process(low_conf)
    assert latch.state == LatchState.IDLE

    # Trigger sign 1 again -> Should trigger anew!
    latch.process(mock_sign1)
    out_new, triggered_new = latch.process(mock_sign1)
    assert triggered_new is True
    assert out_new["is_new_trigger"] is True


def test_prediction_latch_direct_gesture_switch():
    """Verify switching signs directly without rest resets confirmation counter."""
    latch = PredictionLatch(confirmation_frames=2, rest_duration_sec=0.1, drop_threshold=0.40)
    sign_a = {"label_bn": "অ", "label_en": "A", "confidence": 0.90}
    sign_b = {"label_bn": "ক", "label_en": "Ka", "confidence": 0.90}

    # Trigger sign A
    latch.process(sign_a)
    _, trig_a = latch.process(sign_a)
    assert trig_a is True

    # Switch directly to sign B
    _, trig_b1 = latch.process(sign_b)
    assert trig_b1 is False
    assert latch.state == LatchState.CONFIRMING

    # Confirm sign B
    out_b, trig_b2 = latch.process(sign_b)
    assert trig_b2 is True
    assert out_b["label_bn"] == "ক"
