"""Unit tests for CameraWorker loop resilience, universal NaN guards, and HUD overlay safety."""

import math
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage

from desktop_app.ui.components.camera_hud_overlay import CameraHUDOverlay, _safe_conf
from desktop_app.ui.components.circular_gauge import CircularAccuracyGauge
from desktop_app.controllers.camera_worker import CameraWorker
from core_engine.inference.ensemble_predictor import EnsemblePredictor, PredictionLatch


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_safe_conf_helper():
    """Verify _safe_conf sanitizes various invalid or edge-case confidence inputs."""
    assert _safe_conf(float('nan')) == 0.0
    assert _safe_conf(float('inf')) == 0.0
    assert _safe_conf(float('-inf')) == 0.0
    assert _safe_conf(None) == 0.0
    assert _safe_conf("invalid") == 0.0
    assert _safe_conf(-0.5) == 0.0
    assert _safe_conf(1.5) == 1.0
    assert _safe_conf(0.85) == 0.85
    assert _safe_conf(1) == 1.0
    assert _safe_conf(0) == 0.0


def test_hud_overlay_nan_confidence_resilience():
    """Test CameraHUDOverlay.draw_hud with various malformed confidence and landmark payloads."""
    hud = CameraHUDOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. NaN confidence payload
    pred_nan = {"label_bn": "ধন্যবাদ", "confidence": float("nan"), "source": "engine"}
    out_frame = hud.draw_hud(frame, prediction_payload=pred_nan, fps=30.0)
    assert isinstance(out_frame, np.ndarray)
    assert out_frame.shape == (480, 640, 3)

    # 2. None confidence payload
    pred_none = {"label_bn": "ধন্যবাদ", "confidence": None}
    out_frame = hud.draw_hud(frame, prediction_payload=pred_none, fps=float("nan"))
    assert isinstance(out_frame, np.ndarray)

    # 3. Inf confidence payload
    pred_inf = {"label_bn": "ধন্যবাদ", "confidence": float("inf")}
    out_frame = hud.draw_hud(frame, prediction_payload=pred_inf, fps=float("-inf"))
    assert isinstance(out_frame, np.ndarray)

    # 4. Negative confidence payload
    pred_neg = {"label_bn": "ধন্যবাদ", "confidence": -1.0}
    out_frame = hud.draw_hud(frame, prediction_payload=pred_neg, fps=None)
    assert isinstance(out_frame, np.ndarray)


def test_hud_overlay_nan_landmarks_resilience():
    """Test CameraHUDOverlay handles corrupt / NaN landmark arrays without crashing."""
    hud = CameraHUDOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Landmarks containing NaNs
    corrupt_lm = np.full((21, 3), np.nan)
    pred = {"label_bn": "ধন্যবাদ", "confidence": 0.95}

    out_frame = hud.draw_hud(
        frame,
        left_landmarks=corrupt_lm,
        right_landmarks=corrupt_lm,
        prediction_payload=pred,
        fps=30.0
    )
    assert isinstance(out_frame, np.ndarray)
    assert out_frame.shape == (480, 640, 3)

    # Valid landmarks
    valid_lm = np.zeros((21, 3), dtype=np.float32)
    valid_lm[:, 0] = 0.5
    valid_lm[:, 1] = 0.5
    out_frame_valid = hud.draw_hud(
        frame,
        left_landmarks=valid_lm,
        right_landmarks=valid_lm,
        prediction_payload=pred,
        fps=30.0
    )
    assert isinstance(out_frame_valid, np.ndarray)


def test_circular_accuracy_gauge_nan_protection(qapp):
    """Test CircularAccuracyGauge handles NaN and Inf without throwing ValueError."""
    gauge = CircularAccuracyGauge()
    
    # Set NaN
    gauge.set_value(float("nan"))
    assert gauge.value == 0.0
    
    # Set Inf
    gauge.set_value(float("inf"))
    assert gauge.value == 0.0

    # Set None
    gauge.set_value(None)
    assert gauge.value == 0.0

    # Valid value
    gauge.set_value(88.5)
    assert gauge.value == 88.5


def test_prediction_latch_nan_confidence():
    """Test PredictionLatch gracefully handles NaN confidence."""
    latch = PredictionLatch()
    raw_pred = {"label_bn": "ধন্যবাদ", "confidence": float("nan")}
    
    out, is_new = latch.process(raw_pred)
    assert not is_new


def test_camera_worker_fallback_frame_generation(qapp):
    """Test CameraWorker fallback frame generator outputs valid QImage."""
    worker = CameraWorker(camera_id=99)
    fallback_frame = worker._generate_fallback_frame(0.5)
    assert isinstance(fallback_frame, np.ndarray)
    assert fallback_frame.shape == (480, 640, 3)
