"""Hybrid Ensemble Predictor for IsharaConnect.

Fuses:
1. Rule-Augmented Geometric Landmark Engine (deterministic posture reasoning)
2. 151-D / 128-D ONNX Neural Model (deep feature classification)
3. Dynamic Time Warping (DTW) Trajectory Matcher (temporal gesture alignment)
4. Single-Trigger Hysteresis State Latch & Debouncer
"""

from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core_engine.vision.geometric_rule_engine import BdSLGeometricRuleEngine
from core_engine.vision.dtw_matcher import DTWMotionMatcher
from core_engine.inference.predictor import RealTimePredictor
from core_engine.inference.minimal_pair_discriminator import MinimalPairDiscriminator
from core_engine.nlp.master_lexicon import master_lexicon

logger = logging.getLogger(__name__)

# Standard Bengali & English Gloss dictionary for common signs
SIGN_METADATA = {
    "০": {"bn": "০ (শূন্য)", "en": "0 (Zero)", "category": "Digit"},
    "১": {"bn": "১ (এক)", "en": "1 (One)", "category": "Digit"},
    "২": {"bn": "২ (দুই)", "en": "2 (Two)", "category": "Digit"},
    "৩": {"bn": "৩ (তিন)", "en": "3 (Three)", "category": "Digit"},
    "৪": {"bn": "৪ (চার)", "en": "4 (Four)", "category": "Digit"},
    "৫": {"bn": "৫ (পাঁচ)", "en": "5 (Five)", "category": "Digit"},
    "৬": {"bn": "৬ (ছয়)", "en": "6 (Six)", "category": "Digit"},
    "৭": {"bn": "৭ (সাত)", "en": "7 (Seven)", "category": "Digit"},
    "৮": {"bn": "৮ (আট)", "en": "8 (Eight)", "category": "Digit"},
    "৯": {"bn": "৯ (নয়)", "en": "9 (Nine)", "category": "Digit"},
    "dhonnobad": {"bn": "ধন্যবাদ", "en": "Thank you", "category": "Dynamic"},
    "kemon_achen": {"bn": "কেমন আছেন", "en": "How are you", "category": "Dynamic"},
    "sahajjo": {"bn": "সাহায্য", "en": "Help", "category": "Dual-Hand"},
    "shagotom": {"bn": "স্বাগতম", "en": "Welcome", "category": "Dynamic"},
    "ami": {"bn": "আমি", "en": "I / Me", "category": "Pronoun"},
    "apni": {"bn": "আপনি", "en": "You", "category": "Pronoun"},
    "bhalo": {"bn": "ভালো", "en": "Good / Fine", "category": "Adjective"},
}


class LatchState(Enum):
    """Hysteresis state machine states for gesture triggering."""
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"
    EMITTED = "EMITTED"
    HOLDING = "HOLDING"
    REST = "REST"


class PredictionLatch:
    """Stateful Hysteresis Latch ensuring gestures are emitted exactly once per stroke."""

    def __init__(
        self,
        confirmation_frames: int = 3,
        rest_duration_sec: float = 0.3,
        drop_threshold: float = 0.40
    ):
        self.state = LatchState.IDLE
        self.current_sign: Optional[str] = None
        self.consecutive_count: int = 0
        self.confirmation_frames = confirmation_frames
        self.rest_duration_sec = rest_duration_sec
        self.drop_threshold = drop_threshold
        self.last_drop_time: float = 0.0

    def reset(self):
        """Resets the latch to IDLE."""
        self.state = LatchState.IDLE
        self.current_sign = None
        self.consecutive_count = 0
        self.last_drop_time = 0.0

    def process(self, raw_prediction: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Processes raw prediction and returns (annotated_prediction, is_new_trigger)."""
        now = time.time()

        # 1. No hand / Low confidence (< drop_threshold)
        raw_conf = raw_prediction.get("confidence", 0.0) if raw_prediction else 0.0
        if raw_conf is None or not isinstance(raw_conf, (int, float)) or np.isnan(raw_conf) or np.isinf(raw_conf):
            raw_conf = 0.0
        if raw_prediction is None or raw_conf < self.drop_threshold:
            if self.state in [LatchState.EMITTED, LatchState.HOLDING]:
                if self.last_drop_time == 0.0:
                    self.last_drop_time = now
                    self.state = LatchState.REST
                elif (now - self.last_drop_time) >= self.rest_duration_sec:
                    self.reset()
            else:
                self.reset()
            return (raw_prediction, False)

        # 2. Hand active with confidence >= drop_threshold
        self.last_drop_time = 0.0
        sign_label = raw_prediction.get("label_bn") or raw_prediction.get("label_en") or "unknown"

        if self.state in [LatchState.IDLE, LatchState.REST]:
            self.current_sign = sign_label
            self.consecutive_count = 1
            self.state = LatchState.CONFIRMING
            out = dict(raw_prediction)
            out["is_new_trigger"] = False
            out["latch_state"] = self.state.value
            return (out, False)

        elif self.state == LatchState.CONFIRMING:
            if sign_label == self.current_sign:
                self.consecutive_count += 1
                if self.consecutive_count >= self.confirmation_frames:
                    self.state = LatchState.EMITTED
                    out = dict(raw_prediction)
                    out["is_new_trigger"] = True
                    out["latch_state"] = self.state.value
                    return (out, True)
                else:
                    out = dict(raw_prediction)
                    out["is_new_trigger"] = False
                    out["latch_state"] = self.state.value
                    return (out, False)
            else:
                self.current_sign = sign_label
                self.consecutive_count = 1
                out = dict(raw_prediction)
                out["is_new_trigger"] = False
                out["latch_state"] = self.state.value
                return (out, False)

        elif self.state in [LatchState.EMITTED, LatchState.HOLDING]:
            if sign_label != self.current_sign:
                # Direct transition to a different sign without rest
                self.current_sign = sign_label
                self.consecutive_count = 1
                self.state = LatchState.CONFIRMING
                out = dict(raw_prediction)
                out["is_new_trigger"] = False
                out["latch_state"] = self.state.value
                return (out, False)
            else:
                # Sustaining the same sign posture
                self.state = LatchState.HOLDING
                out = dict(raw_prediction)
                out["is_new_trigger"] = False
                out["latch_state"] = self.state.value
                return (out, False)

        return (raw_prediction, False)


class EnsemblePredictor:
    """Ensemble inference coordinator uniting Geometric Rules, Neural Model, DTW, and Hysteresis Latch."""

    def __init__(
        self,
        neural_predictor: Optional[RealTimePredictor] = None,
        geometric_threshold: float = 0.70,
        sensitivity: str = "normal"
    ):
        self.geometric_engine = BdSLGeometricRuleEngine()
        self.dtw_matcher = DTWMotionMatcher()
        self.neural_predictor = neural_predictor or RealTimePredictor()
        self.latch = PredictionLatch()
        self.minimal_pair_discriminator = MinimalPairDiscriminator()
        self.master_lexicon = master_lexicon

        self.geometric_threshold = geometric_threshold
        self.sensitivity = sensitivity
        self._apply_sensitivity(sensitivity)

    def set_sensitivity(self, level: str):
        """Sets sensitivity mode: 'high', 'normal', or 'strict'."""
        self.sensitivity = level.lower()
        self._apply_sensitivity(self.sensitivity)

    def _apply_sensitivity(self, level: str):
        if level == "high":
            self.geometric_threshold = 0.60
            self.neural_confidence_threshold = 0.50
            self.latch.confirmation_frames = 2
            self.latch.drop_threshold = 0.30
        elif level == "strict":
            self.geometric_threshold = 0.80
            self.neural_confidence_threshold = 0.75
            self.latch.confirmation_frames = 3
            self.latch.drop_threshold = 0.45
        else:  # normal
            self.geometric_threshold = 0.70
            self.neural_confidence_threshold = 0.65
            self.latch.confirmation_frames = 2
            self.latch.drop_threshold = 0.35

    def predict_raw(
        self,
        feature_vector: Optional[np.ndarray],
        left_landmarks: Optional[np.ndarray] = None,
        right_landmarks: Optional[np.ndarray] = None,
        temporal_buffer: Optional[np.ndarray] = None,
        target_sign: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Evaluates raw prediction without latch state processing."""
        raw_res: Optional[Dict[str, Any]] = None

        # 1. Evaluate Rule-Augmented Geometric Engine
        geom_slug, geom_conf, finger_status = self.geometric_engine.evaluate_rules(
            left_landmarks, right_landmarks
        )
        if geom_conf is None or not isinstance(geom_conf, (int, float)) or np.isnan(geom_conf) or np.isinf(geom_conf):
            geom_conf = 0.0
        else:
            geom_conf = max(0.0, min(1.0, float(geom_conf)))

        # 2. Dynamic Sign DTW Evaluation (if temporal sequence available)
        if temporal_buffer is not None and len(temporal_buffer) >= 12:
            candidate = target_sign or geom_slug or "dhonnobad"
            dtw_res = self.dtw_matcher.evaluate_gesture_accuracy(temporal_buffer, candidate)
            raw_dtw_score = dtw_res.get("score", 0.0)
            if raw_dtw_score is None or not isinstance(raw_dtw_score, (int, float)) or np.isnan(raw_dtw_score) or np.isinf(raw_dtw_score):
                raw_dtw_score = 0.0
            if dtw_res.get("is_match", False) and raw_dtw_score >= (self.geometric_threshold * 100):
                meta = SIGN_METADATA.get(candidate) or self.master_lexicon.get_sign_by_gloss(candidate) or {"bn": candidate, "en": candidate}
                raw_res = {
                    "label_bn": meta.get("bn") or meta.get("label_bn", candidate),
                    "label_en": meta.get("en") or meta.get("label_en", candidate),
                    "confidence": max(0.0, min(1.0, float(raw_dtw_score / 100.0))),
                    "is_stable": True,
                    "source": "dtw",
                    "finger_status": finger_status,
                    "checklist": finger_status.get("checklist", []),
                    "dtw_score": raw_dtw_score,
                    "slug": candidate
                }

        # 3. Fast-Path: Unambiguous Static Geometric Pose
        if raw_res is None and geom_slug is not None and geom_conf >= self.geometric_threshold:
            meta = SIGN_METADATA.get(geom_slug) or self.master_lexicon.get_sign_by_gloss(geom_slug) or {"bn": geom_slug, "en": geom_slug}
            raw_res = {
                "label_bn": meta.get("bn") or meta.get("label_bn", geom_slug),
                "label_en": meta.get("en") or meta.get("label_en", geom_slug),
                "confidence": geom_conf,
                "is_stable": True,
                "source": "geometric",
                "finger_status": finger_status,
                "checklist": finger_status.get("checklist", []),
                "slug": geom_slug
            }

        # 4. Neural ONNX Classifier Processing
        if raw_res is None and feature_vector is not None and self.neural_predictor is not None:
            neural_pred = None
            try:
                neural_pred = self.neural_predictor.process_frame(feature_vector)
            except Exception as e:
                logger.debug("Neural predictor inference step: %s", e)

            if neural_pred is not None:
                if "confidence" in neural_pred:
                    n_conf = neural_pred["confidence"]
                    if n_conf is None or not isinstance(n_conf, (int, float)) or np.isnan(n_conf) or np.isinf(n_conf):
                        neural_pred["confidence"] = 0.0
                    else:
                        neural_pred["confidence"] = max(0.0, min(1.0, float(n_conf)))
                neural_pred["source"] = "neural"
                neural_pred["finger_status"] = finger_status
                neural_pred["checklist"] = finger_status.get("checklist", [])
                raw_res = neural_pred

        # 5. Soft Geometric Fallback
        if raw_res is None and geom_slug is not None and geom_conf >= (self.geometric_threshold - 0.15):
            meta = SIGN_METADATA.get(geom_slug) or self.master_lexicon.get_sign_by_gloss(geom_slug) or {"bn": geom_slug, "en": geom_slug}
            raw_res = {
                "label_bn": meta.get("bn") or meta.get("label_bn", geom_slug),
                "label_en": meta.get("en") or meta.get("label_en", geom_slug),
                "confidence": geom_conf,
                "is_stable": False,
                "source": "geometric_tentative",
                "finger_status": finger_status,
                "checklist": finger_status.get("checklist", []),
                "slug": geom_slug
            }

        if raw_res is None:
            return None

        # 6. Fine-Grained Minimal Pair Disambiguation
        active_slug = raw_res.get("slug") or raw_res.get("label_bn") or raw_res.get("label_en", "")
        if active_slug and self.minimal_pair_discriminator.identify_cluster([active_slug]):
            dis_res = self.minimal_pair_discriminator.disambiguate(
                candidate_slug=active_slug,
                trajectory_3d=temporal_buffer,
                left_landmarks=left_landmarks,
                right_landmarks=right_landmarks
            )
            if dis_res:
                raw_res["label_bn"] = dis_res["resolved_bn"]
                raw_res["slug"] = dis_res["resolved_slug"]
                raw_res["confidence"] = max(raw_res["confidence"], dis_res["confidence"])
                raw_res["minimal_pair_rationale"] = dis_res.get("rationale", "")
                raw_res["minimal_pair_resolved"] = True

        return raw_res

    def predict(
        self,
        feature_vector: Optional[np.ndarray],
        left_landmarks: Optional[np.ndarray] = None,
        right_landmarks: Optional[np.ndarray] = None,
        temporal_buffer: Optional[np.ndarray] = None,
        target_sign: Optional[str] = None,
        use_latch: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Unified prediction blending Geometric Heuristics, Dynamic DTW, Neural ONNX, and Output Latch.

        Returns:
            Latched prediction dictionary with 'is_new_trigger' flag.
        """
        raw_pred = self.predict_raw(
            feature_vector=feature_vector,
            left_landmarks=left_landmarks,
            right_landmarks=right_landmarks,
            temporal_buffer=temporal_buffer,
            target_sign=target_sign
        )

        if not use_latch:
            return raw_pred

        latched_pred, _ = self.latch.process(raw_pred)
        return latched_pred
