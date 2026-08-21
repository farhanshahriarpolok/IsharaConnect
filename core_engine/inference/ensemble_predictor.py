"""Hybrid Ensemble Predictor for IsharaConnect.

Fuses:
1. Rule-Augmented Geometric Landmark Engine (deterministic posture reasoning)
2. 151-D / 128-D ONNX Neural Model (deep feature classification)
3. Dynamic Time Warping (DTW) Trajectory Matcher (temporal gesture alignment)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core_engine.vision.geometric_rule_engine import BdSLGeometricRuleEngine
from core_engine.vision.dtw_matcher import DTWMotionMatcher
from core_engine.inference.predictor import RealTimePredictor

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


class EnsemblePredictor:
    """Ensemble inference coordinator uniting Geometric Rules, Neural Model, and DTW."""

    def __init__(
        self,
        neural_predictor: Optional[RealTimePredictor] = None,
        geometric_threshold: float = 0.82,
        sensitivity: str = "normal"
    ):
        self.geometric_engine = BdSLGeometricRuleEngine()
        self.dtw_matcher = DTWMotionMatcher()
        self.neural_predictor = neural_predictor or RealTimePredictor()
        
        self.geometric_threshold = geometric_threshold
        self.sensitivity = sensitivity
        self._apply_sensitivity(sensitivity)

    def set_sensitivity(self, level: str):
        """Sets sensitivity mode: 'high', 'normal', or 'strict'."""
        self.sensitivity = level.lower()
        self._apply_sensitivity(self.sensitivity)

    def _apply_sensitivity(self, level: str):
        if level == "high":
            self.geometric_threshold = 0.72
            self.neural_confidence_threshold = 0.60
        elif level == "strict":
            self.geometric_threshold = 0.88
            self.neural_confidence_threshold = 0.80
        else:  # normal
            self.geometric_threshold = 0.82
            self.neural_confidence_threshold = 0.70

    def predict(
        self,
        feature_vector: Optional[np.ndarray],
        left_landmarks: Optional[np.ndarray] = None,
        right_landmarks: Optional[np.ndarray] = None,
        temporal_buffer: Optional[np.ndarray] = None,
        target_sign: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Unified prediction blending Geometric Heuristics, Dynamic DTW, and Neural ONNX.

        Args:
            feature_vector: 151D or 128D normalized feature vector for neural inference.
            left_landmarks: (21, 3) raw/normalized left hand landmarks.
            right_landmarks: (21, 3) raw/normalized right hand landmarks.
            temporal_buffer: (T, 151) rolling sequence of spatial vectors for DTW.
            target_sign: Optional active sign slug being practiced.

        Returns:
            Dictionary containing prediction payload, or None if confidence is too low.
        """
        # 1. Evaluate Rule-Augmented Geometric Engine
        geom_slug, geom_conf, finger_status = self.geometric_engine.evaluate_rules(
            left_landmarks, right_landmarks
        )

        # 2. Dynamic Sign DTW Evaluation (if temporal sequence available)
        if temporal_buffer is not None and len(temporal_buffer) >= 12:
            candidate = target_sign or geom_slug or "dhonnobad"
            dtw_res = self.dtw_matcher.evaluate_gesture_accuracy(temporal_buffer, candidate)
            if dtw_res["is_match"] and dtw_res["score"] >= (self.geometric_threshold * 100):
                meta = SIGN_METADATA.get(candidate, {"bn": candidate, "en": candidate})
                return {
                    "label_bn": meta["bn"],
                    "label_en": meta["en"],
                    "confidence": float(dtw_res["score"] / 100.0),
                    "is_stable": True,
                    "source": "dtw",
                    "finger_status": finger_status,
                    "checklist": finger_status.get("checklist", []),
                    "dtw_score": dtw_res["score"]
                }

        # 3. Fast-Path: Unambiguous Static Geometric Pose
        if geom_slug is not None and geom_conf >= self.geometric_threshold:
            meta = SIGN_METADATA.get(geom_slug, {"bn": geom_slug, "en": geom_slug})
            return {
                "label_bn": meta["bn"],
                "label_en": meta["en"],
                "confidence": geom_conf,
                "is_stable": True,
                "source": "geometric",
                "finger_status": finger_status,
                "checklist": finger_status.get("checklist", [])
            }

        # 4. Neural ONNX Classifier Processing
        neural_pred = None
        if feature_vector is not None and self.neural_predictor is not None:
            try:
                neural_pred = self.neural_predictor.process_frame(feature_vector)
            except Exception as e:
                logger.debug("Neural predictor inference step: %s", e)

        if neural_pred is not None:
            # Attach live geometric finger checklist & status to neural prediction
            neural_pred["source"] = "neural"
            neural_pred["finger_status"] = finger_status
            neural_pred["checklist"] = finger_status.get("checklist", [])
            return neural_pred

        # 5. Soft Geometric Fallback (if any geometric pose was partially matched)
        if geom_slug is not None and geom_conf >= (self.geometric_threshold - 0.15):
            meta = SIGN_METADATA.get(geom_slug, {"bn": geom_slug, "en": geom_slug})
            return {
                "label_bn": meta["bn"],
                "label_en": meta["en"],
                "confidence": geom_conf,
                "is_stable": False,
                "source": "geometric_tentative",
                "finger_status": finger_status,
                "checklist": finger_status.get("checklist", [])
            }

        return None
