"""Real-Time Predictor & Temporal Smoothing Engine.

Maintains a sliding window circular buffer and runs ONNX inference.
Applies temporal smoothing and debounce logic to ensure stability.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from core_engine.inference.config import InferenceConfig

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class RealTimePredictor:
    """Predicts BdSL signs from a stream of landmarks using ONNX Runtime."""

    def __init__(
        self,
        model_path: str = "models/onnx/bdsl_model.onnx",
        labels_path: str = "dataset/labels.json",
        config: InferenceConfig = None,
        agreement_threshold: float = 0.7
    ):
        if config is None:
            config = InferenceConfig()
            
        self.sequence_length = config.sequence_length
        self.confidence_threshold = config.confidence_threshold
        self.debounce_cooldown_sec = config.cooldown_seconds
        self.agreement_window = config.agreement_window
        self.allow_ood_rejection = config.allow_ood_rejection
        self.entropy_threshold = config.entropy_threshold
        self.agreement_threshold = agreement_threshold

        # Load labels
        self.labels = self._load_labels(labels_path)
        
        # Buffers
        # We need a buffer of shape (sequence_length, feature_dim).
        # Feature dim is 128 from normalizer
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        
        # Buffer for recent predictions to apply temporal smoothing (majority voting)
        self.prediction_buffer = deque(maxlen=self.agreement_window)
        
        # Debounce state
        self.last_emitted_id: Optional[int] = None
        self.last_emitted_time: float = 0.0

        # Load ONNX session
        self.ort_session = None
        if ort is not None and Path(model_path).exists():
            try:
                self.ort_session = ort.InferenceSession(model_path)
                self.input_name = self.ort_session.get_inputs()[0].name
                logger.info("ONNX Runtime session initialized successfully.")
            except Exception as e:
                logger.error("Failed to load ONNX model: %s", e)
        else:
            logger.warning("ONNX model %s not found or onnxruntime not installed. Inference will be disabled.", model_path)

    def _load_labels(self, path: str) -> Dict[int, dict]:
        """Load and index labels."""
        p = Path(path)
        if not p.exists():
            logger.warning("Labels file %s not found.", path)
            return {}
            
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return {sign["id"]: sign for sign in data.get("signs", [])}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def process_frame(self, landmarks: np.ndarray) -> Optional[Dict]:
        """Process a single frame's landmarks and return a prediction if stable.
        
        Args:
            landmarks: 1D numpy array of shape (128,)
            
        Returns:
            Dictionary with prediction or None.
        """
        if landmarks.shape != (128,):
            logger.warning("Expected landmarks shape (128,), got %s", landmarks.shape)
            return None
            
        # Push into buffer
        self.landmark_buffer.append(landmarks)
        
        # If buffer is not full, return None
        if len(self.landmark_buffer) < self.sequence_length:
            return None
            
        if self.ort_session is None:
            return None

        # Convert buffer to tensor shape (1, 30, 128)
        input_data = np.array(self.landmark_buffer, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0) # Add batch dimension
        
        # Run inference
        try:
            ort_outs = self.ort_session.run(None, {self.input_name: input_data})
            logits = ort_outs[0]
        except Exception as e:
            logger.error("ONNX inference failed: %s", e)
            return None

        probs = self._softmax(logits)[0] # Shape (num_classes,)
        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        
        # OOD Rejection (Unknown Sign)
        is_unknown = False
        if self.allow_ood_rejection:
            # Normalized Predictive Entropy
            K = len(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-9)) / np.log(K) if K > 1 else 0.0
            
            if confidence < self.confidence_threshold or entropy > self.entropy_threshold:
                is_unknown = True
                
        if is_unknown:
            pred_class_id = -1
            self.prediction_buffer.append(-1)
        elif confidence >= self.confidence_threshold:
            self.prediction_buffer.append(pred_class_id)
        else:
            pred_class_id = -1
            self.prediction_buffer.append(-1) # -1 represents no confident prediction
            
        # Check temporal agreement
        if len(self.prediction_buffer) == self.agreement_window:
            matches = sum(1 for p in self.prediction_buffer if p == pred_class_id)
            agreement_ratio = matches / self.agreement_window
            
            is_stable = agreement_ratio >= self.agreement_threshold
            
            # Emit Unknown if -1 is stable
            if is_stable and pred_class_id == -1:
                # Debounce logic for unknown
                current_time = time.time()
                is_debounced = False
                if self.last_emitted_id == -1:
                    if (current_time - self.last_emitted_time) < self.debounce_cooldown_sec:
                        is_debounced = True
                if not is_debounced:
                    self.last_emitted_id = -1
                    self.last_emitted_time = current_time
                    return {
                        "sign_id": -1,
                        "label_bn": "অজ্ঞাত ইশারা",
                        "label_en": "Unknown Sign",
                        "confidence": confidence,
                        "is_stable": True
                    }
                    
            elif is_stable and pred_class_id != -1:
                current_time = time.time()
                
                # Debounce logic
                is_debounced = False
                if self.last_emitted_id == pred_class_id:
                    if (current_time - self.last_emitted_time) < self.debounce_cooldown_sec:
                        is_debounced = True
                        
                if not is_debounced:
                    self.last_emitted_id = pred_class_id
                    self.last_emitted_time = current_time
                    
                    label_data = self.labels.get(pred_class_id, {})
                    
                    return {
                        "sign_id": pred_class_id,
                        "label_bn": label_data.get("bangla", "Unknown"),
                        "label_en": label_data.get("english", "Unknown"),
                        "confidence": confidence,
                        "is_stable": True
                    }
                    
        return None
