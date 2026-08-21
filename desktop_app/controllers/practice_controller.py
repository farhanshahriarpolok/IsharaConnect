import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

class PracticeSessionManager:
    """Manages state and scoring for the AI Practice Tutor."""
    
    def __init__(self, labels_path: str = "dataset/labels.json"):
        self.labels_path = Path(labels_path)
        self.labels_data = self._load_labels()
        self.signs_map = {s["id"]: s for s in self.labels_data.get("signs", [])}
        
        self.target_sign: Optional[Dict[str, Any]] = None
        
        self.pass_threshold = 0.80  # 80% confidence
        self.hold_duration = 1.5    # seconds
        
        self._consecutive_start_time = 0.0
        self._is_holding = False

    def _load_labels(self) -> dict:
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"signs": []}

    def set_target_sign(self, sign_id: int):
        """Sets the active sign to practice."""
        if sign_id in self.signs_map:
            self.target_sign = self.signs_map[sign_id]
            self.reset_session()
        else:
            raise ValueError(f"Sign ID {sign_id} not found in labels.")

    def reset_session(self):
        """Resets the timing and scoring state."""
        self._consecutive_start_time = 0.0
        self._is_holding = False

    def evaluate_frame(self, prediction: dict) -> dict:
        """
        Evaluates a frame prediction against the target sign.
        prediction dict expected from predictor.py / camera_worker.py:
        {"sign_id": int, "label_bn": str, "confidence": float, "is_stable": bool}
        """
        if not self.target_sign:
            return {"match_score": 0.0, "is_passed": False, "feedback_text": "কোন সাইন নির্বাচন করা হয়নি (No target selected)"}

        pred_id = prediction.get("sign_id", -1)
        conf = prediction.get("confidence", 0.0)
        
        # Calculate dynamic match score (0-100)
        if pred_id == self.target_sign["id"]:
            match_score = conf * 100.0
        else:
            # If a different sign is detected, match score is very low
            match_score = max(0.0, 10.0 - (conf * 100.0))
            
        feedback_text = "হাতটি সঠিকভাবে পজিশন করুন..."
        is_passed = False
        
        if pred_id == self.target_sign["id"] and conf >= self.pass_threshold:
            if not self._is_holding:
                self._is_holding = True
                self._consecutive_start_time = time.time()
                feedback_text = "ধরে রাখুন! (Hold it!)"
            else:
                elapsed = time.time() - self._consecutive_start_time
                if elapsed >= self.hold_duration:
                    is_passed = True
                    feedback_text = "দারুণ হয়েছে! ✅"
                else:
                    feedback_text = f"ধরে রাখুন... {self.hold_duration - elapsed:.1f}s"
        else:
            # Reset hold if confidence drops or wrong sign
            self._is_holding = False
            self._consecutive_start_time = 0.0
            if pred_id == self.target_sign["id"] and conf > 0.4:
                feedback_text = "কাছাকাছি! আরেকটু চেষ্টা করুন।"
            elif pred_id != -1 and conf > 0.6:
                wrong_label = prediction.get("label_bn", "Unknown")
                feedback_text = f"এটি '{wrong_label}' দেখাচ্ছে। আবার চেষ্টা করুন।"

        return {
            "match_score": match_score,
            "is_passed": is_passed,
            "feedback_text": feedback_text
        }
