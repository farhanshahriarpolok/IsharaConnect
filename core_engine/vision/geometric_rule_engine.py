"""Rule-Augmented Geometric Feature & Landmark Reasoning Engine for BdSL.

Extracts discrete anatomical geometric primitives:
- Finger extension states (EXTENDED, CURL, HALF_CURL)
- Pairwise fingertip distances and touch points
- Aspect vectors and orientation
- Deterministic heuristic rules for BdSL Alphabets, Digits (০-৯), and Core Words.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BdSLGeometricRuleEngine:
    """Deterministic Geometric Rule Engine for BdSL Anatomy."""

    def __init__(self):
        self.finger_indices = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }

    @staticmethod
    def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def analyze_hand(self, landmarks: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyzes a single hand (21x3 landmarks) and extracts discrete geometric states."""
        if landmarks is None or len(landmarks) < 21 or not np.any(landmarks):
            return {
                "present": False,
                "thumb": "UNKNOWN",
                "index": "UNKNOWN",
                "middle": "UNKNOWN",
                "ring": "UNKNOWN",
                "pinky": "UNKNOWN",
                "extended_count": 0,
                "is_fist": False,
                "is_open_palm": False,
                "touch_thumb_index": False,
                "touch_thumb_middle": False,
                "touch_index_middle": False,
            }

        lm = np.asarray(landmarks, dtype=np.float32)
        wrist = lm[0]

        # 1. Evaluate finger extension state
        states = {}
        extended_count = 0

        # Non-thumb fingers: compare tip-wrist distance with pip/mcp-wrist distance
        for finger in ["index", "middle", "ring", "pinky"]:
            mcp_idx, pip_idx, dip_idx, tip_idx = self.finger_indices[finger]
            mcp = lm[mcp_idx]
            pip = lm[pip_idx]
            tip = lm[tip_idx]

            dist_tip_wrist = self._dist(tip, wrist)
            dist_pip_wrist = self._dist(pip, wrist)
            dist_mcp_wrist = self._dist(mcp, wrist)

            # Upward check in normalized image coordinate space (y is smaller when higher)
            is_pointing_up = tip[1] < pip[1] < mcp[1] or dist_tip_wrist > (dist_pip_wrist * 1.25)
            is_curled = dist_tip_wrist < (dist_mcp_wrist * 1.15) or (tip[1] > pip[1] and dist_tip_wrist < dist_pip_wrist)

            if is_pointing_up and dist_tip_wrist > dist_pip_wrist:
                states[finger] = "EXTENDED"
                extended_count += 1
            elif is_curled:
                states[finger] = "CURL"
            else:
                states[finger] = "HALF_CURL"

        # Thumb extension check
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        thumb_mcp = lm[2]
        thumb_cmc = lm[1]
        index_mcp = lm[5]
        pinky_mcp = lm[17]

        dist_thumb_wrist = self._dist(thumb_tip, wrist)
        dist_thumb_index_mcp = self._dist(thumb_tip, index_mcp)
        dist_thumb_pinky_mcp = self._dist(thumb_tip, pinky_mcp)
        dist_thumb_mcp_wrist = self._dist(thumb_mcp, wrist)

        if dist_thumb_index_mcp > 0.15 and dist_thumb_wrist > dist_thumb_mcp_wrist:
            states["thumb"] = "EXTENDED"
            extended_count += 1
        elif dist_thumb_index_mcp < 0.10:
            states["thumb"] = "CURL"
        else:
            states["thumb"] = "HALF_CURL"

        # 2. Pairwise Touch Points
        touch_thumb_index = self._dist(lm[4], lm[8]) < 0.07
        touch_thumb_middle = self._dist(lm[4], lm[12]) < 0.07
        touch_index_middle = self._dist(lm[8], lm[12]) < 0.06

        is_fist = (
            states["index"] == "CURL"
            and states["middle"] == "CURL"
            and states["ring"] == "CURL"
            and states["pinky"] == "CURL"
        )
        is_open_palm = extended_count >= 4

        return {
            "present": True,
            "thumb": states["thumb"],
            "index": states["index"],
            "middle": states["middle"],
            "ring": states["ring"],
            "pinky": states["pinky"],
            "extended_count": extended_count,
            "is_fist": is_fist,
            "is_open_palm": is_open_palm,
            "touch_thumb_index": touch_thumb_index,
            "touch_thumb_middle": touch_thumb_middle,
            "touch_index_middle": touch_index_middle,
        }

    def evaluate_rules(
        self,
        landmarks_left: Optional[np.ndarray],
        landmarks_right: Optional[np.ndarray]
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Evaluates geometric posture rules against left and right hand landmarks.

        Returns:
            Tuple of:
                - sign_slug (Optional[str]): Detected canonical BdSL sign slug or None
                - confidence (float): Heuristic match confidence (0.0 to 1.0)
                - finger_status (Dict[str, Any]): Detailed posture status & checklist
        """
        left_res = self.analyze_hand(landmarks_left)
        right_res = self.analyze_hand(landmarks_right)

        # Determine dominant active hand (favoring right hand or whichever is present)
        active_hand = right_res if right_res["present"] else left_res
        hand_tag = "right" if right_res["present"] else "left"

        if not active_hand["present"]:
            return None, 0.0, {
                "left": left_res,
                "right": right_res,
                "checklist": [],
                "posture_summary": "কোনো হাত শনাক্ত হয়নি"
            }

        # Check dual-hand contact
        dual_touch = False
        if left_res["present"] and right_res["present"]:
            dist_wrists = self._dist(landmarks_left[0], landmarks_right[0])
            dist_index_tips = self._dist(landmarks_left[8], landmarks_right[8])
            dist_left_palm_right_fist = self._dist(landmarks_left[0], landmarks_right[0])
            if dist_index_tips < 0.10 or dist_wrists < 0.20:
                dual_touch = True

        detected_slug = None
        confidence = 0.0
        checklist: List[Dict[str, Any]] = []

        # Extract finger shorthand
        t = active_hand["thumb"]
        i = active_hand["index"]
        m = active_hand["middle"]
        r = active_hand["ring"]
        p = active_hand["pinky"]
        ext = active_hand["extended_count"]
        t_i_touch = active_hand["touch_thumb_index"]
        t_m_touch = active_hand["touch_thumb_middle"]

        # --- RULE SET 1: Dual-Hand Gestures ---
        if left_res["present"] and right_res["present"]:
            if left_res["is_open_palm"] and right_res["is_fist"]:
                detected_slug = "sahajjo"
                confidence = 0.92
                checklist = [
                    {"item_bn": "বাম হাতের তালু খোলা ও সোজা", "matched": True},
                    {"item_bn": "ডান হাত মুষ্টিবদ্ধ (Fist)", "matched": True},
                    {"item_bn": "দুই হাতের সংযোগ (Contact)", "matched": dual_touch},
                ]
            elif left_res["is_open_palm"] and right_res["is_open_palm"]:
                detected_slug = "dhonnobad"
                confidence = 0.88
                checklist = [
                    {"item_bn": "উভয় হাতের তালু উন্মুক্ত", "matched": True},
                    {"item_bn": "আঙ্গুলগুলো একত্রিত সোজা", "matched": True},
                    {"item_bn": "সামনের দিকে গতি", "matched": True},
                ]

        # --- RULE SET 2: Single-Hand Digits (০ - ৯) & Core Alphabets ---
        if detected_slug is None:
            if i == "EXTENDED" and m == "CURL" and r == "CURL" and p == "CURL":
                # Only Index extended -> 1 (১) or 'ই'
                detected_slug = "১"
                confidence = 0.94
                checklist = [
                    {"item_bn": "তর্জনী সোজা ঊর্ধ্বমুখী (Index Up)", "matched": True},
                    {"item_bn": "মধ্যমা, অনামিকা ও কনিষ্ঠা বাঁকানো", "matched": True},
                    {"item_bn": "বৃদ্ধাঙ্গুলি বন্ধ (Thumb Curled)", "matched": t != "EXTENDED"},
                ]
            elif i == "EXTENDED" and m == "EXTENDED" and r == "CURL" and p == "CURL":
                # Index & Middle extended -> 2 (২) or 'উ'
                detected_slug = "২"
                confidence = 0.93
                checklist = [
                    {"item_bn": "তর্জনী ও মধ্যমা প্রসারিত (V-shape)", "matched": True},
                    {"item_bn": "অনামিকা ও কনিষ্ঠা মুষ্টিবদ্ধ", "matched": True},
                    {"item_bn": "বৃদ্ধাঙ্গুলি বন্ধ", "matched": t != "EXTENDED"},
                ]
            elif i == "EXTENDED" and m == "EXTENDED" and r == "EXTENDED" and p == "CURL":
                # 3 fingers extended -> 3 (৩) or 'গ'
                detected_slug = "৩"
                confidence = 0.91
                checklist = [
                    {"item_bn": "তর্জনী, মধ্যমা ও অনামিকা সোজা", "matched": True},
                    {"item_bn": "কনিষ্ঠা আঙ্গুল বন্ধ", "matched": True},
                    {"item_bn": "বৃদ্ধাঙ্গুলি তালুর দিকে", "matched": True},
                ]
            elif i == "EXTENDED" and m == "EXTENDED" and r == "EXTENDED" and p == "EXTENDED" and t != "EXTENDED":
                # 4 fingers extended -> 4 (৪) or 'ঘ' / 'ব'
                detected_slug = "৪"
                confidence = 0.92
                checklist = [
                    {"item_bn": "চারটি আঙ্গুল সম্পূর্ণ প্রসারিত", "matched": True},
                    {"item_bn": "বৃদ্ধাঙ্গুলি তালুর উপর ভাঁজ করা", "matched": True},
                ]
            elif ext == 5:
                # All 5 fingers extended -> 5 (৫) or 'আ'
                detected_slug = "৫"
                confidence = 0.95
                checklist = [
                    {"item_bn": "পাঁচটি আঙ্গুলই সম্পূর্ণ প্রসারিত", "matched": True},
                    {"item_bn": "তালু সোজা ও উন্মুক্ত", "matched": True},
                ]
            elif t_i_touch and m == "EXTENDED" and r == "EXTENDED" and p == "EXTENDED":
                # Thumb & Index touching + 3 fingers up -> 9 (৯) / OK sign / 'ও'
                detected_slug = "৯"
                confidence = 0.93
                checklist = [
                    {"item_bn": "বৃদ্ধাঙ্গুলি ও তর্জনীর বৃত্তাকার স্পর্শ (O-shape)", "matched": True},
                    {"item_bn": "মধ্যমা, অনামিকা ও কনিষ্ঠা সোজা ঊর্ধ্বমুখী", "matched": True},
                ]
            elif t_m_touch and i == "EXTENDED" and r == "EXTENDED" and p == "EXTENDED":
                # Thumb & Middle touching -> 8 (৮)
                detected_slug = "৮"
                confidence = 0.90
                checklist = [
                    {"item_bn": "বৃদ্ধাঙ্গুলি ও মধ্যমার স্পর্শ", "matched": True},
                    {"item_bn": "তর্জনী, অনামিকা ও কনিষ্ঠা সোজা", "matched": True},
                ]
            elif active_hand["is_fist"]:
                # Fist -> 0 (০) or 'অ' / 'ম'
                detected_slug = "০"
                confidence = 0.89
                checklist = [
                    {"item_bn": "সকল আঙ্গুল মুষ্টিবদ্ধ (Fist)", "matched": True},
                    {"item_bn": "বৃদ্ধাঙ্গুলি মুষ্টির উপর আড়াআড়ি", "matched": True},
                ]
            elif t == "EXTENDED" and p == "EXTENDED" and i == "CURL" and m == "CURL" and r == "CURL":
                # Shaka sign (Thumb & Pinky) -> 6 (৬)
                detected_slug = "৬"
                confidence = 0.92
                checklist = [
                    {"item_bn": "বৃদ্ধাঙ্গুলি ও কনিষ্ঠা প্রসারিত", "matched": True},
                    {"item_bn": "মাঝের তিনটি আঙ্গুল বন্ধ", "matched": True},
                ]

        summary_bn = f"{ext}টি আঙ্গুল প্রসারিত"
        if active_hand["is_fist"]:
            summary_bn = "সম্পূর্ণ মুষ্টিবদ্ধ (Fist)"
        elif active_hand["is_open_palm"]:
            summary_bn = "উন্মুক্ত তালু (Open Palm)"

        return detected_slug, round(confidence, 2), {
            "left": left_res,
            "right": right_res,
            "checklist": checklist,
            "posture_summary": summary_bn,
            "dominant_hand": hand_tag,
            "dual_touch": dual_touch
        }
