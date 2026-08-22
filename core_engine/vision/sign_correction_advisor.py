"""Hyper-Granular Articulatory Diagnostic Coach for Bangla Sign Language (BdSL).

Performs strict 5-channel anatomical validation using scale-invariant joint angle signatures:
1. Hand Selection & Handedness Gate (RIGHT_ONLY, LEFT_ONLY, DUAL_HAND)
2. Body Anchor Proximity & Location Envelope (Chin, Upper Lip, Cheek, Forehead, Chest, Wrist)
3. 5-Finger Anatomical State Matrix (Thumb, Index, Middle, Ring, Pinky via 15-joint angles)
4. 3D Palm Plane Normal Facing Vector (FACING_CAMERA, FACING_USER, FACING_UP, FACING_DOWN)
5. Facial Non-Manual Markers (NMM FACS Action Units: AU01, AU02, AU04, AU12)

Generates localized, actionable Bengali guidance messages and 4-row live HUD diagnostic checklist cards.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core_engine.nlp.master_lexicon import master_lexicon
from core_engine.vision.spatial_normalizer import SpatialNormalizer, BODY_ANCHOR_MAP

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Articulatory diagnostic result and corrective feedback container."""
    match_score: float                  # Overall match 0.0 - 100.0%
    channel_scores: Dict[str, float]    # Individual channel scores 0.0 - 1.0
    channel_status: Dict[str, str]      # "ok", "warn", "error" for each channel
    is_match: bool                      # True if match_score >= threshold and no critical errors
    corrective_hints: List[str]         # Hyper-specific Bengali guidance hints
    checklist_rows: List[Dict[str, str]]  # 4-row HUD checklist display items
    target_gloss: str
    target_bn: str
    target_en: str
    dominant_issue: Optional[str] = None


class SignCorrectionAdvisor:
    """Hyper-Granular Articulatory Diagnostic Coach analyzing 15-joint kinematic signatures."""

    def __init__(self, acceptance_threshold: float = 85.0):
        self.acceptance_threshold = acceptance_threshold
        self.master_lexicon = master_lexicon
        self.normalizer = SpatialNormalizer()

    def evaluate_user_posture(
        self,
        target_sign: str,
        right_landmarks: Optional[np.ndarray] = None,
        left_landmarks: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
        pose_landmarks: Optional[np.ndarray] = None,
        trajectory_3d: Optional[np.ndarray] = None
    ) -> DiagnosticResult:
        """Evaluates live posture across 5 channels with exact finger-by-finger and handedness checks."""
        spec = self.master_lexicon.get_articulatory_spec(target_sign)
        slug = spec.get("slug", target_sign)

        # ── Channel 1: Hand Selection & Handedness ───────────────────────────
        req_hand = spec.get("required_hand", "RIGHT_ONLY")
        hand_score, hand_status, hand_hint, active_lm = self._eval_handedness(
            req_hand, right_landmarks, left_landmarks
        )

        # ── Channel 2: Spatial Body Anchor Proximity ─────────────────────────
        target_anchor = spec.get("target_body_anchor", "NEUTRAL_SPACE")
        pos_score, pos_status, pos_hint, dist_cm = self._eval_anchor_position(
            active_lm, target_anchor, face_landmarks
        )

        # ── Channel 3: 5-Finger Articulation & 15-Joint Angles ────────────────
        target_fingers = spec.get("finger_states", {})
        finger_score, finger_status, finger_hints, detected_fingers = self._eval_finger_states(
            active_lm, target_fingers
        )

        # ── Channel 4: Palm Plane Normal Direction ───────────────────────────
        target_facing = spec.get("palm_facing", "FACING_CAMERA")
        palm_score, palm_status, palm_hint, detected_facing = self._eval_palm_facing(
            active_lm, target_facing
        )

        # ── Channel 5: Facial Non-Manual Markers (NMM FACS) ──────────────────
        facs_score, facs_status, facs_hint = self._eval_facs(
            face_landmarks, spec
        )

        channel_scores = {
            "handedness": hand_score,
            "position": pos_score,
            "fingers": finger_score,
            "orientation": palm_score,
            "facs": facs_score,
            "handshape": finger_score,  # Alias for backward compatibility
        }

        channel_status = {
            "handedness": hand_status,
            "position": pos_status,
            "fingers": finger_status,
            "orientation": palm_status,
            "facs": facs_status,
            "handshape": finger_status,  # Alias for backward compatibility
        }

        # Weighted Composite Score: Fingers (40%) + Anchor Position (30%) + Palm Direction (20%) + FACS (10%)
        # Handedness acts as a hard gating multiplier
        raw_composite = (
            finger_score * 0.40 +
            pos_score * 0.30 +
            palm_score * 0.20 +
            facs_score * 0.10
        )
        match_score = round(raw_composite * hand_score * 100.0, 1)

        # Hard gating for critical errors
        has_critical_error = (
            hand_status == "error"
            or pos_status == "error"
            or finger_status == "error"
            or (spec.get("facs_mandatory") and facs_status == "error")
        )
        is_match = (match_score >= self.acceptance_threshold) and not has_critical_error

        # Aggregate prioritized Bengali corrective hints
        hints: List[str] = []
        if not is_match:
            if hand_hint:
                hints.append(hand_hint)
            if pos_hint:
                hints.append(pos_hint)
            hints.extend(finger_hints)
            if palm_hint:
                hints.append(palm_hint)
            if facs_hint:
                hints.append(facs_hint)
        else:
            hints.append("ভঙ্গি চমৎকার ও নিখুঁত! এভাবে ২ সেকেন্ড ধরে রাখুন।")

        # ── Build 4-Row HUD Checklist Card Data ──────────────────────────────
        checklist_rows = self._build_checklist_rows(
            req_hand, hand_status,
            target_anchor, pos_status,
            detected_fingers, target_fingers, finger_status,
            target_facing, detected_facing, palm_status
        )

        dominant_issue = None
        if not is_match:
            dominant_issue = min(channel_scores, key=channel_scores.get)

        return DiagnosticResult(
            match_score=match_score,
            channel_scores=channel_scores,
            channel_status=channel_status,
            is_match=is_match,
            corrective_hints=hints[:4] if hints else ["ক্যামেরার সামনে হাত স্পষ্ট রাখুন।"],
            checklist_rows=checklist_rows,
            target_gloss=slug,
            target_bn=spec.get("label_bn", slug),
            target_en=spec.get("label_en", slug),
            dominant_issue=dominant_issue
        )

    # ── Channel 1: Handedness Evaluation ─────────────────────────────────────

    def _eval_handedness(
        self,
        req_hand: str,
        r_lm: Optional[np.ndarray],
        l_lm: Optional[np.ndarray]
    ) -> Tuple[float, str, Optional[str], Optional[np.ndarray]]:
        has_right = r_lm is not None and len(r_lm) >= 21 and not np.isnan(r_lm).any() and np.any(r_lm)
        has_left = l_lm is not None and len(l_lm) >= 21 and not np.isnan(l_lm).any() and np.any(l_lm)

        if req_hand == "RIGHT_ONLY":
            if has_right:
                return 1.0, "ok", None, r_lm
            elif has_left:
                return 0.0, "error", "⚠️ ভুল হাত! অনুগ্রহ করে ডান হাত ব্যবহার করুন।", l_lm
            else:
                return 0.0, "error", "⚠️ ডান হাত ক্যামেরার সামনে দৃশ্যমান রাখুন।", None

        elif req_hand == "LEFT_ONLY":
            if has_left:
                return 1.0, "ok", None, l_lm
            elif has_right:
                return 0.0, "error", "⚠️ ভুল হাত! অনুগ্রহ করে বাম হাত ব্যবহার করুন।", r_lm
            else:
                return 0.0, "error", "⚠️ বাম হাত ক্যামেরার সামনে দৃশ্যমান রাখুন।", None

        elif req_hand == "DUAL_HAND":
            if has_right and has_left:
                return 1.0, "ok", None, r_lm
            else:
                return 0.4, "warn", "⚠️ উভয় হাত (দুই হাত) ক্যামেরার সামনে সমান উচ্চতায় প্রস্তুত রাখুন।", r_lm if has_right else l_lm

        return 1.0, "ok", None, r_lm if has_right else l_lm

    # ── Channel 2: Body Anchor Position Evaluation ───────────────────────────

    def _eval_anchor_position(
        self,
        active_lm: Optional[np.ndarray],
        target_anchor: str,
        face_lm: Optional[np.ndarray]
    ) -> Tuple[float, str, Optional[str], float]:
        if active_lm is None or len(active_lm) < 21:
            return 0.0, "error", "⚠️ হাতের অবস্থান শনাক্ত করা যায়নি।", 50.0

        wrist = active_lm[0]
        score, dist_cm = self.normalizer.calculate_anchor_proximity(
            wrist, target_anchor, face_lm
        )

        anchor_bn_map = {
            "CHIN": "চিবুকের কাছে",
            "UPPER_LIP": "ঠোঁটের ওপর (গোঁফের কাছে)",
            "LIP_UPPER": "ঠোঁটের ওপর",
            "PHILTRUM": "মুখের কাছে",
            "CHEEK": "ডান গালের কাছে",
            "CHEEK_RIGHT": "ডান গালের কাছে",
            "CHEEK_LEFT": "বাম গালের কাছে",
            "FOREHEAD": "কপালের সামনে",
            "CHEST": "বুকের সামনে",
            "CHEST_MID": "বুকের মাঝে",
            "LEFT_WRIST": "বাম হাতের কবজির ওপর",
            "NEUTRAL_SPACE": "ক্যামেরা ফ্রেমের মাঝে"
        }
        loc_str = anchor_bn_map.get(target_anchor.upper(), "নির্দিষ্ট অবস্থানে")

        hint = None
        if score < 0.70:
            if wrist[1] > 0.65 and target_anchor in ["CHIN", "UPPER_LIP", "FOREHEAD", "CHEEK", "CHEEK_RIGHT"]:
                hint = f"⚠️ হাত নিচে রয়েছে। হাতটি উপরে {loc_str} তুলুন।"
            else:
                hint = f"⚠️ হাতটি {loc_str} নিয়ে আসুন।"

        status = "ok" if score >= 0.75 else "warn" if score >= 0.45 else "error"
        return score, status, hint, dist_cm

    # ── Channel 3: 5-Finger Anatomical States Evaluation ──────────────────────

    def _eval_finger_states(
        self,
        active_lm: Optional[np.ndarray],
        target_states: Dict[str, str]
    ) -> Tuple[float, str, List[str], Dict[str, str]]:
        if active_lm is None or len(active_lm) < 21:
            return 0.0, "error", ["⚠️ হাত ক্যামেরার সামনে রাখুন।"], {}

        norm_lm = self.normalizer.normalize_landmarks(active_lm)
        angles_15 = self.normalizer.calculate_15_joint_angles(norm_lm)
        detected_states = self.normalizer.detect_finger_states(norm_lm, angles_15)

        finger_names_bn = {
            "thumb": "বৃদ্ধাঙ্গুলি",
            "index": "তর্জনী",
            "middle": "মধ্যমা",
            "ring": "অনামিকা",
            "pinky": "কনিষ্ঠা"
        }

        correct_count = 0
        hints = []

        for f_name in ["thumb", "index", "middle", "ring", "pinky"]:
            target_st = target_states.get(f_name, "EXTENDED")
            act_st = detected_states.get(f_name, "CURL_FULL")
            bn_name = finger_names_bn[f_name]

            if act_st == target_st or (target_st == "EXTENDED" and act_st == "EXTENDED"):
                correct_count += 1
            else:
                if target_st == "EXTENDED":
                    hints.append(f"⚠️ {bn_name} আঙুলটি সম্পূর্ণ সোজা রাখুন।")
                elif target_st == "CURL_FULL":
                    hints.append(f"⚠️ {bn_name} আঙুল মুষ্টিবদ্ধ করুন।")
                elif target_st == "HOOK_BENT":
                    hints.append(f"⚠️ {bn_name} আঙুলটি সামান্য বাঁকান (হুক আকৃতি)।")
                elif target_st in ["TOUCHING_INDEX", "TOUCHING_THUMB"]:
                    hints.append("⚠️ বৃদ্ধাঙ্গুলি ও তর্জনীর ডগা স্পর্শ করে গোলক বানান।")
                elif target_st == "ACROSS_PALM":
                    hints.append(f"⚠️ {bn_name} তালুর ভেতরের দিকে বাঁকান।")

        score = correct_count / 5.0
        status = "ok" if score >= 0.80 else "warn" if score >= 0.50 else "error"
        return score, status, hints, detected_states

    # ── Channel 4: Palm Facing Direction Evaluation ──────────────────────────

    def _eval_palm_facing(
        self,
        active_lm: Optional[np.ndarray],
        target_facing: str
    ) -> Tuple[float, str, Optional[str], str]:
        if active_lm is None or len(active_lm) < 21:
            return 0.0, "error", "⚠️ তালুর অভিমুখ শনাক্ত করা যায়নি।", "UNKNOWN"

        detected_facing = self.normalizer.detect_palm_facing(active_lm)

        score = 1.0 if detected_facing == target_facing else 0.4
        hint = None

        if score < 0.7:
            if target_facing == "FACING_CAMERA":
                hint = "⚠️ হাতের তালু ক্যামেরার দিকে (সামনের দিকে) ঘোরান।"
            elif target_facing == "FACING_USER":
                hint = "⚠️ হাতের তালু নিজের দিকে ঘোরান।"
            elif target_facing == "FACING_UP":
                hint = "⚠️ হাতের তালু উপরের দিকে রাখুন।"
            elif target_facing == "FACING_DOWN":
                hint = "⚠️ হাতের তালু নিচের দিকে রাখুন।"
            else:
                hint = "⚠️ হাতের তালু নির্দেশিত দিকে প্রস্তুত রাখুন।"

        status = "ok" if score >= 0.75 else "warn"
        return score, status, hint, detected_facing

    # ── Channel 5: Facial NMM FACS Evaluation ────────────────────────────────

    def _eval_facs(
        self,
        face_lm: Optional[np.ndarray],
        spec: Dict[str, Any]
    ) -> Tuple[float, str, Optional[str]]:
        req_facs = spec.get("facs_action_units", spec.get("facs_required", {}))
        if not req_facs:
            return 1.0, "ok", None

        is_mandatory = spec.get("facs_mandatory", False)

        hint = None
        if "AU04" in req_facs and req_facs["AU04"] > 0.3:
            hint = "⚠️ প্রশ্নবোধক বাক্য—দয়া করে ভ্রু সামান্য কুঁচকান (AU04)।"
        elif "AU12" in req_facs and req_facs["AU12"] > 0.3:
            hint = "⚠️ মুখে মৃদু হাসি রাখুন (AU12)।"
        elif "AU01" in req_facs and req_facs["AU01"] > 0.3:
            hint = "⚠️ ভ্রু কিছুটা উপরে তুলুন (AU01/AU02)।"

        if face_lm is None or len(face_lm) < 10:
            if is_mandatory:
                return 0.0, "error", hint
            return 0.8, "ok", None

        return 1.0, "ok", None

    # ── 4-Row HUD Checklist Builder ──────────────────────────────────────────

    @staticmethod
    def _build_checklist_rows(
        req_hand: str, hand_status: str,
        target_anchor: str, pos_status: str,
        detected_fingers: Dict[str, str], target_fingers: Dict[str, str], finger_status: str,
        target_facing: str, detected_facing: str, palm_status: str
    ) -> List[Dict[str, str]]:
        """Constructs 4 structured status rows for rendering inside the camera HUD."""
        # Row 1: Hand
        hand_text = "[ডান হাত ✅]" if req_hand == "RIGHT_ONLY" and hand_status == "ok" else (
            "[উভয় হাত ✅]" if req_hand == "DUAL_HAND" and hand_status == "ok" else "[⚠️ ডান হাত ব্যবহার করুন]"
        )

        # Row 2: Location
        anchor_bn = {
            "CHIN": "চিবুক",
            "UPPER_LIP": "গোঁফ/ঠোঁট",
            "LIP_UPPER": "গোঁফ/ঠোঁট",
            "CHEEK": "গাল",
            "CHEEK_RIGHT": "গাল",
            "FOREHEAD": "কপাল",
            "CHEST": "বুক",
            "CHEST_MID": "বুক",
            "NEUTRAL_SPACE": "ফ্রেমের মাঝে"
        }.get(target_anchor.upper(), "নির্দিষ্ট স্থান")
        pos_text = f"[{anchor_bn} ✅]" if pos_status == "ok" else f"[⚠️ {anchor_bn}ে তুলুন]"

        # Row 3: Fingers
        if finger_status == "ok":
            finger_text = "[আঙুলসমূহ নিখুঁত ✅]"
        else:
            # List specific errors
            wrong_fingers = [f for f, t in target_fingers.items() if detected_fingers.get(f) != t]
            if "index" in wrong_fingers:
                finger_text = "[তর্জনী ❌ সোজা রাখুন]"
            elif "thumb" in wrong_fingers:
                finger_text = "[বুড়ো আঙুল ❌ বাঁকান]"
            else:
                finger_text = "[আঙুল ❌ প্রস্তুত করুন]"

        # Row 4: Palm Direction
        facing_bn = {
            "FACING_CAMERA": "সামনে",
            "FACING_USER": "নিজের দিকে",
            "FACING_UP": "উপরে",
            "FACING_DOWN": "নিচে"
        }.get(target_facing, "সঠিক দিকে")
        palm_text = f"[{facing_bn} ✅]" if palm_status == "ok" else f"[⚠️ তালু {facing_bn} ঘোরান]"

        return [
            {"row": 1, "icon": "✋", "title": "হাত", "status": hand_status, "text": hand_text},
            {"row": 2, "icon": "📍", "title": "অবস্থান", "status": pos_status, "text": pos_text},
            {"row": 3, "icon": "🖐️", "title": "আঙুল", "status": finger_status, "text": finger_text},
            {"row": 4, "icon": "🔄", "title": "তালু", "status": palm_status, "text": palm_text},
        ]
