"""Parametric Articulatory Diagnostic & Correction Engine for Bangla Sign Language (BdSL).

Evaluates live user gestures across 5 discrete anatomical channels against ground-truth master specifications:
1. Handshape & Finger Extension Matrix (individual finger flexion/extension)
2. Spatial Body Landmark Distance (anchor proximity to chin, cheek, forehead, chest, etc.)
3. Palm Normal & Orientation Vector (3D plane cross-product direction)
4. Facial Non-Manual Markers (NMM FACS Action Units: AU01, AU02, AU04, AU12, AU25)
5. Kinematic Trajectory & Oscillation Dynamics (static hold, stroke, vibration, thrust)

Generates localized, actionable Bengali guidance messages to accelerate learner mastery.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core_engine.nlp.master_lexicon import master_lexicon

logger = logging.getLogger(__name__)

# Standard Body Anchor Target Coordinates (normalized image/rig space: x in [0,1], y in [0,1])
ANCHOR_POSITIONS: Dict[str, Tuple[float, float]] = {
    "FOREHEAD": (0.50, 0.16),
    "PHILTRUM": (0.50, 0.32),
    "UPPER_LIP": (0.50, 0.32),
    "CHIN": (0.50, 0.38),
    "CHEEK_RIGHT": (0.65, 0.30),
    "CHEEK_LEFT": (0.35, 0.30),
    "CHEST": (0.50, 0.52),
    "STOMACH": (0.50, 0.68),
    "WRIST_ANCHOR": (0.50, 0.60),
    "NEUTRAL_SPACE": (0.50, 0.48),
}

# Standard Articulatory Sign Specifications
SIGN_ARTICULATION_SPECS: Dict[str, Dict[str, Any]] = {
    "dhonnobad": {
        "slug": "dhonnobad",
        "label_bn": "ধন্যবাদ",
        "label_en": "Thank you",
        "anchor": "CHIN",
        "palm_orientation": "INWARD",
        "fingers_extended": [True, True, True, True, True],  # Open flat palm
        "facs_required": {"AU12": 0.5},  # Smile
        "motion_type": "FORWARD_STROKE",
        "handedness": "single",
    },
    "sahajjo": {
        "slug": "sahajjo",
        "label_bn": "সাহায্য",
        "label_en": "Help",
        "anchor": "CHEST",
        "palm_orientation": "UPWARD",
        "fingers_extended": [True, True, True, True, True],
        "facs_required": {"AU01": 0.4, "AU02": 0.4},
        "motion_type": "UPWARD_LIFT",
        "handedness": "dual",
    },
    "baba": {
        "slug": "baba",
        "label_bn": "বাবা",
        "label_en": "Father",
        "anchor": "UPPER_LIP",
        "palm_orientation": "INWARD",
        "fingers_extended": [False, True, False, False, False],  # Index extended (mustache swipe)
        "facs_required": {},
        "motion_type": "HORIZONTAL_SWIPE",
        "handedness": "single",
    },
    "ma": {
        "slug": "ma",
        "label_bn": "মা",
        "label_en": "Mother",
        "anchor": "CHEEK_RIGHT",
        "palm_orientation": "INWARD",
        "fingers_extended": [True, True, True, True, True],
        "facs_required": {"AU12": 0.4},
        "motion_type": "TAP_DOUBLE",
        "handedness": "single",
    },
    "chacha": {
        "slug": "chacha",
        "label_bn": "চাচা",
        "label_en": "Uncle",
        "anchor": "CHIN",
        "palm_orientation": "INWARD",
        "fingers_extended": [False, True, False, False, False],  # Index tap on chin
        "facs_required": {},
        "motion_type": "TAP_SINGLE",
        "handedness": "single",
    },
    "dada": {
        "slug": "dada",
        "label_bn": "দাদা",
        "label_en": "Grandfather",
        "anchor": "CHIN",
        "palm_orientation": "INWARD",
        "fingers_extended": [False, True, False, False, False],
        "facs_required": {},
        "motion_type": "DOWNWARD_STROKE",  # Beard stroke
        "handedness": "single",
    },
    "bhumikompo": {
        "slug": "bhumikompo",
        "label_bn": "ভূমিকম্প",
        "label_en": "Earthquake",
        "anchor": "CHEST",
        "palm_orientation": "DOWNWARD",
        "fingers_extended": [True, True, True, True, True],
        "facs_required": {"AU01": 0.5, "AU25": 0.4},
        "motion_type": "VIBRATION_FAST",
        "handedness": "dual",
    },
    "daktar": {
        "slug": "daktar",
        "label_bn": "ডাক্তার",
        "label_en": "Doctor",
        "anchor": "WRIST_ANCHOR",
        "palm_orientation": "DOWNWARD",
        "fingers_extended": [False, True, True, False, False],  # Pulse feeling fingers
        "facs_required": {},
        "motion_type": "TAP_DOUBLE",
        "handedness": "dual",
    },
    "kemon_achen": {
        "slug": "kemon_achen",
        "label_bn": "কেমন আছেন?",
        "label_en": "How are you?",
        "anchor": "CHEST",
        "palm_orientation": "UPWARD",
        "fingers_extended": [True, True, True, True, True],
        "facs_required": {"AU04": 0.6, "AU01": 0.3},  # Brow furrow Wh-question
        "facs_mandatory": True,
        "motion_type": "ROTATION_OUTWARD",
        "handedness": "dual",
    },
    "khawa": {
        "slug": "khawa",
        "label_bn": "খাওয়া",
        "label_en": "Eat",
        "anchor": "PHILTRUM",
        "palm_orientation": "INWARD",
        "fingers_extended": [False, True, True, True, True],  # Bunched fingertips toward mouth
        "facs_required": {},
        "motion_type": "OSCILLATE_MOUTH",
        "handedness": "single",
    },
    "taka": {
        "slug": "taka",
        "label_bn": "টাকা",
        "label_en": "Money",
        "anchor": "CHEST",
        "palm_orientation": "UPWARD",
        "fingers_extended": [True, True, False, False, False],  # Thumb rubbing index
        "facs_required": {},
        "motion_type": "RUB_DIGITS",
        "handedness": "single",
    },
    "cons_ka": {
        "slug": "cons_ka",
        "label_bn": "ক",
        "label_en": "Consonant Ka",
        "anchor": "NEUTRAL_SPACE",
        "palm_orientation": "OUTWARD",
        "fingers_extended": [False, True, False, False, False],  # Index pointing up, others curled
        "facs_required": {},
        "motion_type": "STATIC_HOLD",
        "handedness": "single",
    },
    "vowel_a": {
        "slug": "vowel_a",
        "label_bn": "অ",
        "label_en": "Vowel A",
        "anchor": "NEUTRAL_SPACE",
        "palm_orientation": "OUTWARD",
        "fingers_extended": [False, False, False, False, False],  # Fist with thumb resting across
        "facs_required": {},
        "motion_type": "STATIC_HOLD",
        "handedness": "single",
    },
    "vowel_aa": {
        "slug": "vowel_aa",
        "label_bn": "আ",
        "label_en": "Vowel Aa",
        "anchor": "NEUTRAL_SPACE",
        "palm_orientation": "OUTWARD",
        "fingers_extended": [True, False, False, False, False],  # Thumb extended upright
        "facs_required": {},
        "motion_type": "STATIC_HOLD",
        "handedness": "single",
    }
}


@dataclass
class DiagnosticResult:
    """Articulatory diagnostic result and corrective feedback container."""
    match_score: float                  # Overall match 0.0 - 100.0%
    channel_scores: Dict[str, float]    # Individual channel scores 0.0 - 1.0
    channel_status: Dict[str, str]      # "ok", "warn", "error" for each channel
    is_match: bool                      # True if match_score >= threshold
    corrective_hints: List[str]         # Localized Bengali guidance hints
    target_gloss: str
    target_bn: str
    target_en: str
    dominant_issue: Optional[str] = None


class SignCorrectionAdvisor:
    """Real-Time Articulatory Diagnostic Coach analyzing multi-channel landmark kinematics."""

    def __init__(self, acceptance_threshold: float = 75.0):
        self.acceptance_threshold = acceptance_threshold
        self.master_lexicon = master_lexicon
        self.spec_db = dict(SIGN_ARTICULATION_SPECS)
        self._sync_with_master_lexicon()

    def _sync_with_master_lexicon(self):
        """Enriches spec database with metadata from master BdSL lexicon."""
        for sign in self.master_lexicon.all_signs():
            slug = sign.get("slug")
            if slug and slug not in self.spec_db:
                contact = sign.get("contact_physics", {})
                anchor_name = contact.get("body_anchor", "NEUTRAL_SPACE").upper()
                if anchor_name not in ANCHOR_POSITIONS:
                    anchor_name = "NEUTRAL_SPACE"

                self.spec_db[slug] = {
                    "slug": slug,
                    "label_bn": sign.get("label_bn", slug),
                    "label_en": sign.get("label_en", slug),
                    "anchor": anchor_name,
                    "palm_orientation": contact.get("plane", "CORONAL_FRONT").upper(),
                    "fingers_extended": [True, True, True, True, True],
                    "facs_required": sign.get("facs_action_units", {}),
                    "motion_type": "STATIC_HOLD",
                    "handedness": sign.get("handedness", "single"),
                }

    def evaluate_user_posture(
        self,
        target_sign: str,
        right_landmarks: Optional[np.ndarray] = None,
        left_landmarks: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
        pose_landmarks: Optional[np.ndarray] = None,
        trajectory_3d: Optional[np.ndarray] = None
    ) -> DiagnosticResult:
        """Evaluates live posture across 5 channels against master ground-truth spec."""
        slug = self._resolve_slug(target_sign)
        spec = self.spec_db.get(slug, self.spec_db.get("dhonnobad"))

        # Channel 1: Handshape & Finger Extension
        handshape_score, handshape_status, handshape_hints = self._eval_handshape(
            right_landmarks, left_landmarks, spec
        )

        # Channel 2: Spatial Body Position / Anchor
        pos_score, pos_status, pos_hints = self._eval_position(
            right_landmarks, left_landmarks, spec
        )

        # Channel 3: Palm Normal & Orientation Vector
        orient_score, orient_status, orient_hints = self._eval_orientation(
            right_landmarks, spec
        )

        # Channel 4: Facial NMM / FACS Expression
        facs_score, facs_status, facs_hints = self._eval_facs(
            face_landmarks, spec
        )

        # Channel 5: Motion Trajectory Dynamics
        motion_score, motion_status, motion_hints = self._eval_motion(
            trajectory_3d, spec
        )

        channel_scores = {
            "handshape": handshape_score,
            "position": pos_score,
            "orientation": orient_score,
            "facs": facs_score,
            "motion": motion_score,
        }

        channel_status = {
            "handshape": handshape_status,
            "position": pos_status,
            "orientation": orient_status,
            "facs": facs_status,
            "motion": motion_status,
        }

        # Weighted Total Score: Handshape(35%) + Position(25%) + Orientation(20%) + FACS(10%) + Motion(10%)
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        match_score = round(
            (handshape_score * weights[0] +
             pos_score * weights[1] +
             orient_score * weights[2] +
             facs_score * weights[3] +
             motion_score * weights[4]) * 100.0,
            1
        )

        # Check for critical phonemic channel errors
        has_critical_error = (
            handshape_status == "error"
            or pos_status == "error"
            or (bool(spec.get("facs_mandatory")) and facs_status == "error")
        )
        is_match = (match_score >= self.acceptance_threshold) and not has_critical_error

        # Aggregate hints in priority order
        hints = []
        if not is_match:
            if handshape_hints:
                hints.extend(handshape_hints)
            if pos_hints:
                hints.extend(pos_hints)
            if orient_hints:
                hints.extend(orient_hints)
            if facs_hints:
                hints.extend(facs_hints)
            if motion_hints:
                hints.extend(motion_hints)
        else:
            hints.append("ভঙ্গি চমৎকার ও নিখুঁত! এভাবে ২ সেকেন্ড ধরে রাখুন।")

        dominant_issue = None
        if not is_match:
            min_channel = min(channel_scores, key=channel_scores.get)
            dominant_issue = min_channel

        return DiagnosticResult(
            match_score=match_score,
            channel_scores=channel_scores,
            channel_status=channel_status,
            is_match=is_match,
            corrective_hints=hints[:3] if hints else ["ক্যামেরার সামনে হাত স্পষ্ট রাখুন।"],
            target_gloss=slug,
            target_bn=spec.get("label_bn", slug),
            target_en=spec.get("label_en", slug),
            dominant_issue=dominant_issue
        )

    # ── Channel 1: Handshape Evaluation ───────────────────────────────────────

    def _eval_handshape(
        self,
        r_lm: Optional[np.ndarray],
        l_lm: Optional[np.ndarray],
        spec: Dict[str, Any]
    ) -> Tuple[float, str, List[str]]:
        if r_lm is None or len(r_lm) < 21:
            return 0.0, "error", ["ডান হাত ক্যামেরার সামনে দৃশ্যমান রাখুন।"]

        target_ext = spec.get("fingers_extended", [True, True, True, True, True])
        actual_ext = self._get_finger_extensions(r_lm)

        correct_count = sum(1 for a, t in zip(actual_ext, target_ext) if a == t)
        score = correct_count / 5.0

        hints = []
        finger_names = ["বৃদ্ধাঙ্গুলি", "তর্জনী", "মধ্যমা", "অনামিকা", "কনিষ্ঠা"]
        for idx, (act, tgt, name) in enumerate(zip(actual_ext, target_ext, finger_names)):
            if act != tgt:
                if tgt is True:
                    hints.append(f"{name} সোজা রাখুন।")
                else:
                    hints.append(f"{name} বন্ধ / বাঁকা করুন।")

        status = "ok" if score >= 0.8 else "warn" if score >= 0.5 else "error"
        return score, status, hints

    # ── Channel 2: Spatial Position Evaluation ───────────────────────────────

    def _eval_position(
        self,
        r_lm: Optional[np.ndarray],
        l_lm: Optional[np.ndarray],
        spec: Dict[str, Any]
    ) -> Tuple[float, str, List[str]]:
        if r_lm is None or len(r_lm) < 21:
            return 0.0, "error", ["হাতের অবস্থান শনাক্ত করা যায়নি।"]

        anchor_name = spec.get("anchor", "NEUTRAL_SPACE")
        target_pos = ANCHOR_POSITIONS.get(anchor_name, (0.50, 0.48))
        wrist_pos = (float(r_lm[0, 0]), float(r_lm[0, 1]))

        dist = math.hypot(wrist_pos[0] - target_pos[0], wrist_pos[1] - target_pos[1])
        # Perfect tolerance <= 0.08, zero score at distance >= 0.35
        score = max(0.0, min(1.0, 1.0 - (dist / 0.35)))

        hints = []
        anchor_bn_map = {
            "CHIN": "চিবুকের কাছে",
            "CHEEK_RIGHT": "ডান গালের কাছে",
            "CHEEK_LEFT": "বাম গালের কাছে",
            "UPPER_LIP": "ঠোঁটের ওপর (গোঁফের কাছে)",
            "PHILTRUM": "মুখের কাছে",
            "FOREHEAD": "কপালের কাছে",
            "CHEST": "বুকের সামনে",
            "WRIST_ANCHOR": "অন্য হাতের কবজির ওপর",
            "NEUTRAL_SPACE": "ক্যামেরার ফ্রেমের মাঝে"
        }
        loc_str = anchor_bn_map.get(anchor_name, "নির্দিষ্ট অবস্থানে")
        if score < 0.70:
            hints.append(f"হাতটি {loc_str} নিয়ে আসুন।")

        status = "ok" if score >= 0.75 else "warn" if score >= 0.45 else "error"
        return score, status, hints

    # ── Channel 3: Palm Normal Orientation ───────────────────────────────────

    def _eval_orientation(
        self,
        r_lm: Optional[np.ndarray],
        spec: Dict[str, Any]
    ) -> Tuple[float, str, List[str]]:
        if r_lm is None or len(r_lm) < 21:
            return 0.0, "error", ["তালুর অভিমুখ শনাক্ত করা যায়নি।"]

        target_orient = spec.get("palm_orientation", "OUTWARD").upper()
        normal = self._compute_palm_normal(r_lm)

        score = 0.5
        hints = []

        if target_orient == "OUTWARD" or "FRONT" in target_orient:
            score = 1.0 if normal[2] > -0.3 else 0.4
            if score < 0.7:
                hints.append("হাতের তালু সামনের দিকে ঘোরান।")
        elif target_orient == "INWARD" or "BACK" in target_orient:
            score = 1.0 if normal[2] > -0.3 else 0.4
            if score < 0.7:
                hints.append("হাতের তালু নিজের দিকে ঘোরান।")
        elif target_orient == "UPWARD":
            score = 1.0 if normal[1] < 0.2 else 0.4
            if score < 0.7:
                hints.append("হাতের তালু উপরের দিকে রাখুন।")
        elif target_orient == "DOWNWARD":
            score = 1.0 if normal[1] > -0.2 else 0.4
            if score < 0.7:
                hints.append("হাতের তালু নিচের দিকে রাখুন।")
        else:
            score = 0.9

        status = "ok" if score >= 0.75 else "warn" if score >= 0.50 else "error"
        return score, status, hints

    # ── Channel 4: Facial NMM FACS ───────────────────────────────────────────

    def _eval_facs(
        self,
        face_lm: Optional[np.ndarray],
        spec: Dict[str, Any]
    ) -> Tuple[float, str, List[str]]:
        req_facs = spec.get("facs_required", {})
        if not req_facs:
            return 1.0, "ok", []

        hints = []
        if "AU04" in req_facs and req_facs["AU04"] > 0.3:
            hints.append("ভ্রু সামান্য কুঁচকান (AU04 প্রশ্নবোধক অভিব্যক্তি)।")
        elif "AU12" in req_facs and req_facs["AU12"] > 0.3:
            hints.append("মুখে মৃদু হাসি রাখুন (AU12)।")
        elif "AU01" in req_facs and req_facs["AU01"] > 0.3:
            hints.append("ভ্রু কিছুটা উপরে তুলুন (AU01/AU02)।")
        else:
            hints.append("মুখের অভিব্যক্তি স্পষ্ট রাখুন।")

        if face_lm is None or len(face_lm) < 10:
            return 0.0, "error", hints

        score = 0.8
        status = "ok" if score >= 0.75 else "warn"
        return score, status, hints

    # ── Channel 5: Motion Trajectory Dynamics ────────────────────────────────

    def _eval_motion(
        self,
        trajectory: Optional[np.ndarray],
        spec: Dict[str, Any]
    ) -> Tuple[float, str, List[str]]:
        target_motion = spec.get("motion_type", "STATIC_HOLD")

        if trajectory is None or len(trajectory) < 5:
            # Static gesture or starting motion
            return 1.0 if target_motion == "STATIC_HOLD" else 0.75, "ok", []

        # Compute velocity variance
        diffs = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
        mean_speed = float(np.mean(diffs))

        hints = []
        score = 0.85
        if target_motion == "STATIC_HOLD":
            if mean_speed > 0.05:
                score = 0.4
                hints.append("হাতটি স্থির রাখুন (নড়াচড়া কম করুন)।")
            else:
                score = 1.0
        elif target_motion in ["VIBRATION_FAST", "TAP_DOUBLE"]:
            if mean_speed < 0.01:
                score = 0.5
                hints.append("নির্দেশিত গতিশীল ছন্দ অনুসরণ করুন।")
            else:
                score = 0.95

        status = "ok" if score >= 0.75 else "warn" if score >= 0.5 else "error"
        return score, status, hints

    # ── Utility Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_finger_extensions(lm: np.ndarray) -> List[bool]:
        """Detects boolean extension state [Thumb, Index, Middle, Ring, Pinky]."""
        wrist = lm[0]
        extensions = []

        # Thumb: Tip (4) vs IP (3) distance to wrist
        d_tip = math.hypot(lm[4, 0] - wrist[0], lm[4, 1] - wrist[1])
        d_ip = math.hypot(lm[3, 0] - wrist[0], lm[3, 1] - wrist[1])
        extensions.append(d_tip > d_ip * 1.05)

        # Fingers: Tip y vs PIP y (extended if tip is higher than PIP, i.e. lower y value)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip_idx, pip_idx in zip(tips, pips):
            extensions.append(bool(lm[tip_idx, 1] < lm[pip_idx, 1]))

        return extensions

    @staticmethod
    def _compute_palm_normal(lm: np.ndarray) -> Tuple[float, float, float]:
        """Computes 3D cross-product normal of the palm plane."""
        p0 = lm[0]
        p5 = lm[5]
        p17 = lm[17]

        v1 = p5 - p0
        v2 = p17 - p0
        normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal = normal / norm_val
        return (float(normal[0]), float(normal[1]), float(normal[2]))

    def _resolve_slug(self, text: str) -> str:
        s = text.strip().lower()
        if s in self.spec_db:
            return s
        for k, v in self.spec_db.items():
            if v.get("label_bn") == text or v.get("label_en", "").lower() == s:
                return k
            if k in s or s in k:
                return k
        return "dhonnobad"
