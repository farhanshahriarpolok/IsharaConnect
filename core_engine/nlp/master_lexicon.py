"""Tier 2: Master BdSL Lexicon Query & Schema Engine.

Provides unified runtime querying, category indexing, and kinematic profile retrieval
for standardized Bangladesh Sign Language (BdSL) signs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEXICON_JSON_PATH = Path(__file__).resolve().parents[2] / "dataset" / "lexicon" / "master_bdsl_lexicon.json"


class MasterBdSLLexicon:
    """In-memory indexing and query interface for the Master BdSL Lexical Database."""

    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = json_path or LEXICON_JSON_PATH
        self.signs_by_slug: Dict[str, Dict[str, Any]] = {}
        self.signs_by_bn: Dict[str, Dict[str, Any]] = {}
        self.signs_by_category: Dict[str, List[Dict[str, Any]]] = {}
        self._load_lexicon()

    def _load_lexicon(self):
        """Loads and indexes the master lexicon from JSON."""
        if not self.json_path.exists():
            logger.warning("Lexicon file %s not found. Initializing empty fallback.", self.json_path)
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.dactylology = data.get("tier_1_complete_dactylology", {})
            self.continuous_corpus = data.get("tier_3_4_continuous_corpus", [])

            for sign in data.get("signs", []):
                slug = sign.get("slug", "")
                bn = sign.get("label_bn", "").strip()
                cat = sign.get("category", "General")

                if slug:
                    self.signs_by_slug[slug] = sign
                if bn:
                    self.signs_by_bn[bn] = sign

                if cat not in self.signs_by_category:
                    self.signs_by_category[cat] = []
                self.signs_by_category[cat].append(sign)

            logger.info("Loaded %d master BdSL signs from %s", len(self.signs_by_slug), self.json_path)
        except Exception as e:
            logger.error("Failed to load master BdSL lexicon: %s", e)

    def get_vowels_dactylology(self) -> Dict[str, Any]:
        """Returns Tier-1 complete vowel dactylology inventory with kar triggers."""
        return getattr(self, "dactylology", {}).get("VOWELS", {})

    def get_continuous_corpus(self) -> List[Dict[str, Any]]:
        """Returns Tier-3/4 continuous syntactic BdSL corpus sentences."""
        return getattr(self, "continuous_corpus", [])

    def get_sign_by_gloss(self, gloss: str) -> Optional[Dict[str, Any]]:
        """Resolves sign metadata by Bengali gloss, slug, or English label."""
        if not gloss:
            return None
        clean = gloss.strip()
        
        # 1. Exact Bengali match
        if clean in self.signs_by_bn:
            return self.signs_by_bn[clean]
        
        # 2. Exact slug match
        clean_slug = clean.lower().replace(" ", "_")
        if clean_slug in self.signs_by_slug:
            return self.signs_by_slug[clean_slug]

        # 3. Fuzzy search in English names or slugs
        for s in self.signs_by_slug.values():
            if s.get("label_en", "").lower() == clean.lower():
                return s
            if clean in s.get("label_bn", "") or s.get("label_bn", "") in clean:
                return s

        return None

    def get_signs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Returns all signs in a given domain category."""
        return self.signs_by_category.get(category, [])

    def get_kinematic_profile(self, gloss: str) -> Optional[Dict[str, Any]]:
        """Returns the 3D Bézier anchors, contact physics, and FACS vectors for a sign."""
        sign = self.get_sign_by_gloss(gloss)
        if not sign:
            return None

        return {
            "slug": sign.get("slug"),
            "label_bn": sign.get("label_bn"),
            "handshape": sign.get("handshape"),
            "stokoe_notation": sign.get("stokoe_notation"),
            "bezier_anchors_3d": sign.get("bezier_anchors_3d", {}),
            "facs_action_units": sign.get("facs_action_units", {}),
            "contact_physics": sign.get("contact_physics", {}),
            "timing_ms": sign.get("timing_ms", {})
        }

    def all_signs(self) -> List[Dict[str, Any]]:
        """Returns list of all signs in the database."""
        return list(self.signs_by_slug.values())

    def get_all_signs(self) -> List[Dict[str, Any]]:
        """Alias for all_signs()."""
        return self.all_signs()

    def search_signs(self, query: str) -> List[Dict[str, Any]]:
        """Searches signs across Bengali, English, and category fields."""
        q = query.strip().lower()
        results = []
        for s in self.signs_by_slug.values():
            if (
                q in s.get("label_bn", "").lower()
                or q in s.get("label_en", "").lower()
                or q in s.get("slug", "").lower()
                or q in s.get("category", "").lower()
            ):
                results.append(s)
        return results


    def get_articulatory_spec(self, gloss: str) -> Dict[str, Any]:
        """Returns hyper-granular articulatory specification and step-by-step instructions for a sign."""
        sign = self.get_sign_by_gloss(gloss) or {}
        slug = sign.get("slug", gloss)

        # Built-in Special Dactylology & Core Sign Overrides
        dactylology_specs = {
            "cons_ka": {"label_bn": "ক", "label_en": "Consonant Ka", "handshape": "INDEX_EXTENDED", "anchor": "NEUTRAL_SPACE", "articulator_type": "INDEX_TIP", "motion_type": "STATIC_HOLD", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}},
            "vowel_a": {"label_bn": "অ", "label_en": "Vowel A", "handshape": "FIST", "anchor": "NEUTRAL_SPACE", "articulator_type": "PALM_CENTER", "motion_type": "STATIC_HOLD", "finger_states": {"thumb": "CURL_FULL", "index": "CURL_FULL", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}},
            "vowel_aa": {"label_bn": "আ", "label_en": "Vowel Aa", "handshape": "THUMB_UP", "anchor": "NEUTRAL_SPACE", "articulator_type": "THUMB_TIP", "motion_type": "STATIC_HOLD", "finger_states": {"thumb": "EXTENDED", "index": "CURL_FULL", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}},
            "dhonnobad": {"label_bn": "ধন্যবাদ", "label_en": "Thank you", "handshape": "OPEN_PALM", "anchor": "CHIN", "articulator_type": "FINGERTIPS_FLAT", "motion_type": "ARC_FORWARD", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"}, "palm_facing": "FACING_CAMERA"},
            "baba": {"label_bn": "বাবা", "label_en": "Father", "handshape": "INDEX_HOOK", "anchor": "UPPER_LIP", "articulator_type": "THUMB_INDEX_PINCH", "motion_type": "PULL_RIGHT", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_CAMERA"},
            "ma": {"label_bn": "মা", "label_en": "Mother", "handshape": "OPEN_PALM", "anchor": "CHEEK_RIGHT", "articulator_type": "INDEX_TIP", "motion_type": "TAP_TWICE", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"}, "palm_facing": "FACING_CAMERA"},
            "salam": {"label_bn": "সালাম", "label_en": "Salam / Greetings", "handshape": "OPEN_PALM", "anchor": "FOREHEAD", "articulator_type": "FINGERTIPS_FLAT", "motion_type": "ARC_FORWARD", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"}, "palm_facing": "FACING_CAMERA"},
            "daktar": {"label_bn": "ডাক্তার", "label_en": "Doctor", "handshape": "V_SHAPE", "anchor": "LEFT_WRIST", "articulator_type": "DUAL_INDEX_MIDDLE", "motion_type": "TAP_TWICE", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "EXTENDED", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_DOWN"},
            "chacha": {"label_bn": "চাচা", "label_en": "Uncle", "handshape": "INDEX_HOOK", "anchor": "CHIN", "articulator_type": "INDEX_HOOK", "motion_type": "TAP_ONCE", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_CAMERA"},
            "debor": {"label_bn": "দেবর", "label_en": "Brother-in-law", "handshape": "POINT", "anchor": "NOSE", "articulator_type": "INDEX_TIP", "motion_type": "PULL_DOWN", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_USER"},
            "dulabhai": {"label_bn": "দুলাভাই", "label_en": "Elder Brother-in-law", "handshape": "V_SHAPE", "anchor": "NOSE", "articulator_type": "V_SHAPE_MIDDLE_INDEX", "motion_type": "PULL_DOWN", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "EXTENDED", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_USER"},
            "bhai": {"label_bn": "ভাই", "label_en": "Brother", "handshape": "POINT", "anchor": "CHEST_MID", "articulator_type": "PARALLEL_INDEX_TOUCH", "motion_type": "STATIC_HOLD", "handedness": "dual", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_CAMERA"},
            "bon": {"label_bn": "বোন", "label_en": "Sister", "handshape": "POINT", "anchor": "NOSE", "articulator_type": "INDEX_TIP", "motion_type": "TAP_ONCE", "finger_states": {"thumb": "CURL_FULL", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_CAMERA"},
            "bhumikompo": {"label_bn": "ভূমিকম্প", "label_en": "Earthquake", "handshape": "OPEN_PALM", "anchor": "NEUTRAL_SPACE", "articulator_type": "FINGERTIPS_FLAT", "motion_type": "HIGH_FREQ_VIBRATION", "handedness": "dual", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"}, "palm_facing": "FACING_DOWN"},
            "sahajjo": {"label_bn": "সাহায্য", "label_en": "Help", "handshape": "OPEN_PALM", "anchor": "CHEST_MID", "articulator_type": "PALM_CLASP", "motion_type": "BOOST_UPWARD", "handedness": "dual", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"}, "palm_facing": "FACING_UP"},
            "cha": {"label_bn": "চা", "label_en": "Tea", "handshape": "HS_PINCH_CUP", "anchor": "CHEST_MID", "articulator_type": "THUMB_INDEX_PINCH", "motion_type": "TAP_TWICE", "handedness": "dual", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_CAMERA"},
            "coffee": {"label_bn": "কফি", "label_en": "Coffee", "handshape": "HS_FIST_ON_FIST", "anchor": "CHEST_MID", "articulator_type": "PALM_CLASP", "motion_type": "CIRCULAR_ORBIT", "handedness": "dual", "finger_states": {"thumb": "CURL_FULL", "index": "CURL_FULL", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_DOWN"},
            "dudh": {"label_bn": "দুধ", "label_en": "Milk", "handshape": "HS_SQUEEZE_FIST", "anchor": "CHEST_MID", "articulator_type": "FINGERTIPS_FLAT", "motion_type": "PULL_DOWN", "handedness": "dual", "finger_states": {"thumb": "CURL_FULL", "index": "CURL_FULL", "middle": "CURL_FULL", "ring": "CURL_FULL", "pinky": "CURL_FULL"}, "palm_facing": "FACING_USER"},
            "kemon_achen": {"label_bn": "কেমন আছেন?", "label_en": "How are you?", "handshape": "OPEN_PALM", "anchor": "CHEST_MID", "articulator_type": "FINGERTIPS_FLAT", "motion_type": "STATIC_HOLD", "finger_states": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"}, "facs_action_units": {"AU04": 0.6, "AU01": 0.3}, "facs_mandatory": True, "handedness": "dual"}
        }
        override = dactylology_specs.get(slug, dactylology_specs.get(gloss, {}))

        handedness = str(override.get("handedness", sign.get("handedness", "single"))).lower()
        is_dual = handedness in ["dual", "both", "2"]
        req_hand = "DUAL_HAND" if is_dual else "RIGHT_ONLY"

        contact = sign.get("contact_physics", {})
        anchor = override.get("anchor", contact.get("body_anchor", "NEUTRAL_SPACE")).upper()
        if anchor in ["LIP_UPPER", "PHILTRUM"]:
            anchor = "UPPER_LIP"

        plane = str(contact.get("plane", "CORONAL")).upper()
        palm_facing = override.get("palm_facing", "FACING_CAMERA" if "FRONT" in plane or "CORONAL" in plane else ("FACING_USER" if "SAGITTAL" in plane else "FACING_DOWN"))

        handshape = str(override.get("handshape", sign.get("handshape", "OPEN_PALM"))).upper()
        default_finger_states = {
            "thumb": "EXTENDED" if "OPEN" in handshape or "THUMB" in handshape else "CURL_FULL",
            "index": "EXTENDED" if "OPEN" in handshape or "INDEX" in handshape or "POINT" in handshape else "CURL_FULL",
            "middle": "EXTENDED" if "OPEN" in handshape or "V_" in handshape else "CURL_FULL",
            "ring": "EXTENDED" if "OPEN" in handshape else "CURL_FULL",
            "pinky": "EXTENDED" if "OPEN" in handshape or "I_" in handshape else "CURL_FULL",
        }
        finger_states = override.get("finger_states", sign.get("finger_states", default_finger_states))

        touch_type = contact.get("touch_type", "STATIC_HOLD").upper()
        if "DOUBLE_TAP" in touch_type or "TAP_TWICE" in touch_type:
            default_motion = "TAP_TWICE"
        elif "TAP" in touch_type:
            default_motion = "TAP_ONCE"
        elif "SWIPE" in touch_type or "STROKE" in touch_type:
            default_motion = "DOWNWARD_STROKE"
        elif "VIBRAT" in touch_type:
            default_motion = "HIGH_FREQ_VIBRATION"
        else:
            default_motion = "STATIC_HOLD"
        motion_type = override.get("motion_type", sign.get("motion_type", default_motion))

        # Articulator resolution
        default_articulator = "INDEX_TIP" if "INDEX" in handshape or "POINT" in handshape else ("FINGERTIPS_FLAT" if "OPEN" in handshape else "AUTO")
        articulator_type = override.get("articulator_type", sign.get("articulator_type", default_articulator))

        label_bn = override.get("label_bn", sign.get("label_bn", slug))
        label_en = override.get("label_en", sign.get("label_en", slug))
        step_1 = "উভয় হাত সমান উচ্চতায় প্রস্তুত করুন" if is_dual else "ডান হাত ব্যবহার করুন"
        anchor_bn = {
            "CHIN": "চিবুকের কাছে",
            "UPPER_LIP": "ঠোঁটের ওপর (গোঁফের কাছে)",
            "CHEEK": "ডান গালের কাছে",
            "CHEEK_RIGHT": "ডান গালের কাছে",
            "FOREHEAD": "কপালের সামনে",
            "NOSE": "নাকের কাছে",
            "LEFT_WRIST": "বাম হাতের কবজির ওপর",
            "CHEST": "বুকের সামনে",
            "CHEST_MID": "বুকের মাঝে",
            "NEUTRAL_SPACE": "ক্যামেরা ফ্রেমের মাঝে"
        }.get(anchor, "নির্দিষ্ট স্থানে")
        step_2 = f"হাতটি {anchor_bn} নিয়ে আসুন"
        step_3 = f"হাতের আঙুলগুলো '{label_bn}' এর নির্দেশিত আকৃতিতে প্রস্তুত রাখুন"
        step_4 = "তালু ক্যামেরার দিকে রেখে স্থির রাখুন" if palm_facing == "FACING_CAMERA" else "তালু নির্দেশিত দিকে রেখে নির্দিষ্ট গতিশীল স্পর্শ সম্পন্ন করুন"

        facs_units = override.get("facs_action_units", sign.get("facs_action_units", {}))
        facs_mandatory = override.get("facs_mandatory", sign.get("facs_mandatory", False))

        return {
            "slug": slug,
            "label_bn": label_bn,
            "label_en": label_en,
            "required_hand": override.get("required_hand", sign.get("required_hand", req_hand)),
            "target_body_anchor": override.get("target_body_anchor", sign.get("target_body_anchor", anchor)),
            "articulator_type": articulator_type,
            "target_anchor_tolerance_cm": sign.get("target_anchor_tolerance_cm", 6.0),
            "palm_facing": palm_facing,
            "finger_states": finger_states,
            "facs_action_units": facs_units,
            "facs_mandatory": facs_mandatory,
            "motion_type": motion_type,
            "instructions_bn": sign.get("instructions_bn", {
                "step_1_hand": step_1,
                "step_2_location": step_2,
                "step_3_fingers": step_3,
                "step_4_palm_action": step_4
            })
        }


# Module-level singleton
master_lexicon = MasterBdSLLexicon()

