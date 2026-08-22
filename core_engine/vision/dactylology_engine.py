"""Tier 1: Full Dactylology & Trigger Synthesizer Engine for BdSL.

Implements:
1. Complete 50-Grapheme Inventory (11 Vowels, 39 Consonants, 10 Digits, Diacritics).
2. Dynamic Trigger Transformation Matrix (T0-T7) for ligatures and conjuncts.
3. Sliding-window cumulative confidence filtering with debounce thresholding.
"""

from collections import deque
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Complete BdSL 50-Grapheme + Digits + Diacritics Inventory
VOWELS = {
    "অ": {"slug": "vowel_a", "bn": "অ", "name_en": "Vowel A", "category": "Vowel", "type": "independent"},
    "আ": {"slug": "vowel_aa", "bn": "আ", "name_en": "Vowel Aa", "category": "Vowel", "type": "independent"},
    "ই": {"slug": "vowel_i", "bn": "ই", "name_en": "Vowel I", "category": "Vowel", "type": "independent"},
    "ঈ": {"slug": "vowel_ee", "bn": "ঈ", "name_en": "Vowel Ee", "category": "Vowel", "type": "independent"},
    "উ": {"slug": "vowel_u", "bn": "উ", "name_en": "Vowel U", "category": "Vowel", "type": "independent"},
    "ঊ": {"slug": "vowel_oo", "bn": "ঊ", "name_en": "Vowel Oo", "category": "Vowel", "type": "independent"},
    "ঋ": {"slug": "vowel_ri", "bn": "ঋ", "name_en": "Vowel Ri", "category": "Vowel", "type": "independent"},
    "এ": {"slug": "vowel_e", "bn": "এ", "name_en": "Vowel E", "category": "Vowel", "type": "independent"},
    "ঐ": {"slug": "vowel_oi", "bn": "ঐ", "name_en": "Vowel Oi", "category": "Vowel", "type": "independent"},
    "ও": {"slug": "vowel_o", "bn": "ও", "name_en": "Vowel O", "category": "Vowel", "type": "independent"},
    "ঔ": {"slug": "vowel_ou", "bn": "ঔ", "name_en": "Vowel Ou", "category": "Vowel", "type": "independent"},
}

CONSONANTS = {
    "ক": {"slug": "cons_ka", "bn": "ক", "name_en": "Ka", "category": "Consonant"},
    "খ": {"slug": "cons_kha", "bn": "খ", "name_en": "Kha", "category": "Consonant"},
    "গ": {"slug": "cons_ga", "bn": "গ", "name_en": "Ga", "category": "Consonant"},
    "ঘ": {"slug": "cons_gha", "bn": "ঘ", "name_en": "Gha", "category": "Consonant"},
    "ঙ": {"slug": "cons_uma", "bn": "ঙ", "name_en": "Uma", "category": "Consonant"},
    "চ": {"slug": "cons_ca", "bn": "চ", "name_en": "Cha", "category": "Consonant"},
    "ছ": {"slug": "cons_cha", "bn": "ছ", "name_en": "Chha", "category": "Consonant"},
    "জ": {"slug": "cons_ja", "bn": "জ", "name_en": "Ja", "category": "Consonant"},
    "ঝ": {"slug": "cons_jha", "bn": "ঝ", "name_en": "Jha", "category": "Consonant"},
    "ঞ": {"slug": "cons_nia", "bn": "ঞ", "name_en": "Nia", "category": "Consonant"},
    "ট": {"slug": "cons_tta", "bn": "ট", "name_en": "Tta", "category": "Consonant"},
    "ঠ": {"slug": "cons_ttha", "bn": "ঠ", "name_en": "Ttha", "category": "Consonant"},
    "ড": {"slug": "cons_dda", "bn": "ড", "name_en": "Dda", "category": "Consonant"},
    "ঢ": {"slug": "cons_ddha", "bn": "ঢ", "name_en": "Ddha", "category": "Consonant"},
    "ণ": {"slug": "cons_nna", "bn": "ণ", "name_en": "Nna", "category": "Consonant"},
    "ত": {"slug": "cons_ta", "bn": "ত", "name_en": "Ta", "category": "Consonant"},
    "থ": {"slug": "cons_tha", "bn": "থ", "name_en": "Tha", "category": "Consonant"},
    "দ": {"slug": "cons_da", "bn": "দ", "name_en": "Da", "category": "Consonant"},
    "ধ": {"slug": "cons_dha", "bn": "ধ", "name_en": "Dha", "category": "Consonant"},
    "ন": {"slug": "cons_na", "bn": "ন", "name_en": "Na", "category": "Consonant"},
    "প": {"slug": "cons_pa", "bn": "প", "name_en": "Pa", "category": "Consonant"},
    "ফ": {"slug": "cons_pha", "bn": "ফ", "name_en": "Pha", "category": "Consonant"},
    "ব": {"slug": "cons_ba", "bn": "ব", "name_en": "Ba", "category": "Consonant"},
    "ভ": {"slug": "cons_bha", "bn": "ভ", "name_en": "Bha", "category": "Consonant"},
    "ম": {"slug": "cons_ma", "bn": "ম", "name_en": "Ma", "category": "Consonant"},
    "য": {"slug": "cons_ya", "bn": "য", "name_en": "Antastha Ja", "category": "Consonant"},
    "র": {"slug": "cons_ra", "bn": "র", "name_en": "Ra", "category": "Consonant"},
    "ল": {"slug": "cons_la", "bn": "ল", "name_en": "La", "category": "Consonant"},
    "শ": {"slug": "cons_sha", "bn": "শ", "name_en": "Talobbo Sha", "category": "Consonant"},
    "ষ": {"slug": "cons_ssa", "bn": "ষ", "name_en": "Murdhonno Sha", "category": "Consonant"},
    "স": {"slug": "cons_sa", "bn": "স", "name_en": "Donto Sha", "category": "Consonant"},
    "হ": {"slug": "cons_ha", "bn": "হ", "name_en": "Ha", "category": "Consonant"},
    "ড়": {"slug": "cons_rra", "bn": "ড়", "name_en": "Dda-bindu Rra", "category": "Consonant"},
    "ঢ়": {"slug": "cons_rrha", "bn": "ঢ়", "name_en": "Ddha-bindu Rrha", "category": "Consonant"},
    "য়": {"slug": "cons_yya", "bn": "য়", "name_en": "Antastha Ya", "category": "Consonant"},
    "ৎ": {"slug": "diac_khanda_ta", "bn": "ৎ", "name_en": "Khanda Ta", "category": "Special"},
    "ং": {"slug": "diac_anusvara", "bn": "ং", "name_en": "Anusvara", "category": "Diacritic"},
    "ঃ": {"slug": "diac_visarga", "bn": "ঃ", "name_en": "Visarga", "category": "Diacritic"},
    "ঁ": {"slug": "diac_chandrabindu", "bn": "ঁ", "name_en": "Chandrabindu", "category": "Diacritic"},
}

DIGITS = {
    "০": {"slug": "num_0", "bn": "০", "name_en": "0 (Zero)", "category": "Digit"},
    "১": {"slug": "num_1", "bn": "১", "name_en": "1 (One)", "category": "Digit"},
    "২": {"slug": "num_2", "bn": "২", "name_en": "2 (Two)", "category": "Digit"},
    "৩": {"slug": "num_3", "bn": "৩", "name_en": "3 (Three)", "category": "Digit"},
    "৪": {"slug": "num_4", "bn": "৪", "name_en": "4 (Four)", "category": "Digit"},
    "৫": {"slug": "num_5", "bn": "৫", "name_en": "5 (Five)", "category": "Digit"},
    "৬": {"slug": "num_6", "bn": "৬", "name_en": "6 (Six)", "category": "Digit"},
    "৭": {"slug": "num_7", "bn": "৭", "name_en": "7 (Seven)", "category": "Digit"},
    "৮": {"slug": "num_8", "bn": "৮", "name_en": "8 (Eight)", "category": "Digit"},
    "৯": {"slug": "num_9", "bn": "৯", "name_en": "9 (Nine)", "category": "Digit"},
}

# Master Grapheme Index
MASTER_GRAPHEMES = {**VOWELS, **CONSONANTS, **DIGITS}

# Trigger Transformation Rules (T0-T7)
TRIGGER_MAP = {
    "T0": "IDENTITY",             # Base character unchanged
    "T1": "KAR_AA",               # Attach A-kar (া) or synthesize আ
    "T2": "KAR_I",                # Attach I-kar (ি) or synthesize ই
    "T3": "KAR_U",                # Attach U-kar (ু) or synthesize উ
    "T4": "CONJUNCT_KSHA",        # Synthesize ক + ষ -> ক্ষ
    "T5": "CONJUNCT_GYA",         # Synthesize জ + ঞ -> জ্ঞ
    "T6": "DIACRITIC_ATTACH",     # Attach ং, ঃ, ঁ
    "T7": "VIRAMA_HALANT",        # Attach halant (্)
}

CONJUNCT_DICTIONARY = {
    ("ক", "ষ"): "ক্ষ",
    ("জ", "ঞ"): "জ্ঞ",
    ("ঙ", "ক"): "ঙ্ক",
    ("ঙ", "গ"): "ঙ্গ",
    ("ত", "ত"): "ত্ত",
    ("ত", "র"): "ত্র",
    ("শ", "র"): "শ্র",
    ("স", "থ"): "স্থ",
    ("দ", "ধ"): "দ্ধ",
    ("ব", "দ"): "ব্দ",
}


class DactylologyEngine:
    """Tier 1: High-precision Fingerspelling & Trigger Synthesizer Engine."""

    def __init__(
        self,
        cumulative_confidence_delta: float = 0.85,
        window_size: int = 15,
        debounce_latency_s: float = 1.32
    ):
        self.delta = cumulative_confidence_delta
        self.window_size = window_size
        self.debounce_latency_s = debounce_latency_s

        # Sliding window buffer: stores (char, confidence, timestamp)
        self.sample_buffer: deque = deque(maxlen=window_size)
        self.last_emitted_char: Optional[str] = None
        self.last_emitted_time: float = 0.0
        self.active_trigger: Optional[str] = None

    def reset(self):
        """Clears buffers and reset state."""
        self.sample_buffer.clear()
        self.last_emitted_char = None
        self.last_emitted_time = 0.0
        self.active_trigger = None

    def get_grapheme_meta(self, token: str) -> Optional[Dict[str, Any]]:
        """Resolves metadata by Bengali character or slug."""
        if token in MASTER_GRAPHEMES:
            return MASTER_GRAPHEMES[token]
        for item in MASTER_GRAPHEMES.values():
            if item["slug"] == token or item["bn"] == token:
                return item
        return None

    def apply_trigger_transform(self, base_char: str, trigger_id: str) -> str:
        """Applies dynamic trigger transformation matrix (T0-T7) to a base character."""
        t_id = trigger_id.upper()
        if t_id == "T0":
            return base_char
        elif t_id == "T1":
            return base_char + "া" if base_char in CONSONANTS else "আ"
        elif t_id == "T2":
            return base_char + "ি" if base_char in CONSONANTS else "ই"
        elif t_id == "T3":
            return base_char + "ু" if base_char in CONSONANTS else "উ"
        elif t_id == "T4":
            if base_char == "ক":
                return "ক্ষ"
            return base_char + "্" + "ষ"
        elif t_id == "T5":
            if base_char == "জ":
                return "জ্ঞ"
            return base_char + "্" + "ঞ"
        elif t_id == "T6":
            return base_char + "ঁ"
        elif t_id == "T7":
            return base_char + "্"
        return base_char

    def synthesize_conjunct(self, char1: str, char2: str) -> str:
        """Synthesizes compound conjunct graphemes from dual base characters."""
        pair = (char1, char2)
        if pair in CONJUNCT_DICTIONARY:
            return CONJUNCT_DICTIONARY[pair]
        return f"{char1}্{char2}"

    def process_character_prediction(
        self,
        raw_class: str,
        confidence: float,
        timestamp: Optional[float] = None,
        trigger_id: Optional[str] = None
    ) -> Optional[str]:
        """Ingests a raw classifier prediction, applies temporal cumulative confidence

        filtering, evaluates trigger transformations, and debounces output.

        Returns:
            Resolved grapheme string if confidence is sustained and stable, else None.
        """
        now = timestamp if timestamp is not None else time.time()
        
        # Sanitize inputs
        if raw_class is None or confidence is None or math.isnan(confidence) or math.isinf(confidence):
            return None
        
        conf = max(0.0, min(1.0, float(confidence)))
        
        # Resolve character
        char_meta = self.get_grapheme_meta(raw_class)
        resolved_char = char_meta["bn"] if char_meta else raw_class

        # Apply trigger if specified
        if trigger_id and trigger_id in TRIGGER_MAP:
            resolved_char = self.apply_trigger_transform(resolved_char, trigger_id)

        # Append to temporal sliding window
        self.sample_buffer.append((resolved_char, conf, now))

        if len(self.sample_buffer) < 3:
            return None

        # Calculate cumulative sliding-window confidence for candidate
        matching_samples = [s for s in self.sample_buffer if s[0] == resolved_char]
        if not matching_samples:
            return None

        # Cumulative Confidence: \sum_{t=1}^T \frac{1}{N} \sum c_i(t)
        total_conf = sum(s[1] for s in matching_samples)
        avg_conf = total_conf / len(matching_samples)
        window_agreement = len(matching_samples) / len(self.sample_buffer)
        cumulative_score = avg_conf * window_agreement

        if cumulative_score >= self.delta:
            # Debounce check
            time_since_last = now - self.last_emitted_time
            if resolved_char != self.last_emitted_char or time_since_last >= self.debounce_latency_s:
                self.last_emitted_char = resolved_char
                self.last_emitted_time = now
                return resolved_char

        return None
