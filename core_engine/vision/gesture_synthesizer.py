"""Speech-to-BdSL Visual Avatar & Gesture Synthesizer Engine.

Translates spoken or typed Bengali text into sequenced BdSL gesture animations,
resolving lexical sign cards and falling back to automated finger-spelling (বর্ণানুক্রমিক বানান).
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BdSLGestureSynthesizer:
    """Translates Bengali text into sequenced BdSL gesture cards and animation timing frames."""

    def __init__(self, labels_path: str = "dataset/labels.json", cards_dir: str = "dataset/visual_cards"):
        self.labels_path = labels_path
        self.cards_dir = cards_dir
        self.labels_data = {}
        self.word_to_sign = {}

        # Alphabet to SVG card mapping for fingerspelling
        self.alphabet_to_card = {
            "অ": "a.svg", "আ": "aa.svg", "ই": "i.svg", "ঈ": "i.svg", "উ": "u.svg", "ঊ": "u.svg",
            "ঋ": "ri.svg", "এ": "e.svg", "ঐ": "e.svg", "ও": "o.svg", "ঔ": "o.svg",
            "ক": "ka.svg", "খ": "kha.svg", "গ": "ga.svg", "ঘ": "gha.svg", "ঙ": "nga.svg",
            "চ": "cha.svg", "ছ": "chha.svg", "জ": "ja.svg", "ঝ": "jha.svg", "ঞ": "nya.svg",
            "ট": "ta_.svg", "ঠ": "tha_.svg", "ড": "da_.svg", "ঢ": "dha_.svg", "ণ": "na_.svg",
            "ত": "ta.svg", "থ": "tha.svg", "দ": "da.svg", "ধ": "dha.svg", "ন": "na.svg",
            "প": "pa.svg", "ফ": "pha.svg", "ব": "ba.svg", "ভ": "bha.svg", "ম": "ma.svg",
            "য": "ya.svg", "র": "ra.svg", "ল": "la.svg", "শ": "sha.svg", "ষ": "sa.svg", "স": "sa.svg", "হ": "ha.svg",
            "ড়": "ra.svg", "ঢ়": "ra.svg", "য়": "ya.svg", "ৎ": "ta.svg",
            "০": "a.svg", "১": "ek.svg", "২": "dui.svg", "৩": "tin.svg"
        }

        # Common Vocabulary Word mappings
        self.direct_word_map = {
            "ধন্যবাদ": {"slug": "dhonnobad", "svg": "dhonnobad.svg", "en": "Thank you", "type": "Dynamic | Single"},
            "ডাক্তার": {"slug": "daktar", "svg": "daktar.svg", "en": "Doctor", "type": "Dynamic | Dual"},
            "হাসপাতাল": {"slug": "hospital", "svg": "hospital.svg", "en": "Hospital", "type": "Dynamic | Dual"},
            "সাহায্য": {"slug": "shahajjo", "svg": "shahajjo.svg", "en": "Help", "type": "Dynamic | Dual"},
            "পানি": {"slug": "pani", "svg": "pani.svg", "en": "Water", "type": "Dynamic | Single"},
            "খাবার": {"slug": "khabar", "svg": "khabar.svg", "en": "Food", "type": "Dynamic | Single"},
            "ওষুধ": {"slug": "osud", "svg": "osud.svg", "en": "Medicine", "type": "Dynamic | Dual"},
            "ভালো": {"slug": "bhalo", "svg": "bhalo.svg", "en": "Good", "type": "Static | Single"},
            "খারাপ": {"slug": "kharap", "svg": "kharap.svg", "en": "Bad", "type": "Dynamic | Single"},
            "কেমন আছেন": {"slug": "kemon_achen", "svg": "kemon_achen.svg", "en": "How are you?", "type": "Dynamic | Dual"},
            "স্বাগতম": {"slug": "shagotom", "svg": "shagotom.svg", "en": "Welcome", "type": "Dynamic | Dual"},
            "আসসালামু আলাইকুম": {"slug": "assalamu_alaikum", "svg": "assalamu_alaikum.svg", "en": "Hello", "type": "Dynamic | Single"},
            "নাম": {"slug": "naam", "svg": "naam.svg", "en": "Name", "type": "Dynamic | Dual"},
            "আমি": {"slug": "ami", "svg": "ami.svg", "en": "I / Me", "type": "Static | Single"},
            "আমার": {"slug": "ami", "svg": "ami.svg", "en": "My", "type": "Static | Single"},
            "তুমি": {"slug": "tumi", "svg": "tumi.svg", "en": "You", "type": "Static | Single"},
            "তোমার": {"slug": "tumi", "svg": "tumi.svg", "en": "Your", "type": "Static | Single"},
            "আপনি": {"slug": "apni", "svg": "apni.svg", "en": "You (Polite)", "type": "Static | Single"},
            "আপনার": {"slug": "apni", "svg": "apni.svg", "en": "Your (Polite)", "type": "Static | Single"},
            "বুঝেছি": {"slug": "bujhechi", "svg": "bujhechi.svg", "en": "Understood", "type": "Dynamic | Single"},
            "বুঝিনি": {"slug": "bujhini", "svg": "bujhini.svg", "en": "Not Understood", "type": "Dynamic | Single"},
            "ঠিক আছে": {"slug": "thik_ache", "svg": "thik_ache.svg", "en": "All Right", "type": "Static | Single"},
            "এক": {"slug": "ek", "svg": "ek.svg", "en": "One", "type": "Static | Single"},
            "দুই": {"slug": "dui", "svg": "dui.svg", "en": "Two", "type": "Static | Single"},
            "তিন": {"slug": "tin", "svg": "tin.svg", "en": "Three", "type": "Static | Single"}
        }

        self._load_labels()

    def _load_labels(self):
        """Loads signs metadata from labels.json."""
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels_data = json.load(f)
                for sign in self.labels_data.get("signs", []):
                    bn = sign.get("label_bn", "").strip()
                    if bn:
                        self.word_to_sign[bn] = sign
            except Exception as e:
                logger.warning("Could not parse labels.json: %s", e)

    def _clean_token(self, token: str) -> str:
        """Strips punctuation and whitespace from token."""
        return token.strip("।,!?;:\'\"()[]{} ")

    def synthesize_text_to_gestures(self, text: str, speed: float = 1.0) -> List[Dict[str, Any]]:
        """Converts raw Bengali text into a sequenced timeline of visual BdSL gesture frames.

        Args:
            text: Input sentence or speech transcript.
            speed: Playback speed multiplier (e.g. 1.0 = normal 800ms per gesture).

        Returns:
            List of gesture frame dictionaries ready for avatar visualization.
        """
        if not text or not text.strip():
            return []

        base_duration_word = int(900 / max(0.2, speed))
        base_duration_spell = int(600 / max(0.2, speed))

        raw_words = text.split()
        gestures = []

        # Multi-word phrase lookahead (e.g. "কেমন আছেন", "ঠিক আছে", "আসসালামু আলাইকুম")
        i = 0
        while i < len(raw_words):
            # Check 2-word phrase
            if i < len(raw_words) - 1:
                two_word = f"{self._clean_token(raw_words[i])} {self._clean_token(raw_words[i+1])}"
                if two_word in self.direct_word_map:
                    info = self.direct_word_map[two_word]
                    card_file = os.path.join(self.cards_dir, info["svg"])
                    gestures.append({
                        "token": two_word,
                        "type": "word",
                        "sign_slug": info["slug"],
                        "label_bn": two_word,
                        "label_en": info["en"],
                        "motion_type": info["type"],
                        "card_path": card_file if os.path.exists(card_file) else None,
                        "duration_ms": base_duration_word,
                        "description": f"Standard BdSL sign for '{two_word}' ({info['en']})"
                    })
                    i += 2
                    continue

            # Single word check
            clean_w = self._clean_token(raw_words[i])
            if not clean_w:
                i += 1
                continue

            # Direct dictionary word match
            if clean_w in self.direct_word_map:
                info = self.direct_word_map[clean_w]
                card_file = os.path.join(self.cards_dir, info["svg"])
                gestures.append({
                    "token": clean_w,
                    "type": "word",
                    "sign_slug": info["slug"],
                    "label_bn": clean_w,
                    "label_en": info["en"],
                    "motion_type": info["type"],
                    "card_path": card_file if os.path.exists(card_file) else None,
                    "duration_ms": base_duration_word,
                    "description": f"Standard BdSL sign for '{clean_w}' ({info['en']})"
                })
            elif clean_w in self.word_to_sign:
                sign = self.word_to_sign[clean_w]
                slug = sign.get("slug", "unknown")
                card_file = os.path.join(self.cards_dir, f"{slug}.svg")
                gestures.append({
                    "token": clean_w,
                    "type": "word",
                    "sign_slug": slug,
                    "label_bn": clean_w,
                    "label_en": sign.get("label_en", clean_w),
                    "motion_type": f"{sign.get('motion_type', 'dynamic').title()} | {sign.get('handedness', 'single').title()}",
                    "card_path": card_file if os.path.exists(card_file) else None,
                    "duration_ms": base_duration_word,
                    "description": sign.get("description", f"Sign for {clean_w}")
                })
            else:
                # Fallback: Automatic Finger-Spelling (বর্ণানুক্রমিক বানান)
                for char in clean_w:
                    if char in self.alphabet_to_card:
                        svg_name = self.alphabet_to_card[char]
                        card_file = os.path.join(self.cards_dir, svg_name)
                        gestures.append({
                            "token": char,
                            "type": "fingerspell",
                            "sign_slug": f"spell_{char}",
                            "label_bn": f"{clean_w} [{char}]",
                            "label_en": f"Spell: {char}",
                            "motion_type": "Static | Single Hand",
                            "card_path": card_file if os.path.exists(card_file) else None,
                            "duration_ms": base_duration_spell,
                            "description": f"Finger-spelling letter '{char}'"
                        })

            i += 1

        return gestures
