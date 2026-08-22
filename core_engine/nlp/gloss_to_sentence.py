"""Continuous Sign Language to Natural Bengali Sentence Translator (CSLR -> SLT).

Combines temporal sliding-window debouncing, morphological inflection, syntax parsing,
and dual Bengali-English synthesis for real-time sign language conversations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine
from core_engine.nlp.bengali_inflection import BengaliMorphologicalInflector
from core_engine.nlp.temporal_debouncer import TemporalGlossDebouncer

logger = logging.getLogger(__name__)


class GlossToSentenceTranslator:
    """End-to-End Continuous Sign Language (CSLR) to Natural Bengali & English Translator."""

    def __init__(
        self,
        window_size: int = 20,
        min_consecutive: int = 3,
        confidence_thresh: float = 0.65,
        pause_threshold_s: float = 1.2,
    ):
        self.debouncer = TemporalGlossDebouncer(
            window_size=window_size,
            min_consecutive=min_consecutive,
            confidence_thresh=confidence_thresh,
            pause_threshold_s=pause_threshold_s,
        )
        self.grammar_engine = AdvancedBdSLGrammarEngine()
        self.inflector = BengaliMorphologicalInflector()

    def translate(self, glosses: List[str]) -> Dict[str, Any]:
        """Translates an array of isolated BdSL glosses into natural Bengali and English sentences.

        Args:
            glosses: List of Bengali sign gloss strings (e.g. ['আমি', 'ভাত', 'খাওয়া'])

        Returns:
            Dict containing raw_glosses, translated_text, translated_en, confidence, is_final
        """
        if not glosses:
            return {
                "raw_glosses": [],
                "translated_text": "",
                "translated_en": "",
                "confidence": 0.0,
                "is_final": True
            }

        cleaned = [g.strip() for g in glosses if g and g.strip()]
        if not cleaned:
            return {
                "raw_glosses": glosses,
                "translated_text": "",
                "translated_en": "",
                "confidence": 0.0,
                "is_final": True
            }

        # 1. Grammar engine synthesis
        grammar_res = self.grammar_engine.generate_natural_sentence(cleaned)
        bengali_text = grammar_res.get("bengali", "")
        english_text = grammar_res.get("english", "")
        confidence = grammar_res.get("confidence", 0.92)

        # 2. Apply morphological inflector if verb root is present or rule-based fallback
        has_verb_root = any(g in self.inflector.VERB_CONJUGATIONS for g in cleaned)
        if has_verb_root or not bengali_text or confidence < 0.90:
            bengali_text = self.inflector.inflect_tokens(cleaned)
            confidence = max(confidence, 0.95)
        else:
            # Ensure correct terminal punctuation
            is_question = any(t in self.inflector.INTERROGATIVE_TOKENS for t in cleaned)
            mark = "?" if is_question else "।"
            if not bengali_text.endswith("।") and not bengali_text.endswith("?") and not bengali_text.endswith("!"):
                bengali_text = f"{bengali_text}{mark}"

        return {
            "raw_glosses": list(cleaned),
            "translated_text": bengali_text,
            "translated_en": english_text,
            "confidence": round(confidence, 2),
            "is_final": True
        }

    def process_stream(
        self,
        sign_slug: str,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Ingests continuous real-time predictions frame-by-frame.

        Returns the active translation state and whether a sentence boundary was triggered.
        """
        new_stable = self.debouncer.add_prediction(sign_slug, confidence, timestamp)
        is_boundary = self.debouncer.is_sentence_boundary()
        current_tokens = self.debouncer.get_stable_tokens()

        if is_boundary and current_tokens:
            flushed_tokens = self.debouncer.flush()
            result = self.translate(flushed_tokens)
            result["is_final"] = True
            result["boundary_triggered"] = True
            return result

        if current_tokens:
            result = self.translate(current_tokens)
            result["is_final"] = False
            result["boundary_triggered"] = False
            return result

        return {
            "raw_glosses": [],
            "translated_text": "",
            "translated_en": "",
            "confidence": 0.0,
            "is_final": False,
            "boundary_triggered": False
        }

    def reset(self):
        """Resets the debouncer and translator buffers."""
        self.debouncer.reset()
