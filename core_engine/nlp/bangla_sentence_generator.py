"""BanglaSentenceToSignCompiler — Categorical SOV Gloss Compiler.

Converts natural conversational Bangla sentences into syntactically ordered
BdSL gloss sequences following Bangla Academy grammar conventions:

    [সময়] + [কর্তা] + [কর্ম] + [ধাতুমূল/ক্রিয়া] + [প্রশ্নবাচক]

Differentiator from BdSLSyntaxEngine:
  - Explicit categorical role tagging (Time / Subject / Object / Verb / WH)
  - Tight integration with MasterBdSLLexicon for lexicon-aware token resolution
  - Returns recommended_nmm directly (AU01_02 brow-raise for questions)
  - Bridge method to_sentence_plan() emits a CoarticulatedSentencePlan compatible
    with the CoarticulatedSentenceSynthesizer
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared linguistic inventories  (merged with BdSLSyntaxEngine vocabulary)
# ---------------------------------------------------------------------------

# Full verb-stem normalisation table (superset of BdSLSyntaxEngine.STEM_DICTIONARY)
_VERB_STEM_MAP: Dict[str, str] = {
    # খাওয়া
    "খাচ্ছি": "খাওয়া", "খাবো": "খাওয়া", "খেয়েছি": "খাওয়া", "খেয়ে": "খাওয়া",
    "খায়": "খাওয়া", "খাচ্ছো": "খাওয়া", "খেয়েছে": "খাওয়া", "খেয়েছেন": "খাওয়া",
    # যাওয়া
    "যাচ্ছি": "যাওয়া", "যাবো": "যাওয়া", "গিয়েছিলাম": "যাওয়া", "যাও": "যাওয়া",
    "গিয়েছি": "যাওয়া", "গেছে": "যাওয়া", "যাচ্ছো": "যাওয়া", "গিয়েছেন": "যাওয়া",
    # আসা
    "আসছি": "আসা", "আসবো": "আসা", "এসেছি": "আসা", "আসুন": "আসা",
    "আসে": "আসা", "এসেছে": "আসা", "এসেছো": "আসা", "এসেছেন": "আসা",
    # দেখা
    "দেখছি": "দেখা", "দেখবো": "দেখা", "দেখেছি": "দেখা",
    "দেখছে": "দেখা", "দেখছেন": "দেখা",
    # বলা
    "বলছি": "বলা", "বলবো": "বলা", "বলেছি": "বলা", "বলুন": "বলা",
    "বলছে": "বলা", "বলছেন": "বলা",
    # করা
    "করছি": "করা", "করেছি": "করা", "করবো": "করা", "করে": "করা",
    "করছেন": "করা", "করছো": "করা",
    # পড়া
    "পড়ছি": "পড়া", "পড়েছি": "পড়া", "পড়বো": "পড়া", "পড়ে": "পড়া",
    "পড়ছেন": "পড়া", "পড়ছো": "পড়া",
    # ঘুমানো
    "ঘুমাচ্ছি": "ঘুমানো", "ঘুমিয়েছি": "ঘুমানো", "ঘুমাবো": "ঘুমানো",
    "ঘুমায়": "ঘুমানো", "ঘুমিয়েছেন": "ঘুমানো",
    # দেওয়া
    "দিচ্ছি": "দেওয়া", "দিয়েছি": "দেওয়া", "দেবো": "দেওয়া",
    "দেয়": "দেওয়া", "দিয়েছেন": "দেওয়া",
    # নেওয়া
    "নিচ্ছি": "নেওয়া", "নিয়েছি": "নেওয়া", "নেবো": "নেওয়া",
    "নেয়": "নেওয়া", "নিয়েছেন": "নেওয়া",
    # ফিরে-আসা / compound verbs
    "ফিরবো": "ফিরে-আসা", "ফিরে আসবো": "ফিরে-আসা", "ফিরে আসবে": "ফিরে-আসা",
    # শুনা / শোনা
    "শুনছি": "শোনা", "শুনবো": "শোনা", "শুনেছি": "শোনা",
    # লেখা
    "লিখছি": "লেখা", "লিখবো": "লেখা", "লিখেছি": "লেখা",
}

_TIME_WORDS: Set[str] = {
    "এখন", "আজকে", "আজ", "গতকাল", "আগামীকাল", "কাল",
    "সকালে", "বিকেলে", "রাতে", "দুপুরে", "প্রতিদিন",
    "সবসময়", "মাঝে", "মাঝেমধ্যে", "শীঘ্রই",
}

_SUBJECT_PRONOUNS: Set[str] = {
    "আমি", "আমরা", "তুমি", "তোমরা", "আপনি", "আপনারা",
    "তিনি", "সে", "তারা", "এরা", "ওরা", "ওই", "এই",
}

_WH_WORDS: Set[str] = {
    "কি", "কী", "কেন", "কোথায়", "কখন", "কে", "কাকে",
    "কার", "কীভাবে", "কিভাবে", "কেমন",
}

_CONJUNCTIONS: Set[str] = {
    "এবং", "ও", "কিন্তু", "অথবা", "বা", "আর",
    "সুতরাং", "তবে", "নাকি", "তাহলে", "তো",
}

_ALL_VERB_STEMS: Set[str] = set(_VERB_STEM_MAP.values())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BanglaSentenceToSignCompiler:
    """বাংলা একাডেমি ও ভাষাতাত্ত্বিক ব্যাকরণ নিয়মে সাধারণ বাংলা চলতি বাক্যকে
    ধারাবাহিক BdSL সাইন বাক্যে রূপান্তর করে।

    Parameters
    ----------
    master_lexicon_db : Dict[str, Any]
        A flat dict mapping Bengali gloss strings to sign metadata.
        Typically sourced from ``MasterBdSLLexicon.signs_by_bn`` or
        ``MasterBdSLLexicon.all_signs()``.
        Pass an empty dict ``{}`` for lexicon-free operation.
    lemmatizer : BanglaFoundationLemmatizer, optional
        When provided, used as a fallback for verb tokens not covered by the
        static ``_VERB_STEM_MAP`` and to strip noun case-endings from object
        tokens before gloss assignment.
    """

    def __init__(
        self,
        master_lexicon_db: Dict[str, Any],
        lemmatizer: Optional[Any] = None,
    ) -> None:
        self.lexicon: Dict[str, Any] = master_lexicon_db
        # Optional BanglaFoundationLemmatizer for unseen verb forms & noun stripping
        self.lemmatizer: Optional[Any] = lemmatizer

        # Expose for external inspection / extension
        self.verb_stem_map: Dict[str, str] = dict(_VERB_STEM_MAP)
        self.time_words: Set[str] = set(_TIME_WORDS)
        self.conjunctions: Set[str] = set(_CONJUNCTIONS)
        self.subject_pronouns: Set[str] = set(_SUBJECT_PRONOUNS)
        self.wh_words: Set[str] = set(_WH_WORDS)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def compile_sentence(self, natural_bengali_text: str) -> Dict[str, Any]:
        """Convert a natural Bangla sentence into an ordered BdSL gloss sequence.

        Applies:
          1. Punctuation stripping & tokenisation
          2. Conjunction deletion
          3. Verb root normalisation (stem mapping)
          4. Categorical SOV role tagging
          5. Lexicon-aware token validation
          6. Ordering: [সময়] [কর্তা] [কর্ম] [ক্রিয়া] [প্রশ্ন]
          7. NMM FACS recommendation derivation

        Returns
        -------
        Dict with keys:
          ``natural_input``    — original text
          ``syntactic_glosses``— ordered BdSL gloss list
          ``is_interrogative`` — bool
          ``recommended_nmm``  — AU float dict for facial expression
          ``lexicon_coverage`` — {gloss: bool} for each output gloss
          ``applied_rules``    — list of transformation descriptions
        """
        if not natural_bengali_text or not natural_bengali_text.strip():
            return self._empty_result(natural_bengali_text or "")

        is_question = (
            "?" in natural_bengali_text
            or any(w in natural_bengali_text.split() for w in self._wh_token_set(natural_bengali_text))
        )

        raw_tokens = self._tokenize(natural_bengali_text)
        applied_rules: List[str] = []

        time_tokens: List[str] = []
        subject_tokens: List[str] = []
        object_tokens: List[str] = []
        verb_tokens: List[str] = []
        interrogative_tokens: List[str] = []

        for word in raw_tokens:
            # Step 1: Conjunction deletion
            if word in self.conjunctions:
                applied_rules.append(f"ConjunctionDeletion({word})")
                continue

            # Step 2: Verb stem normalisation — static map first, lemmatizer fallback
            normalized = self.verb_stem_map.get(word, word)
            if normalized != word:
                applied_rules.append(f"VerbStemNormalisation({word}→{normalized})")
            elif self.lemmatizer is not None and normalized == word:
                # Fallback: regex-based suffix stripping for unseen inflected forms
                analysis = self.lemmatizer.analyse(word)
                if analysis.is_verb and analysis.changed:
                    normalized = analysis.lemma
                    applied_rules.append(
                        f"LemmatizerFallback({word}→{normalized}, {analysis.tense_hint})"
                    )

            # Step 3: Categorical role assignment
            if normalized in self.time_words or word in self.time_words:
                time_tokens.append(normalized)
            elif word in self.wh_words:
                interrogative_tokens.append(word)
            elif word in self.subject_pronouns:
                subject_tokens.append(normalized)
            elif normalized in _ALL_VERB_STEMS:
                verb_tokens.append(normalized)
            else:
                object_tokens.append(normalized)

        # Step 4: Bangla Academy SOV ordering
        syntactic_glosses = (
            time_tokens
            + subject_tokens
            + object_tokens
            + verb_tokens
            + interrogative_tokens
        )

        # Step 5: Lexicon coverage check
        lexicon_coverage = {g: self._in_lexicon(g) for g in syntactic_glosses}
        uncovered = [g for g, ok in lexicon_coverage.items() if not ok]
        if uncovered:
            applied_rules.append(f"LexiconMiss({', '.join(uncovered)})")

        # Step 6: Recommended NMM
        recommended_nmm = self._build_nmm(is_question, syntactic_glosses)

        return {
            "natural_input": natural_bengali_text,
            "syntactic_glosses": syntactic_glosses,
            "is_interrogative": is_question,
            "recommended_nmm": recommended_nmm,
            "lexicon_coverage": lexicon_coverage,
            "applied_rules": applied_rules,
        }

    def compile_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Compile a list of sentences, returning one result dict per sentence."""
        return [self.compile_sentence(s) for s in sentences]

    def to_sentence_plan(
        self,
        natural_bengali_text: str,
        template_id: str = "DYNAMIC",
        default_blend_ms: int = 150,
        default_stroke_ms: int = 550,
    ) -> "CoarticulatedSentencePlan":  # type: ignore[name-defined]  # imported lazily
        """Compile and wrap the result as a ``CoarticulatedSentencePlan``.

        Enables direct drop-in into ``CoarticulatedSentenceSynthesizer.synthesize()``.

        Parameters
        ----------
        natural_bengali_text : str
            Input sentence.
        template_id : str
            Identifier for the produced plan (defaults to "DYNAMIC").
        default_blend_ms : int
            Transition blend window applied uniformly (no corpus coarticulation map).
        default_stroke_ms : int
            Per-gloss stroke duration estimate used to compute total_duration_ms.
        """
        # Lazy import to avoid circular dependency
        from core_engine.nlp.sentence_plan_dto import (
            CoarticulatedSentencePlan,
            GlossTransitionSpec,
        )
        from core_engine.dsl.isharabakya_schema import NMMExpressionSegment

        result = self.compile_sentence(natural_bengali_text)
        glosses: List[str] = result["syntactic_glosses"]

        transitions = [
            GlossTransitionSpec(
                from_gloss=glosses[i],
                to_gloss=glosses[i + 1],
                blend_ms=default_blend_ms,
            )
            for i in range(len(glosses) - 1)
        ]

        # Derive a simple NMM timeline from recommended_nmm
        nmm = result["recommended_nmm"]
        total_ms = max(
            len(glosses) * default_stroke_ms
            + sum(t.blend_ms for t in transitions),
            500,
        )
        nmm_timeline = [
            NMMExpressionSegment(
                timestamp_range=(0, total_ms),
                facs={k: float(v) for k, v in nmm.items()},
            )
        ]

        return CoarticulatedSentencePlan(
            template_id=template_id,
            spoken_text=natural_bengali_text,
            gloss_sequence=glosses,
            transitions=transitions,
            nmm_timeline=nmm_timeline,
            total_duration_ms=total_ms,
            domain="General",
            applied_rules=result["applied_rules"],
            is_interrogative=result["is_interrogative"],
        )

    # ------------------------------------------------------------------
    # Lexicon integration helpers
    # ------------------------------------------------------------------

    def load_from_master_lexicon(self, lexicon: Any) -> None:
        """Populate the internal lexicon from a ``MasterBdSLLexicon`` instance.

        Accepts any object exposing ``signs_by_bn: Dict[str, Any]``.
        """
        if hasattr(lexicon, "signs_by_bn"):
            self.lexicon = lexicon.signs_by_bn
            logger.info(
                "BanglaSentenceToSignCompiler: loaded %d signs from MasterBdSLLexicon",
                len(self.lexicon),
            )
        else:
            logger.warning(
                "BanglaSentenceToSignCompiler.load_from_master_lexicon: "
                "provided object has no 'signs_by_bn' attribute."
            )

    def _in_lexicon(self, gloss: str) -> bool:
        """Return True if the gloss is resolvable in the attached lexicon."""
        return bool(self.lexicon.get(gloss))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Strip punctuation and split into whitespace tokens."""
        cleaned = re.sub(r"[।?!,;:\"\'\(\)\[\]]", " ", text)
        return [t.strip() for t in cleaned.split() if t.strip()]

    def _wh_token_set(self, text: str) -> Set[str]:
        """Return which WH words appear in the raw text."""
        tokens = set(re.sub(r"[।?!,;:\"\'\(\)\[\]]", " ", text).split())
        return tokens & self.wh_words

    def _build_nmm(self, is_question: bool, glosses: List[str]) -> Dict[str, float]:
        """Derive recommended Non-Manual Marker FACS parameters.

        For polar/WH questions: brow raise (AU01+AU02) + head tilt forward.
        For positive sentiment glosses: lip-corner pull (AU12).
        Default: neutral expression.
        """
        _POSITIVE_GLOSSES = {"ভালো", "ধন্যবাদ", "স্বাগতম", "সুন্দর", "খুশি"}
        _ALERT_GLOSSES = {"অসুস্থ", "ব্যথা", "আগুন", "বন্যা", "সাবধান", "বিপদ"}

        if is_question:
            return {
                "AU01_02_brow_raise": 0.85,
                "AU04_brow_lowerer": 0.0,
                "AU12_lip_corner_pull": 0.0,
                "head_tilt_forward": 6.0,
            }

        if any(g in _POSITIVE_GLOSSES for g in glosses):
            return {
                "AU01_02_brow_raise": 0.0,
                "AU04_brow_lowerer": 0.0,
                "AU06_cheek_raiser": 0.4,
                "AU12_lip_corner_pull": 0.7,
                "head_tilt_forward": 0.0,
            }

        if any(g in _ALERT_GLOSSES for g in glosses):
            return {
                "AU01_02_brow_raise": 0.3,
                "AU04_brow_lowerer": 0.7,
                "AU12_lip_corner_pull": 0.0,
                "head_tilt_forward": -2.0,
            }

        return {
            "AU01_02_brow_raise": 0.0,
            "AU04_brow_lowerer": 0.0,
            "AU12_lip_corner_pull": 0.2,
            "head_tilt_forward": 0.0,
        }

    @staticmethod
    def _empty_result(text: str) -> Dict[str, Any]:
        return {
            "natural_input": text,
            "syntactic_glosses": [],
            "is_interrogative": False,
            "recommended_nmm": {
                "AU01_02_brow_raise": 0.0,
                "AU12_lip_corner_pull": 0.0,
                "head_tilt_forward": 0.0,
            },
            "lexicon_coverage": {},
            "applied_rules": [],
        }
