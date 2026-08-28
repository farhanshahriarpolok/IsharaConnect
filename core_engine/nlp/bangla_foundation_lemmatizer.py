"""BanglaFoundationLemmatizer — Bangla Language Foundation Morphological Analyser.

Rule-based suffix-stripping lemmatizer for Bangla surface forms:
  - Noun/Pronoun vibhakti (কারক বিভক্তি) stripping: বাড়িতে → বাড়ি
  - Verb suffix stripping → canonical root recovery: খাচ্ছিলাম → খাওয়া
  - Full morphological breakdown (tense, person, aspect) for each form

Design decisions
----------------
- Ordered suffix patterns applied longest-match-first to avoid partial strip.
- Irregular stem canonicalisation table handles roots whose surface form diverges
  from BLF canonical lemma (শুন → শোনা, লিখ → লেখা, etc.).
- Returns original token unchanged when no rule fires (safe fallback).
- Thread-safe: no mutable state after __init__.

Relationship to existing modules
---------------------------------
- BengaliMorphologicalInflector (bengali_inflection.py): conjugation tables,
  canonical root → surface form (synthesis direction).
- BanglaFoundationLemmatizer                           : surface form → canonical
  root (analysis direction). Complements the inflector.
- BanglaSentenceToSignCompiler (bangla_sentence_generator.py): uses this as a
  fallback when the static _VERB_STEM_MAP does not contain an inflected form.
- BdSLSyntaxEngine (bdsl_syntax_engine.py): uses this as a fallback when the
  static STEM_DICTIONARY does not contain an inflected form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Morphological analysis result DTO
# ---------------------------------------------------------------------------

@dataclass
class MorphAnalysis:
    """Complete morphological breakdown for a single Bangla surface token."""

    surface: str
    """Original (uninflected/conjugated) surface form."""

    lemma: str
    """Canonical dictionary form (বাড়িতে → বাড়ি, খাচ্ছিলাম → খাওয়া)."""

    is_verb: bool = False
    """True when the token was identified as a conjugated verb."""

    stripped_suffix: str = ""
    """The suffix that was removed (for traceability)."""

    tense_hint: str = ""
    """Coarse tense/aspect label inferred from the matched suffix."""

    changed: bool = False
    """True when lemma != surface (a rule fired)."""


# ---------------------------------------------------------------------------
# Internal linguistic inventories
# ---------------------------------------------------------------------------

# Irregular stem → canonical lemma map
# Keys are the stem left after suffix stripping; values are the BLF canonical forms.
_IRREGULAR_ROOTS: Dict[str, str] = {
    # Vowel-mutation roots
    "খা":    "খাওয়া",
    "যা":    "যাওয়া",
    "দে":    "দেওয়া",
    "নে":    "নেওয়া",
    "হ":     "হওয়া",
    # Surface-canonical divergence (লিখ/শুন surface ≠ লেখা/শোনা canonical)
    "লিখ":  "লেখা",
    "শুন":  "শোনা",
    "শোন":  "শোনা",
    # আসা suppletive root
    "আ":    "আসা",
    # ঘুম → ঘুমানো
    "ঘুম":  "ঘুমানো",
    "ঘুমা": "ঘুমানো",
    # Compound verb fragments
    "ফির":  "ফিরে-আসা",
    "ফিরে": "ফিরে-আসা",
}

# Set of known verb roots for stem plausibility check
_KNOWN_VERB_STEMS: frozenset = frozenset(_IRREGULAR_ROOTS.keys()) | frozenset([
    "খা", "যা", "আস", "দেখ", "বল", "কর", "পড়", "লিখ", "শুন", "শোন",
    "ঘুম", "ঘুমা", "নে", "দে", "হ", "থাক", "ভালোবাস", "ফির", "ফিরে",
    "বস", "উঠ", "নামছ", "নাম", "কাঁদ", "হাসছ", "হাস", "ডাক",
])

# Verb suffix patterns ordered longest → shortest for greedy matching.
# Each entry: (compiled_regex, suffix_label_for_tense_hint)
_VERB_SUFFIXES: List[Tuple[re.Pattern[str], str]] = [
    # চ্ছিলাম-class: present-continuous past MUST come first to beat ছিলাম-only imperfect
    (re.compile(r'(চ্ছিলাম|চ্ছিলেন|চ্ছিলে|চ্ছিলিস|চ্ছিলি|চ্ছিল)$'), 'present_continuous_past'),
    # Past imperfect / habitual past
    (re.compile(r'(তেছিলাম|ছিলাম|তেছিলে|ছিলেন|ছিলিস|ছিলে|ছিলি|ছিল)$'), 'past_imperfect'),
    # Present perfect — strip whole suffix inc. leading ে
    (re.compile(r'(েছিলাম|েছিলেন|েছিলে|েছিলিস|েছেন|েছিস|েছ|েছি|েছে)$'), 'past_perfect'),
    # Present continuous
    (re.compile(r'(তেছি|তেছেন|তেছিস|চ্ছেন|চ্ছিস|চ্ছি|চ্ছে|চ্ছো|ছেন|ছিস|ছি|ছে)$'), 'present_continuous'),
    # Future
    (re.compile(r'(বেন|বো|বি|বে|ব)$'), 'future'),
    # Present simple / habitual
    (re.compile(r'(তাম|তেন|তিস|তি|তো)$'), 'present_simple_habitual'),
    # Conditional / imperative
    (re.compile(r'(লেন|লিস|লো)$'), 'conditional_imperative'),
]


# Noun / pronoun case-ending patterns — longest-match first
_NOUN_SUFFIXES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r'(গুলোতে|দেরকে)$'), 'locative_plural/accusative_plural'),
    (re.compile(r'(গুলোর|গুলো)$'), 'genitive_plural/plural'),
    (re.compile(r'(খানায়|খানাতে|খানিতে)$'), 'locative'),
    (re.compile(r'(খানা|খানি)$'), 'specific'),
    (re.compile(r'(দের)$'), 'genitive_plural'),
    # Accusative কে  (not রে here — রে is ambiguous, handled below)
    (re.compile(r'(কে)$'), 'accusative'),
    # Locative তে / য়ে / এ  (multi-char, unambiguous)
    (re.compile(r'(তে|য়ে|এ)$'), 'locative'),
    # Genitive ের / র (multi-char, unambiguous)
    (re.compile(r'(ের)$'), 'genitive'),
    # Specific টা / টি
    (re.compile(r'(টা|টি)$'), 'specific'),
    # Plain locative ে — must come before the ambiguous র pattern
    (re.compile(r'(ে)$'), 'locative_short'),
    # Genitive র — only after ে locative has been tried; skip if last consonant is র
    (re.compile(r'(?<=[^র])(র)$'), 'genitive_r'),
    # Accusative রে — after ে and locative patterns (catches তোমারে etc.)
    (re.compile(r'(রে)$'), 'accusative_re'),
    # য় adverbial
    (re.compile(r'(য়)$'), 'adverbial'),
]


# Minimum token length below which no stripping is attempted
_MIN_TOKEN_LEN: int = 3


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BanglaFoundationLemmatizer:
    """বাংলা-ভাষা-ফাউন্ডেশন ভিত্তিক ধাতু নিষ্কাশন ও বিভক্তি ফিল্টারিং মডিউল।

    Parameters
    ----------
    extra_verb_stems : Dict[str, str], optional
        Additional ``{surface: canonical}`` override mappings merged at runtime.
        Entries here take precedence over the built-in suffix-stripping rules.
    """

    def __init__(
        self,
        extra_verb_stems: Optional[Dict[str, str]] = None,
    ) -> None:
        # Module-level patterns (pre-compiled, shared across instances)
        self._verb_suffixes = _VERB_SUFFIXES
        self._noun_suffixes = _NOUN_SUFFIXES
        self._irregular_roots: Dict[str, str] = dict(_IRREGULAR_ROOTS)
        self._known_verb_stems: frozenset = _KNOWN_VERB_STEMS

        # Caller-supplied overrides
        if extra_verb_stems:
            self._direct_map: Dict[str, str] = dict(extra_verb_stems)
        else:
            self._direct_map = {}

        # Legacy-compatible public attributes (match original class interface)
        self.inflections: List[str] = [
            r'(ে|ের|কে|রে|তে|য়ে|য়|র|দের|গুলোতে|গুলো|খানা|খানি|টা|টি)$'
        ]
        self.verb_suffix_patterns: List[Tuple[str, str]] = [
            (r'(তেছি|ছিলাম|বেন|বো|বি|িস|েছি|তাম|তে|লে|চ্ছিস|চ্ছি|চ্ছে|ব)$', '')
        ]

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def analyse(self, token: str) -> MorphAnalysis:
        """Return a complete morphological breakdown for a single token.

        Analysis order:
          1. Direct lookup override (highest priority)
          2. Noun case-ending stripping (tried first to avoid ambiguous suffixes
             like তে/ে being mistaken for verb habitual/short person endings)
          3. Verb suffix stripping (only fires if stem is plausibly a verb root)
        """
        if not token or not token.strip():
            return MorphAnalysis(surface=token, lemma=token)

        # 0. Direct lookup override (caller-supplied or class defaults)
        if token in self._direct_map:
            canonical = self._direct_map[token]
            return MorphAnalysis(
                surface=token, lemma=canonical, is_verb=True,
                stripped_suffix="(override)", tense_hint="known", changed=True
            )

        # 1. Try noun / case-ending stripping FIRST (unambiguous for common endings)
        noun_result = self._strip_noun_suffix(token)
        if noun_result is not None:
            return noun_result

        # 2. Try verb suffix stripping (plausibility-gated)
        verb_result = self._strip_verb_suffix(token)
        if verb_result is not None:
            return verb_result

        # 3. No rule fired — return unchanged
        return MorphAnalysis(surface=token, lemma=token)

    def lemmatize(self, token: str) -> str:
        """Return just the canonical lemma string for *token*."""
        return self.analyse(token).lemma

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Return canonical lemmas for a list of tokens (in order)."""
        return [self.lemmatize(t) for t in tokens]

    def analyse_tokens(self, tokens: List[str]) -> List[MorphAnalysis]:
        """Return full MorphAnalysis for a list of tokens."""
        return [self.analyse(t) for t in tokens]

    # ------------------------------------------------------------------
    # Legacy interface (matches original BanglaFoundationLemmatizer API)
    # ------------------------------------------------------------------

    def strip_inflection(self, noun: str) -> str:
        """বিশেষ্য বা সর্বনামের বিভক্তি ছেঁটে মূল শব্দ বের করে।

        Examples
        --------
        >>> lemmatizer.strip_inflection("বাড়িতে")
        'বাড়ি'
        >>> lemmatizer.strip_inflection("ডাক্তারের")
        'ডাক্তার'
        """
        result = self._strip_noun_suffix(noun)
        return result.lemma if result else noun

    def extract_root_verb(self, verb: str) -> str:
        """ক্রিয়াপদের প্রত্যয় ছেঁটে মূল ক্রিয়া বের করে।

        Examples
        --------
        >>> lemmatizer.extract_root_verb("খাচ্ছিলাম")
        'খাওয়া'
        >>> lemmatizer.extract_root_verb("লিখছি")
        'লেখা'
        """
        result = self._strip_verb_suffix(verb)
        return result.lemma if result else verb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_verb_suffix(self, token: str) -> Optional[MorphAnalysis]:
        """Try each verb suffix pattern (longest first). Return MorphAnalysis or None.

        Accepts the result only if the remaining stem is plausible as a verb root
        (found in _KNOWN_VERB_STEMS or is a short consonant-cluster stem).
        """
        if len(token) < _MIN_TOKEN_LEN:
            return None

        for pattern, tense_hint in self._verb_suffixes:
            m = pattern.search(token)
            if m:
                suffix = m.group(0)
                stem = token[: m.start()]
                if not stem:
                    continue
                # Plausibility gate: only accept if stem is a known verb root
                # OR the token is clearly a long conjugated form (len > 4).
                is_plausible = (
                    stem in self._known_verb_stems
                    or len(token) > 4
                )
                if not is_plausible:
                    continue
                canonical = self._canonicalise_verb_stem(stem)
                return MorphAnalysis(
                    surface=token,
                    lemma=canonical,
                    is_verb=True,
                    stripped_suffix=suffix,
                    tense_hint=tense_hint,
                    changed=(canonical != token),
                )
        return None

    def _canonicalise_verb_stem(self, stem: str) -> str:
        """Map a raw verb stem to its canonical Bangla dictionary lemma."""
        # Strip any residual vowel sign (matra) left by suffix stripping
        # e.g., 'করে' (from করেছিলাম after stripping েছিলাম) → 'কর'
        clean_stem = re.sub(r'[েিীুূ]$', '', stem)

        # Direct irregular lookup (try both cleaned and original stem)
        if clean_stem in self._irregular_roots:
            return self._irregular_roots[clean_stem]
        if stem in self._irregular_roots:
            return self._irregular_roots[stem]

        # Regular stems: append standard infinitive suffix '-া'
        if clean_stem:
            if clean_stem.endswith('া') or clean_stem.endswith('ানো') or clean_stem.endswith('ওয়া'):
                return clean_stem
            return clean_stem + 'া'

        return stem

    def _strip_noun_suffix(self, token: str) -> Optional[MorphAnalysis]:
        """Try each noun case-ending pattern (longest first). Return MorphAnalysis or None."""
        if len(token) < _MIN_TOKEN_LEN:
            return None

        for pattern, case_label in self._noun_suffixes:
            m = pattern.search(token)
            if m:
                suffix = m.group(0)
                root = token[: m.start()]
                if len(root) < 2:
                    continue
                return MorphAnalysis(
                    surface=token,
                    lemma=root,
                    is_verb=False,
                    stripped_suffix=suffix,
                    tense_hint=case_label,
                    changed=(root != token),
                )
        return None
