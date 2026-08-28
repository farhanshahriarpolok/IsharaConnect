"""Unit tests for BanglaFoundationLemmatizer.

Covers:
  - strip_inflection() noun vibhakti stripping (backward-compat interface)
  - extract_root_verb() verb suffix stripping (backward-compat interface)
  - analyse() MorphAnalysis DTO fields
  - Irregular root canonicalisation (শুন→শোনা, লিখ→লেখা)
  - Longest-match-first suffix ordering
  - lemmatize() / lemmatize_tokens() / analyse_tokens() batch API
  - direct map override (extra_verb_stems)
  - Edge cases: short tokens, empty string, no-match passthrough
  - BanglaSentenceToSignCompiler lemmatizer= fallback integration
  - BdSLSyntaxEngine lemmatizer= fallback integration
"""

from __future__ import annotations

import pytest

from core_engine.nlp.bangla_foundation_lemmatizer import (
    BanglaFoundationLemmatizer,
    MorphAnalysis,
)
from core_engine.nlp.bangla_sentence_generator import BanglaSentenceToSignCompiler
from core_engine.nlp.bdsl_syntax_engine import BdSLSyntaxEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lemmatizer() -> BanglaFoundationLemmatizer:
    return BanglaFoundationLemmatizer()


@pytest.fixture
def lemmatizer_with_overrides() -> BanglaFoundationLemmatizer:
    return BanglaFoundationLemmatizer(
        extra_verb_stems={"কথা_বলেছি": "বলা", "ঘুরে_এসেছি": "ঘুরে-আসা"}
    )


# ===========================================================================
# 1. strip_inflection() — noun case-ending stripping
# ===========================================================================

class TestStripInflection:

    @pytest.mark.parametrize("surface,expected", [
        ("বাড়িতে",    "বাড়ি"),
        ("ডাক্তারের",  "ডাক্তার"),
        ("হাসপাতালে",  "হাসপাতাল"),
        ("বাজারে",     "বাজার"),
        ("স্কুলটা",    "স্কুল"),
        ("ছেলেটি",    "ছেলে"),
        ("বইগুলো",    "বই"),
        ("মানুষদের",   "মানুষ"),
    ])
    def test_noun_stripping(self, lemmatizer, surface, expected):
        assert lemmatizer.strip_inflection(surface) == expected

    def test_no_strip_when_no_suffix(self, lemmatizer):
        """Words without a recognisable suffix should be returned unchanged."""
        assert lemmatizer.strip_inflection("বাড়ি") == "বাড়ি"

    def test_too_short_not_stripped(self, lemmatizer):
        """Tokens shorter than 3 characters should not be modified."""
        assert lemmatizer.strip_inflection("এ") == "এ"


# ===========================================================================
# 2. extract_root_verb() — verb suffix stripping
# ===========================================================================

class TestExtractRootVerb:

    @pytest.mark.parametrize("surface,expected", [
        ("খাচ্ছি",      "খাওয়া"),
        ("খাচ্ছিলাম",  "খাওয়া"),
        ("যাচ্ছি",      "যাওয়া"),
        ("যাবো",       "যাওয়া"),
        ("লিখছি",      "লেখা"),    # irregular: লিখ → লেখা
        ("শুনছি",      "শোনা"),    # irregular: শুন → শোনা
        ("পড়েছি",     "পড়া"),
        ("করেছেন",     "করা"),
        ("দেখবেন",     "দেখা"),
        ("ঘুমাচ্ছি",   "ঘুমানো"),  # irregular: ঘুমা → ঘুমানো
    ])
    def test_verb_root_extraction(self, lemmatizer, surface, expected):
        assert lemmatizer.extract_root_verb(surface) == expected

    def test_unknown_verb_returned_unchanged(self, lemmatizer):
        """Verbs with no matching suffix should be returned unchanged."""
        result = lemmatizer.extract_root_verb("করা")
        assert result == "করা"


# ===========================================================================
# 3. analyse() MorphAnalysis DTO
# ===========================================================================

class TestAnalyse:

    def test_is_verb_true_for_verb(self, lemmatizer):
        analysis = lemmatizer.analyse("খাচ্ছি")
        assert analysis.is_verb is True

    def test_is_verb_false_for_noun(self, lemmatizer):
        analysis = lemmatizer.analyse("বাড়িতে")
        assert analysis.is_verb is False

    def test_surface_preserved(self, lemmatizer):
        analysis = lemmatizer.analyse("খাচ্ছি")
        assert analysis.surface == "খাচ্ছি"

    def test_lemma_set(self, lemmatizer):
        analysis = lemmatizer.analyse("খাচ্ছি")
        assert analysis.lemma == "খাওয়া"

    def test_stripped_suffix_non_empty(self, lemmatizer):
        analysis = lemmatizer.analyse("খাচ্ছি")
        assert analysis.stripped_suffix != ""

    def test_tense_hint_non_empty_for_verb(self, lemmatizer):
        analysis = lemmatizer.analyse("খাচ্ছিলাম")
        assert "past" in analysis.tense_hint or "imperfect" in analysis.tense_hint

    def test_changed_true_when_rule_fires(self, lemmatizer):
        assert lemmatizer.analyse("খাচ্ছি").changed is True

    def test_changed_false_for_passthrough(self, lemmatizer):
        assert lemmatizer.analyse("করা").changed is False

    def test_no_rule_fires_on_empty(self, lemmatizer):
        analysis = lemmatizer.analyse("")
        assert analysis.lemma == ""
        assert analysis.changed is False


# ===========================================================================
# 4. Irregular root canonicalisation
# ===========================================================================

class TestIrregularRoots:

    def test_লিখ_to_লেখা(self, lemmatizer):
        assert lemmatizer.lemmatize("লিখেছেন") == "লেখা"

    def test_শুন_to_শোনা(self, lemmatizer):
        assert lemmatizer.lemmatize("শুনছি") == "শোনা"

    def test_ঘুমা_to_ঘুমানো(self, lemmatizer):
        assert lemmatizer.lemmatize("ঘুমাচ্ছি") == "ঘুমানো"

    def test_খা_to_খাওয়া(self, lemmatizer):
        # The stem "খা" is in the irregular roots table
        assert lemmatizer.lemmatize("খাচ্ছি") == "খাওয়া"

    def test_যা_to_যাওয়া(self, lemmatizer):
        assert lemmatizer.lemmatize("যাচ্ছি") == "যাওয়া"


# ===========================================================================
# 5. Longest-match-first suffix ordering
# ===========================================================================

class TestLongestMatchFirst:

    def test_past_imperfect_not_confused_with_future(self, lemmatizer):
        """'ছিলাম' (past-imperfect) must take priority over 'লাম' or shorter."""
        analysis = lemmatizer.analyse("খাচ্ছিলাম")
        assert "past" in analysis.tense_hint

    def test_past_perfect_longer_suffix_matched(self, lemmatizer):
        """'েছিলাম' must match before 'ছি' or 'ি'."""
        analysis = lemmatizer.analyse("করেছিলাম")
        assert analysis.is_verb is True
        assert analysis.lemma == "করা"


# ===========================================================================
# 6. Batch API
# ===========================================================================

class TestBatchAPI:

    def test_lemmatize_tokens(self, lemmatizer):
        tokens = ["খাচ্ছি", "বাড়িতে", "যাবো"]
        results = lemmatizer.lemmatize_tokens(tokens)
        assert results[0] == "খাওয়া"
        assert results[1] == "বাড়ি"
        assert results[2] == "যাওয়া"

    def test_analyse_tokens_count(self, lemmatizer):
        tokens = ["লিখছি", "পড়েছি"]
        analyses = lemmatizer.analyse_tokens(tokens)
        assert len(analyses) == 2
        assert all(isinstance(a, MorphAnalysis) for a in analyses)

    def test_lemmatize_empty_list(self, lemmatizer):
        assert lemmatizer.lemmatize_tokens([]) == []


# ===========================================================================
# 7. Direct map override (extra_verb_stems)
# ===========================================================================

class TestDirectMapOverride:

    def test_override_takes_precedence(self, lemmatizer_with_overrides):
        result = lemmatizer_with_overrides.lemmatize("কথা_বলেছি")
        assert result == "বলা"

    def test_override_analysis_is_verb(self, lemmatizer_with_overrides):
        analysis = lemmatizer_with_overrides.analyse("কথা_বলেছি")
        assert analysis.is_verb is True
        assert analysis.tense_hint == "known"


# ===========================================================================
# 8. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_string(self, lemmatizer):
        assert lemmatizer.lemmatize("") == ""

    def test_single_char(self, lemmatizer):
        assert lemmatizer.lemmatize("া") == "া"

    def test_two_char_token(self, lemmatizer):
        """Tokens < 3 chars should be returned unchanged."""
        assert lemmatizer.lemmatize("এ") == "এ"

    def test_already_canonical(self, lemmatizer):
        assert lemmatizer.lemmatize("করা") == "করা"

    def test_whitespace_stripped(self, lemmatizer):
        """analyse() is given a non-empty string: no crash."""
        analysis = lemmatizer.analyse("   ")
        # strip() is empty → passthrough
        assert analysis.lemma == "   " or analysis.surface == "   "


# ===========================================================================
# 9. BanglaSentenceToSignCompiler — lemmatizer= fallback integration
# ===========================================================================

class TestCompilerLemmatizerFallback:

    @pytest.fixture
    def compiler(self, lemmatizer) -> BanglaSentenceToSignCompiler:
        return BanglaSentenceToSignCompiler(
            master_lexicon_db={}, lemmatizer=lemmatizer
        )

    def test_unseen_verb_form_resolved_via_lemmatizer(self, compiler):
        """'লিখছেন' is not in _VERB_STEM_MAP; lemmatizer should resolve it to 'লেখা'."""
        result = compiler.compile_sentence("আপনি লিখছেন।")
        assert "লেখা" in result["syntactic_glosses"]

    def test_lemmatizer_fallback_rule_recorded(self, compiler):
        result = compiler.compile_sentence("আপনি লিখছেন।")
        assert any("LemmatizerFallback" in r for r in result["applied_rules"])

    def test_static_map_still_preferred(self, compiler):
        """For 'যাচ্ছি' (in static map), the rule should say VerbStemNormalisation."""
        result = compiler.compile_sentence("আমি যাচ্ছি।")
        assert any("VerbStemNormalisation" in r for r in result["applied_rules"])
        assert not any("LemmatizerFallback" in r for r in result["applied_rules"])

    def test_no_lemmatizer_compiler_still_works(self):
        """Compiler without lemmatizer= should not raise."""
        compiler = BanglaSentenceToSignCompiler(master_lexicon_db={})
        result = compiler.compile_sentence("আমি বাড়ি যাচ্ছি।")
        assert isinstance(result["syntactic_glosses"], list)


# ===========================================================================
# 10. BdSLSyntaxEngine — lemmatizer= fallback integration
# ===========================================================================

class TestSyntaxEngineLemmatizerFallback:

    @pytest.fixture
    def engine_with_lemmatizer(self, lemmatizer) -> BdSLSyntaxEngine:
        return BdSLSyntaxEngine(lemmatizer=lemmatizer)

    def test_unseen_verb_resolved_via_lemmatizer(self, engine_with_lemmatizer):
        result = engine_with_lemmatizer.text_to_bdsl_gloss("আমি শুনছি।")
        assert "শোনা" in result["glosses"]

    def test_lemmatizer_rule_recorded_in_syntax_engine(self, engine_with_lemmatizer):
        result = engine_with_lemmatizer.text_to_bdsl_gloss("আমি লিখছি।")
        assert any("LemmatizerFallback" in r for r in result["applied_rules"])

    def test_syntax_engine_without_lemmatizer_unchanged(self):
        """Default BdSLSyntaxEngine() must still work (backward compat)."""
        engine = BdSLSyntaxEngine()
        result = engine.text_to_bdsl_gloss("আমি বাড়ি যাচ্ছি।")
        assert "glosses" in result
