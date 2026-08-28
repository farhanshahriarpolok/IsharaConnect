"""Unit tests for BanglaSentenceToSignCompiler.

Covers:
  - Declarative sentence SOV ordering (time / subject / object / verb)
  - Conjunction deletion
  - Verb stem normalisation
  - WH-question detection & terminal interrogative placement
  - NMM recommendation (question brow-raise, positive, alert, neutral)
  - Lexicon coverage reporting
  - Empty / whitespace input guard
  - compile_batch()
  - to_sentence_plan() CoarticulatedSentencePlan integration
  - SentenceToGlossPipeline(compiler=...) routing
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from core_engine.nlp.bangla_sentence_generator import BanglaSentenceToSignCompiler
from core_engine.nlp.sentence_to_gloss_pipeline import (
    CoarticulatedSentencePlan,
    SentenceToGlossPipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def compiler_empty_lexicon() -> BanglaSentenceToSignCompiler:
    return BanglaSentenceToSignCompiler(master_lexicon_db={})


@pytest.fixture
def compiler_with_lexicon() -> BanglaSentenceToSignCompiler:
    """Compiler with a minimal mock lexicon covering known glosses."""
    lexicon: Dict[str, Any] = {
        "আমি":   {"label_bn": "আমি"},
        "বাড়ি":  {"label_bn": "বাড়ি"},
        "যাওয়া": {"label_bn": "যাওয়া"},
        "এখন":   {"label_bn": "এখন"},
        "খাওয়া": {"label_bn": "খাওয়া"},
        "ভাত":   {"label_bn": "ভাত"},
    }
    return BanglaSentenceToSignCompiler(master_lexicon_db=lexicon)


# ---------------------------------------------------------------------------
# 1. Declarative sentence — SOV ordering
# ---------------------------------------------------------------------------

class TestDeclarativeSentenceOrdering:

    def test_subject_verb_ordering(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি যাচ্ছি।")
        glosses = result["syntactic_glosses"]
        assert "আমি" in glosses
        assert "যাওয়া" in glosses
        # Subject must appear before verb
        assert glosses.index("আমি") < glosses.index("যাওয়া")

    def test_time_first_ordering(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("এখন আমি বাড়ি যাচ্ছি।")
        glosses = result["syntactic_glosses"]
        assert glosses[0] == "এখন", "Temporal anchor must come first"

    def test_object_before_verb(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি ভাত খাচ্ছি।")
        glosses = result["syntactic_glosses"]
        assert "খাওয়া" in glosses
        assert "ভাত" in glosses
        assert glosses.index("ভাত") < glosses.index("খাওয়া")

    def test_full_sov_order(self, compiler_empty_lexicon):
        """এখন আমি ভাত খাচ্ছি → [এখন, আমি, ভাত, খাওয়া]"""
        result = compiler_empty_lexicon.compile_sentence("এখন আমি ভাত খাচ্ছি।")
        glosses = result["syntactic_glosses"]
        expected_order = ["এখন", "আমি", "ভাত", "খাওয়া"]
        assert glosses == expected_order


# ---------------------------------------------------------------------------
# 2. Conjunction deletion
# ---------------------------------------------------------------------------

class TestConjunctionDeletion:

    def test_conjunction_deleted(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি এবং তুমি যাচ্ছি।")
        assert "এবং" not in result["syntactic_glosses"]

    def test_multiple_conjunctions_deleted(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি ও তুমি এবং সে যাচ্ছি।")
        for conj in ["ও", "এবং"]:
            assert conj not in result["syntactic_glosses"]

    def test_applied_rules_records_deletion(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি এবং বাড়ি যাচ্ছি।")
        assert any("ConjunctionDeletion" in r for r in result["applied_rules"])


# ---------------------------------------------------------------------------
# 3. Verb stem normalisation
# ---------------------------------------------------------------------------

class TestVerbStemNormalisation:

    @pytest.mark.parametrize("inflected,expected_stem", [
        ("যাচ্ছি",   "যাওয়া"),
        ("খাচ্ছি",   "খাওয়া"),
        ("আসছি",    "আসা"),
        ("দেখছি",   "দেখা"),
        ("বলছি",    "বলা"),
        ("গিয়েছিলাম", "যাওয়া"),
        ("ফিরবো",   "ফিরে-আসা"),
    ])
    def test_stem_normalisation(self, compiler_empty_lexicon, inflected, expected_stem):
        result = compiler_empty_lexicon.compile_sentence(f"আমি {inflected}।")
        assert expected_stem in result["syntactic_glosses"]

    def test_stem_rule_recorded(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি যাচ্ছি।")
        assert any("VerbStemNormalisation" in r for r in result["applied_rules"])


# ---------------------------------------------------------------------------
# 4. WH-question detection & terminal placement
# ---------------------------------------------------------------------------

class TestInterrogativeDetection:

    def test_question_mark_detected(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("তুমি কোথায় যাচ্ছো?")
        assert result["is_interrogative"] is True

    def test_wh_word_detected(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("তুমি কেন এখানে আছো।")
        assert result["is_interrogative"] is True

    def test_wh_word_placed_last(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("তুমি কোথায় যাচ্ছো?")
        glosses = result["syntactic_glosses"]
        assert glosses[-1] == "কোথায়"

    def test_declarative_not_interrogative(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি বাড়ি যাচ্ছি।")
        assert result["is_interrogative"] is False


# ---------------------------------------------------------------------------
# 5. NMM recommendations
# ---------------------------------------------------------------------------

class TestNMMRecommendations:

    def test_question_brow_raise(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("তুমি কোথায় যাচ্ছো?")
        nmm = result["recommended_nmm"]
        assert nmm["AU01_02_brow_raise"] == pytest.approx(0.85)
        assert nmm["head_tilt_forward"] == pytest.approx(6.0)

    def test_neutral_no_brow_raise(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি বাড়ি যাচ্ছি।")
        nmm = result["recommended_nmm"]
        assert nmm["AU01_02_brow_raise"] == pytest.approx(0.0)

    def test_positive_gloss_triggers_au12(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি ধন্যবাদ।")
        nmm = result["recommended_nmm"]
        assert nmm.get("AU12_lip_corner_pull", 0.0) > 0.0

    def test_alert_gloss_triggers_au04(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি সাবধান।")
        nmm = result["recommended_nmm"]
        assert nmm.get("AU04_brow_lowerer", 0.0) > 0.0


# ---------------------------------------------------------------------------
# 6. Lexicon coverage
# ---------------------------------------------------------------------------

class TestLexiconCoverage:

    def test_known_gloss_covered(self, compiler_with_lexicon):
        result = compiler_with_lexicon.compile_sentence("আমি বাড়ি যাচ্ছি।")
        cov = result["lexicon_coverage"]
        assert cov.get("আমি") is True
        assert cov.get("যাওয়া") is True

    def test_unknown_gloss_not_covered(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি বাড়ি যাচ্ছি।")
        # Empty lexicon → nothing covered
        for covered in result["lexicon_coverage"].values():
            assert covered is False

    def test_lexicon_miss_rule_recorded(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("আমি বাড়ি যাচ্ছি।")
        assert any("LexiconMiss" in r for r in result["applied_rules"])

    def test_load_from_master_lexicon(self, compiler_empty_lexicon):
        class FakeLexicon:
            signs_by_bn = {"আমি": {"label_bn": "আমি"}}

        compiler_empty_lexicon.load_from_master_lexicon(FakeLexicon())
        assert compiler_empty_lexicon._in_lexicon("আমি") is True

    def test_load_from_master_lexicon_missing_attr(self, compiler_empty_lexicon):
        """Should log a warning and not raise."""
        compiler_empty_lexicon.load_from_master_lexicon(object())  # no signs_by_bn


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string_returns_empty_glosses(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("")
        assert result["syntactic_glosses"] == []
        assert result["is_interrogative"] is False

    def test_whitespace_only_returns_empty(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("   ")
        assert result["syntactic_glosses"] == []

    def test_conjunction_only_returns_empty(self, compiler_empty_lexicon):
        result = compiler_empty_lexicon.compile_sentence("এবং ও কিন্তু")
        assert result["syntactic_glosses"] == []

    def test_natural_input_preserved(self, compiler_empty_lexicon):
        text = "আমি বাড়ি যাচ্ছি।"
        result = compiler_empty_lexicon.compile_sentence(text)
        assert result["natural_input"] == text


# ---------------------------------------------------------------------------
# 8. compile_batch()
# ---------------------------------------------------------------------------

class TestCompileBatch:

    def test_batch_count_matches(self, compiler_empty_lexicon):
        sentences = [
            "আমি বাড়ি যাচ্ছি।",
            "তুমি কোথায় যাচ্ছো?",
            "এখন আমি ভাত খাচ্ছি।",
        ]
        results = compiler_empty_lexicon.compile_batch(sentences)
        assert len(results) == 3

    def test_batch_each_has_glosses_key(self, compiler_empty_lexicon):
        results = compiler_empty_lexicon.compile_batch(["আমি যাচ্ছি।", "তুমি আসছো।"])
        for r in results:
            assert "syntactic_glosses" in r


# ---------------------------------------------------------------------------
# 9. to_sentence_plan() integration
# ---------------------------------------------------------------------------

class TestToSentencePlan:

    def test_returns_plan_type(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan("আমি বাড়ি যাচ্ছি।")
        assert isinstance(plan, CoarticulatedSentencePlan)

    def test_gloss_sequence_populated(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan("আমি বাড়ি যাচ্ছি।")
        assert len(plan.gloss_sequence) >= 1

    def test_transitions_count(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan("আমি বাড়ি যাচ্ছি।")
        assert len(plan.transitions) == len(plan.gloss_sequence) - 1

    def test_nmm_timeline_injected(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan("তুমি কোথায় যাচ্ছো?")
        assert len(plan.nmm_timeline) >= 1
        # Question: brow raise should be present in FACS
        facs = plan.nmm_timeline[0].facs
        assert facs.get("AU01_02_brow_raise", 0.0) == pytest.approx(0.85)

    def test_is_interrogative_forwarded(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan("তুমি কোথায় যাচ্ছো?")
        assert plan.is_interrogative is True

    def test_custom_template_id(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan(
            "আমি যাচ্ছি।", template_id="MY_TMPL_001"
        )
        assert plan.template_id == "MY_TMPL_001"

    def test_total_duration_positive(self, compiler_empty_lexicon):
        plan = compiler_empty_lexicon.to_sentence_plan("আমি বাড়ি যাচ্ছি।")
        assert plan.total_duration_ms > 0


# ---------------------------------------------------------------------------
# 10. SentenceToGlossPipeline(compiler=...) routing
# ---------------------------------------------------------------------------

class TestPipelineCompilerRouting:

    def test_compiler_route_used(self, compiler_empty_lexicon):
        """When compiler= is set, pipeline should use its syntactic_glosses."""
        pipeline = SentenceToGlossPipeline(compiler=compiler_empty_lexicon)
        plan = pipeline.process("আমি বাড়ি যাচ্ছি।")
        assert isinstance(plan, CoarticulatedSentencePlan)
        # Verb stem should be normalised by the compiler
        assert "যাওয়া" in plan.gloss_sequence

    def test_compiler_sov_ordering_via_pipeline(self, compiler_empty_lexicon):
        pipeline = SentenceToGlossPipeline(compiler=compiler_empty_lexicon)
        plan = pipeline.process("এখন আমি ভাত খাচ্ছি।")
        glosses = plan.gloss_sequence
        # Time-first ordering guarantee
        assert glosses[0] == "এখন"

    def test_no_compiler_uses_syntax_engine(self):
        """Without compiler=, pipeline falls back to BdSLSyntaxEngine."""
        pipeline = SentenceToGlossPipeline()
        plan = pipeline.process("আমি বাড়ি যাচ্ছি।")
        assert isinstance(plan, CoarticulatedSentencePlan)

    def test_pipeline_interrogative_forwarded(self, compiler_empty_lexicon):
        pipeline = SentenceToGlossPipeline(compiler=compiler_empty_lexicon)
        plan = pipeline.process("তুমি কোথায় যাচ্ছো?")
        assert plan.is_interrogative is True
