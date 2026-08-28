"""Unit tests for the IsharaBakya-Generative-BdSL-v1 corpus pipeline.

Covers:
  - IsharaBakyaCorpus Pydantic schema validation
  - CoarticulationTransition alias parsing (from/to field)
  - KineticCoarticulationMap blend/pause/slerp lookups
  - NMMExpressionSegment timestamp range queries
  - SentenceBlueprint.get_nmm_at() merged FACS resolution
  - SentenceToGlossPipeline: syntax-only (no corpus) plan generation
  - SentenceToGlossPipeline: corpus-backed plan generation
  - SentenceToGlossPipeline.process_blueprint()
  - CoarticulatedSentenceSynthesizer: frame stream shape & metadata
  - CoarticulatedSentenceSynthesizer: NMM injection correctness
  - Seed corpus JSON round-trip (load → validate → query)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from core_engine.dsl.isharabakya_schema import (
    CoarticulationTransition,
    IsharaBakyaCorpus,
    IsharaBakyaCorpusMetadata,
    KineticCoarticulationMap,
    NMMExpressionSegment,
    SentenceBlueprint,
    GrammaticalDecomposition,
)
from core_engine.nlp.sentence_to_gloss_pipeline import (
    CoarticulatedSentencePlan,
    GlossTransitionSpec,
    SentenceToGlossPipeline,
)
from core_engine.dsl.coarticulated_sentence_synthesizer import (
    CoarticulatedSentenceSynthesizer,
    _make_stub_sign_spec,
    _generate_pause_frames,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "dataset" / "corpus" / "isharabakya_seed_v1.json"
)

SAMPLE_BLUEPRINT_DATA = {
    "template_id": "SEN_TMPL_TEST_0001",
    "domain": "Test Domain",
    "spoken_bangla_variants": ["আমি বাড়ি যাচ্ছি।"],
    "syntactic_gloss_sequence": ["আমি", "বাড়ি", "যাওয়া"],
    "grammatical_decomposition": {
        "subject": "আমি",
        "locative_predicate": "বাড়ি",
        "primary_verb_root": "যাওয়া",
    },
    "kinetic_coarticulation_map": {
        "transition_points": [
            {"from": "আমি",  "to": "বাড়ি",   "blend_ms": 120, "slerp_enabled": True,  "spatial_pause_ms": 0},
            {"from": "বাড়ি", "to": "যাওয়া", "blend_ms": 160, "slerp_enabled": False, "spatial_pause_ms": 60},
        ],
        "total_sentence_duration_ms": 1800,
    },
    "nmm_expression_timeline": [
        {"timestamp_range": [0, 900],  "facs": {"AU12": 0.1, "head_pitch": 0.0}},
        {"timestamp_range": [900, 1800], "facs": {"AU12": 0.4, "AU06": 0.3, "head_pitch": -2.0}},
    ],
}


@pytest.fixture
def sample_blueprint() -> SentenceBlueprint:
    return SentenceBlueprint.model_validate(SAMPLE_BLUEPRINT_DATA)


@pytest.fixture
def sample_corpus(sample_blueprint: SentenceBlueprint) -> IsharaBakyaCorpus:
    return IsharaBakyaCorpus(
        corpus_metadata=IsharaBakyaCorpusMetadata(),
        sentence_blueprints=[sample_blueprint],
    )


@pytest.fixture
def pipeline_no_corpus() -> SentenceToGlossPipeline:
    return SentenceToGlossPipeline()


@pytest.fixture
def pipeline_with_corpus(sample_corpus: IsharaBakyaCorpus) -> SentenceToGlossPipeline:
    return SentenceToGlossPipeline(corpus=sample_corpus)


@pytest.fixture
def synthesizer() -> CoarticulatedSentenceSynthesizer:
    return CoarticulatedSentenceSynthesizer(fps=60)


# ===========================================================================
# 1. Schema: CoarticulationTransition alias parsing
# ===========================================================================

class TestCoarticulationTransition:

    def test_from_to_alias_parsing(self):
        """'from' and 'to' JSON keys must map to from_gloss / to_gloss fields."""
        t = CoarticulationTransition.model_validate(
            {"from": "আমি", "to": "বাড়ি", "blend_ms": 130}
        )
        assert t.from_gloss == "আমি"
        assert t.to_gloss == "বাড়ি"
        assert t.blend_ms == 130

    def test_defaults(self):
        t = CoarticulationTransition.model_validate({"from": "A", "to": "B"})
        assert t.blend_ms == 150
        assert t.slerp_enabled is True
        assert t.spatial_pause_ms == 0


# ===========================================================================
# 2. Schema: KineticCoarticulationMap queries
# ===========================================================================

class TestKineticCoarticulationMap:

    @pytest.fixture
    def coart_map(self) -> KineticCoarticulationMap:
        return KineticCoarticulationMap.model_validate(
            SAMPLE_BLUEPRINT_DATA["kinetic_coarticulation_map"]
        )

    def test_get_blend_ms_found(self, coart_map):
        assert coart_map.get_blend_ms("আমি", "বাড়ি") == 120

    def test_get_blend_ms_default(self, coart_map):
        assert coart_map.get_blend_ms("X", "Y") == 150

    def test_get_pause_ms_found(self, coart_map):
        assert coart_map.get_pause_ms("বাড়ি", "যাওয়া") == 60

    def test_get_pause_ms_default(self, coart_map):
        assert coart_map.get_pause_ms("X", "Y") == 0

    def test_slerp_enabled_false(self, coart_map):
        assert coart_map.is_slerp_enabled("বাড়ি", "যাওয়া") is False

    def test_slerp_enabled_true(self, coart_map):
        assert coart_map.is_slerp_enabled("আমি", "বাড়ি") is True

    def test_slerp_default_true(self, coart_map):
        assert coart_map.is_slerp_enabled("unknown", "unknown") is True


# ===========================================================================
# 3. Schema: NMMExpressionSegment
# ===========================================================================

class TestNMMExpressionSegment:

    @pytest.fixture
    def seg(self) -> NMMExpressionSegment:
        return NMMExpressionSegment(
            timestamp_range=(500, 1500),
            facs={"AU12": 0.4, "head_pitch": -2.0}
        )

    def test_applies_at_inside(self, seg):
        assert seg.applies_at(500) is True
        assert seg.applies_at(1000) is True
        assert seg.applies_at(1499) is True

    def test_applies_at_outside(self, seg):
        assert seg.applies_at(499) is False
        assert seg.applies_at(1500) is False  # half-open [500, 1500)

    def test_get_au(self, seg):
        assert seg.get_au("AU12") == pytest.approx(0.4)
        assert seg.get_au("AU06", default=0.99) == pytest.approx(0.99)

    def test_start_end_properties(self, seg):
        assert seg.start_ms == 500
        assert seg.end_ms == 1500


# ===========================================================================
# 4. Schema: SentenceBlueprint helpers
# ===========================================================================

class TestSentenceBlueprint:

    def test_gloss_count(self, sample_blueprint):
        assert sample_blueprint.gloss_count == 3

    def test_total_duration_ms(self, sample_blueprint):
        assert sample_blueprint.total_duration_ms == 1800

    def test_get_nmm_at_first_window(self, sample_blueprint):
        nmm = sample_blueprint.get_nmm_at(400)
        assert nmm.get("AU12") == pytest.approx(0.1)

    def test_get_nmm_at_second_window(self, sample_blueprint):
        nmm = sample_blueprint.get_nmm_at(1200)
        assert nmm.get("AU06") == pytest.approx(0.3)
        assert nmm.get("head_pitch") == pytest.approx(-2.0)

    def test_get_nmm_at_boundary(self, sample_blueprint):
        # At exactly 900ms the second segment begins
        nmm = sample_blueprint.get_nmm_at(900)
        assert nmm.get("AU12") == pytest.approx(0.4)

    def test_iter_gloss_pairs(self, sample_blueprint):
        pairs = list(sample_blueprint.iter_gloss_pairs())
        assert pairs == [("আমি", "বাড়ি"), ("বাড়ি", "যাওয়া")]


# ===========================================================================
# 5. Corpus: load and query
# ===========================================================================

class TestIsharaBakyaCorpus:

    def test_blueprint_count(self, sample_corpus):
        assert sample_corpus.blueprint_count == 1

    def test_get_blueprint_found(self, sample_corpus):
        bp = sample_corpus.get_blueprint("SEN_TMPL_TEST_0001")
        assert bp is not None
        assert bp.domain == "Test Domain"

    def test_get_blueprint_not_found(self, sample_corpus):
        assert sample_corpus.get_blueprint("DOES_NOT_EXIST") is None

    def test_blueprints_by_domain(self, sample_corpus):
        results = sample_corpus.blueprints_by_domain("Test Domain")
        assert len(results) == 1

    def test_blueprints_by_domain_empty(self, sample_corpus):
        results = sample_corpus.blueprints_by_domain("Unknown Domain")
        assert results == []


# ===========================================================================
# 6. Seed corpus JSON: round-trip load
# ===========================================================================

class TestSeedCorpusJSON:

    @pytest.mark.skipif(not SEED_CORPUS_PATH.exists(), reason="Seed corpus file not found")
    def test_load_and_validate(self):
        corpus = IsharaBakyaCorpus.from_json_file(SEED_CORPUS_PATH)
        assert corpus.blueprint_count >= 1
        assert corpus.corpus_metadata.corpus_name == "IsharaBakya-Generative-BdSL-v1"

    @pytest.mark.skipif(not SEED_CORPUS_PATH.exists(), reason="Seed corpus file not found")
    def test_seed_blueprint_0042(self):
        corpus = IsharaBakyaCorpus.from_json_file(SEED_CORPUS_PATH)
        bp = corpus.get_blueprint("SEN_TMPL_DAILY_0042")
        assert bp is not None
        assert "যাওয়া" in bp.syntactic_gloss_sequence
        assert bp.total_duration_ms == 4650

    @pytest.mark.skipif(not SEED_CORPUS_PATH.exists(), reason="Seed corpus file not found")
    def test_seed_coarticulation_transitions(self):
        corpus = IsharaBakyaCorpus.from_json_file(SEED_CORPUS_PATH)
        bp = corpus.get_blueprint("SEN_TMPL_DAILY_0042")
        assert bp is not None
        coart = bp.kinetic_coarticulation_map
        # Spatial pause between যাওয়া → আধ-ঘণ্টা must be 80ms
        assert coart.get_pause_ms("যাওয়া", "আধ-ঘণ্টা") == 80
        # Slerp must be disabled for that transition
        assert coart.is_slerp_enabled("যাওয়া", "আধ-ঘণ্টা") is False


# ===========================================================================
# 7. SentenceToGlossPipeline: syntax-only mode
# ===========================================================================

class TestPipelineSyntaxOnly:

    def test_returns_plan(self, pipeline_no_corpus):
        plan = pipeline_no_corpus.process("আমি বাড়ি যাচ্ছি।")
        assert isinstance(plan, CoarticulatedSentencePlan)

    def test_gloss_sequence_non_empty(self, pipeline_no_corpus):
        plan = pipeline_no_corpus.process("আমি বাড়ি যাচ্ছি।")
        assert len(plan.gloss_sequence) >= 1

    def test_transitions_count(self, pipeline_no_corpus):
        plan = pipeline_no_corpus.process("আমি বাড়ি যাচ্ছি।")
        assert len(plan.transitions) == len(plan.gloss_sequence) - 1

    def test_nmm_timeline_non_empty(self, pipeline_no_corpus):
        plan = pipeline_no_corpus.process("আমি বাড়ি যাচ্ছি।")
        assert len(plan.nmm_timeline) >= 1

    def test_total_duration_positive(self, pipeline_no_corpus):
        plan = pipeline_no_corpus.process("আমি বাড়ি যাচ্ছি।")
        assert plan.total_duration_ms > 0

    def test_empty_input_graceful(self, pipeline_no_corpus):
        plan = pipeline_no_corpus.process("")
        assert isinstance(plan, CoarticulatedSentencePlan)


# ===========================================================================
# 8. SentenceToGlossPipeline: corpus-backed mode
# ===========================================================================

class TestPipelineCorpusBacked:

    def test_corpus_blueprint_used(self, pipeline_with_corpus):
        plan = pipeline_with_corpus.process(
            "আমি বাড়ি যাচ্ছি।", template_id="SEN_TMPL_TEST_0001"
        )
        assert plan.template_id == "SEN_TMPL_TEST_0001"
        # Corpus gloss sequence is authoritative
        assert plan.gloss_sequence == ["আমি", "বাড়ি", "যাওয়া"]

    def test_coarticulation_from_corpus(self, pipeline_with_corpus):
        plan = pipeline_with_corpus.process(
            "আমি বাড়ি যাচ্ছি।", template_id="SEN_TMPL_TEST_0001"
        )
        t = plan.get_transition("আমি", "বাড়ি")
        assert t is not None
        assert t.blend_ms == 120

    def test_pause_from_corpus(self, pipeline_with_corpus):
        plan = pipeline_with_corpus.process(
            "আমি বাড়ি যাচ্ছি।", template_id="SEN_TMPL_TEST_0001"
        )
        t = plan.get_transition("বাড়ি", "যাওয়া")
        assert t is not None
        assert t.spatial_pause_ms == 60

    def test_nmm_timeline_from_corpus(self, pipeline_with_corpus):
        plan = pipeline_with_corpus.process(
            "আমি বাড়ি যাচ্ছি।", template_id="SEN_TMPL_TEST_0001"
        )
        # Second NMM window [900, 1800) should be active at t=1000
        nmm = plan.get_nmm_at(1000)
        assert nmm.get("AU06") == pytest.approx(0.3)

    def test_process_blueprint_directly(self, pipeline_with_corpus, sample_blueprint):
        plan = pipeline_with_corpus.process_blueprint(sample_blueprint)
        assert plan.gloss_sequence == sample_blueprint.syntactic_gloss_sequence
        assert plan.total_duration_ms == 1800


# ===========================================================================
# 9. CoarticulatedSentenceSynthesizer: frame stream
# ===========================================================================

class TestCoarticulatedSentenceSynthesizer:

    def _make_plan(self, glosses: List[str], duration_ms: int = 1800) -> CoarticulatedSentencePlan:
        transitions = [
            GlossTransitionSpec(from_gloss=glosses[i], to_gloss=glosses[i + 1], blend_ms=100)
            for i in range(len(glosses) - 1)
        ]
        nmm_timeline = [
            NMMExpressionSegment(
                timestamp_range=(0, duration_ms),
                facs={"AU12": 0.5, "head_pitch": -1.5},
            )
        ]
        return CoarticulatedSentencePlan(
            template_id="TEST",
            spoken_text="test",
            gloss_sequence=glosses,
            transitions=transitions,
            nmm_timeline=nmm_timeline,
            total_duration_ms=duration_ms,
        )

    def test_frame_stream_non_empty(self, synthesizer):
        plan = self._make_plan(["আমি", "বাড়ি", "যাওয়া"])
        frames = synthesizer.synthesize(plan)
        assert len(frames) > 0

    def test_frame_indices_sequential(self, synthesizer):
        plan = self._make_plan(["আমি", "বাড়ি"])
        frames = synthesizer.synthesize(plan)
        for i, f in enumerate(frames):
            assert f["frame_idx"] == i

    def test_frame_timestamps_non_decreasing(self, synthesizer):
        plan = self._make_plan(["আমি", "বাড়ি", "যাওয়া"])
        frames = synthesizer.synthesize(plan)
        ts = [f["timestamp_ms"] for f in frames]
        assert ts == sorted(ts)

    def test_right_wrist_shape(self, synthesizer):
        plan = self._make_plan(["আমি"])
        frames = synthesizer.synthesize(plan)
        for f in frames:
            assert len(f["right_wrist"]) == 3

    def test_right_hand_21_joints(self, synthesizer):
        plan = self._make_plan(["আমি"])
        frames = synthesizer.synthesize(plan)
        for f in frames:
            assert len(f["right_hand"]) == 21

    def test_nmm_injection_au12(self, synthesizer):
        plan = self._make_plan(["আমি", "বাড়ি"])
        frames = synthesizer.synthesize(plan)
        # All frames fall within the [0, 1800) NMM window — AU12=0.5 injected
        for f in frames:
            assert "AU12" in f["facs"]
            assert f["facs"]["AU12"] == pytest.approx(0.5)

    def test_empty_gloss_sequence(self, synthesizer):
        plan = CoarticulatedSentencePlan(
            template_id="EMPTY",
            spoken_text="",
            gloss_sequence=[],
            transitions=[],
            nmm_timeline=[],
            total_duration_ms=0,
        )
        frames = synthesizer.synthesize(plan)
        assert frames == []

    def test_spatial_pause_generates_pause_frames(self, synthesizer):
        """Transitions with spatial_pause_ms should produce frames with is_pause=True."""
        transitions = [
            GlossTransitionSpec(
                from_gloss="আমি", to_gloss="বাড়ি",
                blend_ms=100, spatial_pause_ms=200
            )
        ]
        plan = CoarticulatedSentencePlan(
            template_id="PAUSE_TEST",
            spoken_text="",
            gloss_sequence=["আমি", "বাড়ি"],
            transitions=transitions,
            nmm_timeline=[],
            total_duration_ms=2000,
        )
        frames = synthesizer.synthesize(plan)
        pause_frames = [f for f in frames if f.get("is_pause")]
        assert len(pause_frames) > 0


# ===========================================================================
# 10. Helpers
# ===========================================================================

class TestHelpers:

    def test_stub_sign_spec_keys(self):
        spec = _make_stub_sign_spec("তুমি")
        assert "phonetics" in spec
        assert "kinematics" in spec
        assert "temporal_phases_ms" in spec
        assert spec["gloss_bn"] == "তুমি"

    def test_generate_pause_frames_count(self):
        dummy_frame = {
            "right_wrist": [0.5, 0.5, 0.0],
            "right_hand": [[0.0, 0.0, 0.0]] * 21,
            "facs": {"AU12": 0.0},
            "is_transition": False,
        }
        frames = _generate_pause_frames(dummy_frame, pause_ms=300, fps=60)
        # 300ms @ 60fps = 18 frames
        assert len(frames) == 18
        assert all(f["is_pause"] for f in frames)
