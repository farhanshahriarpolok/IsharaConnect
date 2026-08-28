"""Spoken Bangla → BdSL Gloss → CoarticulatedSentencePlan Pipeline.

This module bridges natural spoken Bangla text to a fully annotated
CoarticulatedSentencePlan that the CoarticulatedSentenceSynthesizer
can render into a continuous kinematic frame stream.

Pipeline stages:
  1. BdSLSyntaxEngine.text_to_bdsl_gloss()   — SOV gloss extraction
  2. IsharaBakyaCorpus template lookup         — retrieve coarticulation map & NMM timeline
  3. CoarticulatedSentencePlan assembly        — merge gloss + kinetic + expression data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core_engine.nlp.bdsl_syntax_engine import BdSLSyntaxEngine
from core_engine.dsl.isharabakya_schema import (
    IsharaBakyaCorpus,
    KineticCoarticulationMap,
    NMMExpressionSegment,
    SentenceBlueprint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Object: CoarticulatedSentencePlan
# ---------------------------------------------------------------------------

@dataclass
class GlossTransitionSpec:
    """Per-transition metadata consumed by the synthesizer."""
    from_gloss: str
    to_gloss: str
    blend_ms: int = 150
    slerp_enabled: bool = True
    spatial_pause_ms: int = 0


@dataclass
class CoarticulatedSentencePlan:
    """Complete execution plan for one BdSL sentence ready for kinematic synthesis."""

    template_id: str
    spoken_text: str
    gloss_sequence: List[str]
    transitions: List[GlossTransitionSpec]
    nmm_timeline: List[NMMExpressionSegment]
    total_duration_ms: int
    domain: str = "General"
    applied_rules: List[str] = field(default_factory=list)
    is_interrogative: bool = False

    # ------------------------------------------------------------------
    # Convenience query helpers
    # ------------------------------------------------------------------

    def get_nmm_at(self, timestamp_ms: int) -> Dict[str, float]:
        """Return merged FACS AU values active at *timestamp_ms*."""
        merged: Dict[str, float] = {}
        for seg in self.nmm_timeline:
            if seg.applies_at(timestamp_ms):
                merged.update(seg.facs)
        return merged

    def get_transition(self, from_gloss: str, to_gloss: str) -> Optional[GlossTransitionSpec]:
        for t in self.transitions:
            if t.from_gloss == from_gloss and t.to_gloss == to_gloss:
                return t
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SentenceToGlossPipeline:
    """Converts spoken Bangla sentences into CoarticulatedSentencePlan objects.

    Usage (corpus-backed, preferred):
        corpus = IsharaBakyaCorpus.from_json_file("dataset/corpus/isharabakya_seed_v1.json")
        pipeline = SentenceToGlossPipeline(corpus=corpus)
        plan = pipeline.process("আমি এখন একটু বাইরে যাচ্ছি, আধ ঘণ্টার মধ্যে ফিরবো।")

    Usage (syntax-only, no corpus needed):
        pipeline = SentenceToGlossPipeline()
        plan = pipeline.process("আমি বাড়ি যাচ্ছি।")
    """

    # Per-gloss base stroke duration used when no corpus plan is available
    _DEFAULT_STROKE_MS: int = 550
    # Default NMM FACS applied when no corpus timeline is found
    _NEUTRAL_FACS: Dict[str, float] = {
        "AU12": 0.0, "AU06": 0.0, "AU00": 0.0, "head_pitch": 0.0
    }

    def __init__(
        self,
        corpus: Optional[IsharaBakyaCorpus] = None,
        default_blend_ms: int = 150,
        fps: int = 60,
    ) -> None:
        self.syntax_engine = BdSLSyntaxEngine()
        self.corpus = corpus
        self.default_blend_ms = default_blend_ms
        self.fps = fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        spoken_text: str,
        template_id: Optional[str] = None,
    ) -> CoarticulatedSentencePlan:
        """Transform *spoken_text* into a CoarticulatedSentencePlan.

        If *template_id* is provided and exists in the attached corpus, the
        corpus coarticulation map and NMM timeline override any computed values.
        Otherwise the plan is built from the BdSLSyntaxEngine output with
        sensible defaults.
        """
        # Step 1: Extract BdSL gloss sequence from surface Bengali text
        gloss_result = self.syntax_engine.text_to_bdsl_gloss(spoken_text)
        glosses: List[str] = gloss_result.get("glosses", [])
        applied_rules: List[str] = gloss_result.get("applied_rules", [])
        is_interrogative: bool = gloss_result.get("is_interrogative", False)

        if not glosses:
            logger.warning("SentenceToGlossPipeline: empty gloss sequence for '%s'", spoken_text)

        # Step 2: Attempt corpus lookup
        blueprint: Optional[SentenceBlueprint] = None
        if template_id and self.corpus:
            blueprint = self.corpus.get_blueprint(template_id)
            if blueprint:
                # Use the corpus gloss sequence (authoritative) when available
                glosses = blueprint.syntactic_gloss_sequence
                logger.debug("Blueprint '%s' resolved from corpus.", template_id)

        if blueprint is None and self.corpus:
            blueprint = self._match_by_text(spoken_text)

        # Step 3: Build transition specs
        transitions = self._build_transitions(glosses, blueprint)

        # Step 4: Build NMM timeline
        nmm_timeline = self._build_nmm_timeline(glosses, blueprint)

        # Step 5: Compute total duration
        total_ms = self._compute_duration(glosses, transitions, blueprint)

        return CoarticulatedSentencePlan(
            template_id=template_id or (blueprint.template_id if blueprint else "DYNAMIC"),
            spoken_text=spoken_text,
            gloss_sequence=glosses,
            transitions=transitions,
            nmm_timeline=nmm_timeline,
            total_duration_ms=total_ms,
            domain=blueprint.domain if blueprint else "General",
            applied_rules=applied_rules,
            is_interrogative=is_interrogative,
        )

    def process_blueprint(self, blueprint: SentenceBlueprint) -> CoarticulatedSentencePlan:
        """Directly convert a SentenceBlueprint into a CoarticulatedSentencePlan."""
        spoken_text = blueprint.spoken_bangla_variants[0] if blueprint.spoken_bangla_variants else ""
        glosses = blueprint.syntactic_gloss_sequence
        transitions = self._build_transitions(glosses, blueprint)
        nmm_timeline = blueprint.nmm_expression_timeline
        total_ms = blueprint.total_duration_ms

        return CoarticulatedSentencePlan(
            template_id=blueprint.template_id,
            spoken_text=spoken_text,
            gloss_sequence=glosses,
            transitions=transitions,
            nmm_timeline=nmm_timeline,
            total_duration_ms=total_ms,
            domain=blueprint.domain,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_by_text(self, spoken_text: str) -> Optional[SentenceBlueprint]:
        """Best-effort fuzzy match against corpus variants (exact substring match)."""
        if not self.corpus:
            return None
        for bp in self.corpus.sentence_blueprints:
            for variant in bp.spoken_bangla_variants:
                if spoken_text.strip() in variant or variant in spoken_text.strip():
                    return bp
        return None

    def _build_transitions(
        self,
        glosses: List[str],
        blueprint: Optional[SentenceBlueprint],
    ) -> List[GlossTransitionSpec]:
        transitions: List[GlossTransitionSpec] = []
        coart_map: Optional[KineticCoarticulationMap] = (
            blueprint.kinetic_coarticulation_map if blueprint else None
        )
        for i in range(len(glosses) - 1):
            fg, tg = glosses[i], glosses[i + 1]
            blend_ms = (
                coart_map.get_blend_ms(fg, tg) if coart_map else self.default_blend_ms
            )
            slerp = coart_map.is_slerp_enabled(fg, tg) if coart_map else True
            pause_ms = coart_map.get_pause_ms(fg, tg) if coart_map else 0
            transitions.append(
                GlossTransitionSpec(
                    from_gloss=fg,
                    to_gloss=tg,
                    blend_ms=blend_ms,
                    slerp_enabled=slerp,
                    spatial_pause_ms=pause_ms,
                )
            )
        return transitions

    def _build_nmm_timeline(
        self,
        glosses: List[str],
        blueprint: Optional[SentenceBlueprint],
    ) -> List[NMMExpressionSegment]:
        if blueprint and blueprint.nmm_expression_timeline:
            return blueprint.nmm_expression_timeline
        # Default: neutral expression for the entire sentence
        estimated_ms = len(glosses) * self._DEFAULT_STROKE_MS
        return [
            NMMExpressionSegment(
                timestamp_range=(0, max(estimated_ms, 1)),
                facs=self._NEUTRAL_FACS.copy(),
            )
        ]

    def _compute_duration(
        self,
        glosses: List[str],
        transitions: List[GlossTransitionSpec],
        blueprint: Optional[SentenceBlueprint],
    ) -> int:
        if blueprint:
            return blueprint.total_duration_ms
        # Estimate: per-gloss stroke + blend windows + spatial pauses
        stroke_total = len(glosses) * self._DEFAULT_STROKE_MS
        blend_total = sum(t.blend_ms for t in transitions)
        pause_total = sum(t.spatial_pause_ms for t in transitions)
        return stroke_total + blend_total + pause_total
