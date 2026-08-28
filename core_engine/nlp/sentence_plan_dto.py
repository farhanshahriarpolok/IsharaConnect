"""Shared DTOs for the BdSL sentence-level kinematic pipeline.

These dataclasses are kept in a leaf module (no upstream imports from
core_engine.nlp or core_engine.dsl) to prevent circular imports between
the sentence_to_gloss_pipeline and coarticulated_sentence_synthesizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core_engine.dsl.isharabakya_schema import NMMExpressionSegment


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
        """Return merged FACS AU values from all segments active at *timestamp_ms*."""
        merged: Dict[str, float] = {}
        for seg in self.nmm_timeline:
            if seg.applies_at(timestamp_ms):
                merged.update(seg.facs)
        return merged

    def get_transition(
        self, from_gloss: str, to_gloss: str
    ) -> Optional[GlossTransitionSpec]:
        for t in self.transitions:
            if t.from_gloss == from_gloss and t.to_gloss == to_gloss:
                return t
        return None
