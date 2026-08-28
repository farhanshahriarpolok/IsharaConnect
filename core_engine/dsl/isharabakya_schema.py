"""IsharaBakya-Generative-BdSL-v1 Corpus Schema.

Pydantic data models for the sentence-level annotation format that powers
continuous BdSL synthesis from spoken Bangla sentence templates.

Schema layers:
  - CoarticulationTransition  : Kinetic blend parameters between consecutive gloss tokens
  - KineticCoarticulationMap  : Full inter-sign transition sequence with sentence duration
  - NMMExpressionSegment      : FACS AU overlay for a timestamp window
  - GrammaticalDecomposition  : Linguistic structure of the sentence
  - SentenceBlueprint         : Full annotated sentence template
  - IsharaBakyaCorpusMetadata : Corpus-level header
  - IsharaBakyaCorpus         : Top-level corpus container
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field



# ---------------------------------------------------------------------------
# Kinetic Coarticulation Layer
# ---------------------------------------------------------------------------

class CoarticulationTransition(BaseModel):
    """Bézier-blended transition parameters between two consecutive gloss tokens."""

    model_config = ConfigDict(populate_by_name=True)

    from_gloss: str = Field(..., alias="from", description="Source gloss token")
    to_gloss: str = Field(..., alias="to", description="Target gloss token")
    blend_ms: int = Field(150, ge=0, description="Cross-fade / co-articulation blend window (ms)")
    slerp_enabled: bool = Field(True, description="Use Slerp quaternion interpolation for orientation")
    spatial_pause_ms: int = Field(0, ge=0, description="Rest hold inserted between signs (ms)")



class KineticCoarticulationMap(BaseModel):
    """Complete coarticulation schedule for a sentence."""

    transition_points: List[CoarticulationTransition] = Field(
        default_factory=list,
        description="Ordered list of inter-sign transition descriptors",
    )
    total_sentence_duration_ms: int = Field(
        1000, ge=0, description="Total planned playback duration for the sentence (ms)"
    )

    def get_blend_ms(self, from_gloss: str, to_gloss: str) -> int:
        """Return the co-articulation blend window (ms) between two adjacent glosses."""
        for tp in self.transition_points:
            if tp.from_gloss == from_gloss and tp.to_gloss == to_gloss:
                return tp.blend_ms
        return 150  # corpus-wide default

    def get_pause_ms(self, from_gloss: str, to_gloss: str) -> int:
        """Return the spatial hold/pause duration (ms) between two adjacent glosses."""
        for tp in self.transition_points:
            if tp.from_gloss == from_gloss and tp.to_gloss == to_gloss:
                return tp.spatial_pause_ms
        return 0

    def is_slerp_enabled(self, from_gloss: str, to_gloss: str) -> bool:
        """Return whether Slerp orientation blending is enabled for a transition pair."""
        for tp in self.transition_points:
            if tp.from_gloss == from_gloss and tp.to_gloss == to_gloss:
                return tp.slerp_enabled
        return True


# ---------------------------------------------------------------------------
# NMM Expression Timeline Layer
# ---------------------------------------------------------------------------

class NMMExpressionSegment(BaseModel):
    """FACS action unit overrides active during a [start_ms, end_ms] timestamp window."""

    timestamp_range: Tuple[int, int] = Field(
        ..., description="[start_ms, end_ms] window during which these FACS values apply"
    )
    facs: Dict[str, float] = Field(
        default_factory=dict,
        description="AU key-value pairs (e.g. AU12: 0.4, head_pitch: -3.0)",
    )

    @property
    def start_ms(self) -> int:
        return self.timestamp_range[0]

    @property
    def end_ms(self) -> int:
        return self.timestamp_range[1]

    def get_au(self, key: str, default: float = 0.0) -> float:
        """Return the value of a specific AU key within this segment."""
        return float(self.facs.get(key, default))

    def applies_at(self, timestamp_ms: int) -> bool:
        """Return True if this segment is active at the given timestamp (half-open interval)."""
        return self.start_ms <= timestamp_ms < self.end_ms


# ---------------------------------------------------------------------------
# Grammatical Decomposition Layer
# ---------------------------------------------------------------------------

class GrammaticalDecomposition(BaseModel):
    """Linguistic role assignment for each element in the sentence."""

    temporal_anchor: Optional[str] = None
    subject: Optional[str] = None
    locative_predicate: Optional[str] = None
    primary_verb_root: Optional[str] = None
    duration_condition: Optional[str] = None
    compound_result_verb: Optional[str] = None
    object_np: Optional[str] = None
    manner_adverb: Optional[str] = None
    negation_marker: Optional[str] = None
    wh_focus: Optional[str] = None
    extra: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sentence Blueprint
# ---------------------------------------------------------------------------

class SentenceBlueprint(BaseModel):
    """Complete annotated BdSL sentence template."""

    template_id: str = Field(
        ..., description="Unique template identifier (e.g. SEN_TMPL_DAILY_0042)"
    )
    domain: str = Field("General", description="Thematic domain of the sentence")
    spoken_bangla_variants: List[str] = Field(
        ..., min_length=1,
        description="One or more spoken Bengali surface realisations of the same meaning",
    )
    syntactic_gloss_sequence: List[str] = Field(
        ..., min_length=1,
        description="BdSL SOV-ordered gloss token sequence",
    )
    grammatical_decomposition: GrammaticalDecomposition = Field(
        default_factory=GrammaticalDecomposition
    )
    kinetic_coarticulation_map: KineticCoarticulationMap = Field(
        default_factory=KineticCoarticulationMap
    )
    nmm_expression_timeline: List[NMMExpressionSegment] = Field(
        default_factory=list,
        description="Ordered FACS expression windows covering the sentence duration",
    )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_nmm_at(self, timestamp_ms: int) -> Dict[str, float]:
        """Return merged FACS AU values from all segments active at *timestamp_ms*."""
        merged: Dict[str, float] = {}
        for seg in self.nmm_expression_timeline:
            if seg.applies_at(timestamp_ms):
                merged.update(seg.facs)
        return merged

    def iter_gloss_pairs(self):
        """Yield consecutive (from_gloss, to_gloss) pairs for coarticulation lookups."""
        glosses = self.syntactic_gloss_sequence
        for i in range(len(glosses) - 1):
            yield glosses[i], glosses[i + 1]

    @property
    def gloss_count(self) -> int:
        return len(self.syntactic_gloss_sequence)

    @property
    def total_duration_ms(self) -> int:
        return self.kinetic_coarticulation_map.total_sentence_duration_ms


# ---------------------------------------------------------------------------
# Corpus Container
# ---------------------------------------------------------------------------

class IsharaBakyaCorpusMetadata(BaseModel):
    """Top-level corpus metadata header."""

    corpus_name: str = "IsharaBakya-Generative-BdSL-v1"
    target_style: str = "Modern_Conversational_Bengali"
    grammatical_framework: str = "Bangla_Academy_Syntax_Aligned"
    total_sentence_templates: int = 5000
    generative_capacity: str = "50,000+ Unique Continuous Sentences"
    schema_version: str = "1.0.0"
    created_at: Optional[str] = None


class IsharaBakyaCorpus(BaseModel):
    """Root container for the IsharaBakya Generative Corpus."""

    corpus_metadata: IsharaBakyaCorpusMetadata = Field(
        default_factory=IsharaBakyaCorpusMetadata
    )
    sentence_blueprints: List[SentenceBlueprint] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: Any) -> "IsharaBakyaCorpus":
        """Load and validate a corpus JSON file (accepts str or Path)."""
        import json
        from pathlib import Path
        p = Path(path)
        with open(p, "r", encoding="utf-8-sig") as fh:
            raw = json.load(fh)
        return cls.model_validate(raw)

    def get_blueprint(self, template_id: str) -> Optional[SentenceBlueprint]:
        """Return the blueprint with the given template_id, or None."""
        for bp in self.sentence_blueprints:
            if bp.template_id == template_id:
                return bp
        return None

    def blueprints_by_domain(self, domain: str) -> List[SentenceBlueprint]:
        """Return all blueprints in the given thematic domain."""
        return [bp for bp in self.sentence_blueprints if bp.domain == domain]

    @property
    def blueprint_count(self) -> int:
        return len(self.sentence_blueprints)
