"""BdSL Domain Specific Language (DSL) and Landmark Agent Tools."""

from core_engine.dsl.bdsl_tools import (
    load_bdsl_dictionary,
    get_sign_dsl_tool,
    get_sign_landmarks_tool,
)
from core_engine.dsl.agent_pipeline import (
    Agent,
    Task,
    ishara_agent,
    run_pipeline,
)
from core_engine.dsl.bdsl_v3_spec import (
    BdSLV3SignSpec,
    BdSLPhonetics,
    BdSLKinematics,
    BdSLFacialActionUnits,
    BdSLContactPhysics,
    BdSLTemporalPhases,
    BdSLMorphosyntax,
)
from core_engine.dsl.procedural_synthesizer import (
    HyperKinematicSynthesizer,
)
from core_engine.dsl.sequence_blender import MultiSignSequenceBlender
from core_engine.dsl.video_to_v3_extractor import BdSLVideoV3Extractor
from core_engine.dsl.isharabakya_schema import (
    IsharaBakyaCorpus,
    IsharaBakyaCorpusMetadata,
    SentenceBlueprint,
    KineticCoarticulationMap,
    CoarticulationTransition,
    NMMExpressionSegment,
    GrammaticalDecomposition,
)
# CoarticulatedSentenceSynthesizer is NOT imported here to avoid a
# circular import chain:  dsl/__init__ -> coarticulated_sentence_synthesizer
# -> sentence_plan_dto -> dsl/isharabakya_schema -> dsl/__init__
# Import it directly:  from core_engine.dsl.coarticulated_sentence_synthesizer import ...


__all__ = [
    "load_bdsl_dictionary",
    "get_sign_dsl_tool",
    "get_sign_landmarks_tool",
    "Agent",
    "Task",
    "ishara_agent",
    "run_pipeline",
    "BdSLV3SignSpec",
    "BdSLPhonetics",
    "BdSLKinematics",
    "BdSLFacialActionUnits",
    "BdSLContactPhysics",
    "BdSLTemporalPhases",
    "BdSLMorphosyntax",
    "HyperKinematicSynthesizer",
    "MultiSignSequenceBlender",
    "BdSLVideoV3Extractor",
    # IsharaBakya Corpus Schema
    "IsharaBakyaCorpus",
    "IsharaBakyaCorpusMetadata",
    "SentenceBlueprint",
    "KineticCoarticulationMap",
    "CoarticulationTransition",
    "NMMExpressionSegment",
    "GrammaticalDecomposition",
    # CoarticulatedSentenceSynthesizer: import directly from its module to avoid
    # circular imports — from core_engine.dsl.coarticulated_sentence_synthesizer import ...
]
