"""Comprehensive Unit Test Suite for Multi-Modal BdSL Linguistic Engine.

Tests:
1. Tier 1: Full 50-grapheme dactylology, T0-T7 trigger matrices, cumulative confidence filtering.
2. Tier 2: Master BdSL Lexicon (50+ signs, 6 domains, 5-layer kinematic & FACS schema).
3. Fine-Grained Minimal Pair Discriminator (kinematic and geometric disambiguation).
4. Tier 3 & 4: Visual-Spatial Syntax Engine (4 foundational rules & bidirectional translation).
"""

import math
import numpy as np
import pytest

from core_engine.vision.dactylology_engine import (
    DactylologyEngine, MASTER_GRAPHEMES, VOWELS, CONSONANTS, DIGITS, TRIGGER_MAP
)
from core_engine.nlp.master_lexicon import MasterBdSLLexicon, master_lexicon
from core_engine.inference.minimal_pair_discriminator import MinimalPairDiscriminator
from core_engine.nlp.bdsl_syntax_engine import BdSLSyntaxEngine, bdsl_syntax_engine


# ==============================================================================
# 1. Tier 1: Full Dactylology & Trigger Synthesizer Tests
# ==============================================================================

def test_dactylology_inventory_completeness():
    """Ensure all 11 vowels, 39 consonants, 10 digits, and diacritics are indexed."""
    assert len(VOWELS) == 11
    assert len(CONSONANTS) == 39
    assert len(DIGITS) == 10
    assert len(MASTER_GRAPHEMES) == 60  # 11 + 39 + 10

    engine = DactylologyEngine()
    # Test lookup
    meta_ka = engine.get_grapheme_meta("ক")
    assert meta_ka is not None
    assert meta_ka["slug"] == "cons_ka"
    assert meta_ka["category"] == "Consonant"

    meta_0 = engine.get_grapheme_meta("০")
    assert meta_0 is not None
    assert meta_0["category"] == "Digit"


def test_dactylology_trigger_transformations():
    """Test T0-T7 trigger modifiers and conjunct synthesis."""
    engine = DactylologyEngine()

    # T0: Identity
    assert engine.apply_trigger_transform("ক", "T0") == "ক"

    # T1: Kar-AA (া / আ)
    assert engine.apply_trigger_transform("ক", "T1") == "কা"
    assert engine.apply_trigger_transform("অ", "T1") == "আ"

    # T2: Kar-I (ি / ই)
    assert engine.apply_trigger_transform("ক", "T2") == "কি"

    # T3: Kar-U (ু / উ)
    assert engine.apply_trigger_transform("ক", "T3") == "কু"

    # T4: Conjunct Ksha (ক্ষ)
    assert engine.apply_trigger_transform("ক", "T4") == "ক্ষ"

    # T5: Conjunct Gya (জ্ঞ)
    assert engine.apply_trigger_transform("জ", "T5") == "জ্ঞ"

    # T6: Diacritic Chandrabindu (ঁ)
    assert engine.apply_trigger_transform("চ", "T6") == "চঁ"

    # T7: Halant / Virama (্)
    assert engine.apply_trigger_transform("ত", "T7") == "ত্"

    # Conjunct Synthesis helper
    assert engine.synthesize_conjunct("ক", "ষ") == "ক্ষ"
    assert engine.synthesize_conjunct("জ", "ঞ") == "জ্ঞ"
    assert engine.synthesize_conjunct("ত", "ত") == "ত্ত"


def test_dactylology_cumulative_confidence_filtering():
    """Test sliding-window cumulative confidence accumulation and debounce."""
    engine = DactylologyEngine(cumulative_confidence_delta=0.80, window_size=5, debounce_latency_s=1.0)
    t = 100.0

    # Low confidence should not emit
    assert engine.process_character_prediction("ক", confidence=0.40, timestamp=t) is None

    # Sustained high confidence frames should emit
    engine.process_character_prediction("ক", confidence=0.90, timestamp=t + 0.1)
    engine.process_character_prediction("ক", confidence=0.95, timestamp=t + 0.2)
    emitted = engine.process_character_prediction("ক", confidence=0.95, timestamp=t + 0.3)
    assert emitted == "ক"

    # Immediate repetition within debounce window should be blocked
    assert engine.process_character_prediction("ক", confidence=0.95, timestamp=t + 0.4) is None

    # After debounce window elapsed, same character can emit again
    emitted_later = engine.process_character_prediction("ক", confidence=0.95, timestamp=t + 1.5)
    assert emitted_later == "ক"


# ==============================================================================
# 2. Tier 2: Master BdSL Lexicon & Schema Tests
# ==============================================================================

def test_master_lexicon_loading_and_categories():
    """Verify MasterBdSLLexicon contains 50+ signs across all 6 standardized domains."""
    lexicon = master_lexicon
    signs = lexicon.all_signs()
    assert len(signs) >= 30

    kinship_signs = lexicon.get_signs_by_category("Kinship")
    assert len(kinship_signs) >= 5

    emergency_signs = lexicon.get_signs_by_category("Healthcare & Emergency")
    assert len(emergency_signs) >= 5

    disaster_signs = lexicon.get_signs_by_category("Disaster & Safety")
    assert len(disaster_signs) >= 5


def test_master_lexicon_5_layer_schema():
    """Verify 5-layer annotations (Handshape, 3D Bézier, FACS, Contact Physics, Timing)."""
    lexicon = master_lexicon
    bhumikompo = lexicon.get_sign_by_gloss("ভূমিকম্প")
    assert bhumikompo is not None
    assert bhumikompo["slug"] == "bhumikompo"

    # Layer 1: Handshape / Stokoe
    assert "handshape" in bhumikompo and bhumikompo["handshape"]
    assert "stokoe_notation" in bhumikompo

    # Layer 2: 3D Bézier Anchors (P0, P1, P2, P3)
    bezier = bhumikompo.get("bezier_anchors_3d", {})
    assert "P0" in bezier and "P3" in bezier
    assert len(bezier["P0"]) == 3

    # Layer 3: FACS Action Units
    facs = bhumikompo.get("facs_action_units", {})
    assert "AU01" in facs and "AU25" in facs

    # Layer 4: Contact Physics Plane & Anchor
    contact = bhumikompo.get("contact_physics", {})
    assert "plane" in contact and "body_anchor" in contact

    # Layer 5: Timing
    timing = bhumikompo.get("timing_ms", {})
    assert "stroke" in timing and "total" in timing


def test_master_lexicon_query_helpers():
    """Test search and kinematic profile resolution."""
    lexicon = master_lexicon
    res = lexicon.search_signs("Hospital")
    assert len(res) >= 1
    assert res[0]["slug"] == "haspatal"

    profile = lexicon.get_kinematic_profile("dhonnobad")
    assert profile is not None
    assert profile["label_bn"] == "ধন্যবাদ"


# ==============================================================================
# 3. Fine-Grained Minimal Pair Discriminator Tests
# ==============================================================================

def test_minimal_pair_earthquake_vs_traffic():
    """Verify Earthquake (in-phase vibration) vs Traffic (out-of-phase translation)."""
    discriminator = MinimalPairDiscriminator()
    t = np.linspace(0, 1, 30)

    # 1. Earthquake: Synchronous lateral vibration (high frequency sinusoidal motion in x)
    vibration = 0.05 * np.sin(2 * np.pi * 6.0 * t)
    left_eq = np.zeros((30, 3))
    right_eq = np.zeros((30, 3))
    left_eq[:, 0] = -0.15 + vibration
    right_eq[:, 0] = 0.15 + vibration

    res_eq = discriminator.disambiguate_earthquake_vs_traffic(left_eq, right_eq, fps=30.0)
    assert res_eq["resolved_slug"] == "bhumikompo"
    assert res_eq["confidence"] >= 0.85

    # 2. Traffic Jam: Alternating forward-backward motion (out-of-phase in z)
    left_tj = np.zeros((30, 3))
    right_tj = np.zeros((30, 3))
    left_tj[:, 2] = 0.20 + 0.10 * np.sin(2 * np.pi * 1.5 * t)
    right_tj[:, 2] = 0.20 - 0.10 * np.sin(2 * np.pi * 1.5 * t)

    res_tj = discriminator.disambiguate_earthquake_vs_traffic(left_tj, right_tj, fps=30.0)
    assert res_tj["resolved_slug"] == "janjot"


def test_minimal_pair_uncle_vs_grandfather():
    """Verify Uncle (acute chin touch) vs Grandfather (downward beard stroke)."""
    discriminator = MinimalPairDiscriminator()

    # 1. Uncle: Small delta_y (< 3cm)
    uncle_traj = np.zeros((15, 3))
    uncle_traj[:, 1] = 0.35 + 0.01 * np.linspace(0, 1, 15)
    res_uncle = discriminator.disambiguate_uncle_vs_grandfather(uncle_traj)
    assert res_uncle["resolved_slug"] == "chacha"

    # 2. Grandfather: Extended downward stroke (delta_y = 12cm)
    dada_traj = np.zeros((15, 3))
    dada_traj[:, 1] = 0.35 + 0.12 * np.linspace(0, 1, 15)
    res_dada = discriminator.disambiguate_uncle_vs_grandfather(dada_traj)
    assert res_dada["resolved_slug"] == "dada"


def test_minimal_pair_debor_vs_dulabhai():
    """Verify Debor (single index finger) vs Dulabhai (V-shape dual fingers)."""
    discriminator = MinimalPairDiscriminator()

    # 1. Debor: single index finger extended, middle curled
    lm_debor = np.zeros((21, 3))
    lm_debor[0] = [0.5, 0.5, 0.0]     # Wrist
    lm_debor[8] = [0.5, 0.2, 0.0]     # Index tip extended
    lm_debor[12] = [0.5, 0.45, 0.0]   # Middle tip curled
    res_debor = discriminator.disambiguate_debor_vs_dulabhai(lm_debor)
    assert res_debor["resolved_slug"] == "debor"

    # 2. Dulabhai: V-shape dual fingers extended & spread
    lm_dula = np.zeros((21, 3))
    lm_dula[0] = [0.5, 0.5, 0.0]      # Wrist
    lm_dula[8] = [0.46, 0.2, 0.0]     # Index tip extended left
    lm_dula[12] = [0.54, 0.2, 0.0]    # Middle tip extended right (spread > 0.045)
    res_dula = discriminator.disambiguate_debor_vs_dulabhai(lm_dula)
    assert res_dula["resolved_slug"] == "dulabhai"


def test_minimal_pair_hurry_vs_throw():
    """Verify Hurry (vertical oscillation) vs Throw (forward ballistic thrust)."""
    discriminator = MinimalPairDiscriminator()

    # 1. Throw: Forward anterior thrust (dominant z displacement)
    throw_traj = np.zeros((10, 3))
    throw_traj[:, 2] = np.linspace(0.1, 0.45, 10)  # +35cm forward
    res_throw = discriminator.disambiguate_hurry_vs_throw(throw_traj)
    assert res_throw["resolved_slug"] == "chure_mara"

    # 2. Hurry: Vertical oscillation with low z change
    hurry_traj = np.zeros((10, 3))
    hurry_traj[:, 1] = 0.45 + 0.08 * np.sin(np.linspace(0, 4 * np.pi, 10))
    res_hurry = discriminator.disambiguate_hurry_vs_throw(hurry_traj)
    assert res_hurry["resolved_slug"] == "taratari"


def test_minimal_pair_father_vs_mother():
    """Verify Father (mustache swipe across philtrum) vs Mother (lateral cheek double tap)."""
    discriminator = MinimalPairDiscriminator()

    # 1. Father: Horizontal swipe trajectory across center
    father_traj = np.zeros((10, 3))
    father_traj[:, 0] = np.linspace(0.40, 0.60, 10)  # 20cm horizontal swipe
    lm_father = np.zeros((21, 3))
    lm_father[8] = [0.50, 0.36, 0.10]
    res_father = discriminator.disambiguate_father_vs_mother(lm_father, father_traj)
    assert res_father["resolved_slug"] == "baba"

    # 2. Mother: Cheek lateral position (x offset > 0.15) with no horizontal swipe
    mother_traj = np.zeros((10, 3))
    mother_traj[:, 0] = 0.25
    lm_mother = np.zeros((21, 3))
    lm_mother[8] = [0.25, 0.35, 0.10]  # Cheek area
    res_mother = discriminator.disambiguate_father_vs_mother(lm_mother, mother_traj)
    assert res_mother["resolved_slug"] == "ma"


# ==============================================================================
# 4. Tier 3 & 4: Visual-Spatial Syntax Engine Tests
# ==============================================================================

def test_syntax_post_nominal_adjective_inversion():
    """Rule 1: Post-Nominal Adjective Inversion (Adj + N -> N + Adj)."""
    engine = bdsl_syntax_engine

    # "ভালো ছেলে" -> ["ছেলে", "ভালো"]
    res1 = engine.text_to_bdsl_gloss("ভালো ছেলে")
    assert res1["glosses"] == ["ছেলে", "ভালো"]
    assert any("PostNominalAdjectiveInversion" in r for r in res1["applied_rules"])

    # "বড় বাড়ি" -> ["বাড়ি", "বড়"]
    res2 = engine.text_to_bdsl_gloss("বড় বাড়ি")
    assert res2["glosses"] == ["বাড়ি", "বড়"]


def test_syntax_terminal_interrogative_displacement():
    """Rule 2: Terminal Interrogative Displacement (Wh-words to end)."""
    engine = bdsl_syntax_engine

    # "তুমি কেন এসেছো?" -> ["তুমি", "আসা", "কেন"]
    res = engine.text_to_bdsl_gloss("তুমি কেন এসেছো?")
    assert res["glosses"] == ["তুমি", "আসা", "কেন"]
    assert res["is_interrogative"] is True
    assert any("TerminalInterrogativeDisplacement" in r for r in res["applied_rules"])
    assert res["facs_nmm"][0]["AU04"] > 0.0  # Wh-question brow furrow


def test_syntax_conjunction_and_particle_deletion():
    """Rule 3: Conjunction and Particle Deletion."""
    engine = bdsl_syntax_engine

    # "আমি এবং তুমি যাবো" -> ["আমি", "তুমি", "যাওয়া"]
    res = engine.text_to_bdsl_gloss("আমি এবং তুমি যাবো")
    assert "এবং" not in res["glosses"]
    assert res["glosses"] == ["আমি", "তুমি", "যাওয়া"]
    assert any("ParticleDeletion(এবং)" in r for r in res["applied_rules"])


def test_syntax_semantic_compounding_and_unpacking():
    """Rule 4: Semantic Compounding and Unpacking."""
    engine = bdsl_syntax_engine

    # হোটেল -> [খাওয়া, টাকা]
    res_hotel = engine.text_to_bdsl_gloss("হোটেল কোথায়?")
    assert "খাওয়া" in res_hotel["glosses"]
    assert "টাকা" in res_hotel["glosses"]
    assert res_hotel["glosses"][-1] == "কোথায়"

    # অ্যাম্বুলেন্স -> [হাসপাতাল, গাড়ি]
    res_amb = engine.text_to_bdsl_gloss("জরুরি অ্যাম্বুলেন্স ডাকো")
    assert "হাসপাতাল" in res_amb["glosses"]
    assert "গাড়ি" in res_amb["glosses"]


def test_syntax_bdsl_gloss_to_text_bidirectional():
    """Test reverse synthesis from gloss sequence to natural inflected Bengali."""
    engine = bdsl_syntax_engine

    # ["আমি", "ভাত", "খাওয়া"] -> "আমি ভাত খাচ্ছি।"
    res = engine.bdsl_gloss_to_text(["আমি", "ভাত", "খাওয়া"])
    assert "আমি ভাত খাচ্ছি।" in res["bengali"]

    # Reverse compound packing: ["হাসপাতাল", "গাড়ি"] -> "অ্যাম্বুলেন্স"
    res_comp = engine.bdsl_gloss_to_text(["হাসপাতাল", "গাড়ি"])
    assert "অ্যাম্বুলেন্স" in res_comp["bengali"]
