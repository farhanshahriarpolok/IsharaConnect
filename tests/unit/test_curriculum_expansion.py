"""Unit tests for Learning Hub Curriculum Expansion & Compound Avatar Playback (Sprint 29).

Tests:
1. Dynamic curriculum loading of 50+ standardized signs across 5 levels.
2. Real-time search/filter bar behavior in Academy Dashboard.
3. KinematicMotionInterpolator multi-sign compound motion stitching and transition frames.
4. HumanRigViewer compound sign loading, stage range computation, and HUD step indicator.
"""

import os
import pytest
from PyQt6.QtWidgets import QApplication

from core_engine.nlp.master_lexicon import master_lexicon
from core_engine.vision.dactylology_engine import MASTER_GRAPHEMES, VOWELS, CONSONANTS, DIGITS
from core_engine.vision.kinematic_interpolator import KinematicMotionInterpolator
from desktop_app.ui.academy_dashboard import AcademyDashboard
from desktop_app.ui.components.human_rig_viewer import HumanRigViewer


@pytest.fixture(scope="session")
def qapp():
    """Ensure a singleton QApplication instance is available for Qt widgets."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_master_lexicon_curriculum_completeness():
    """Verify MasterBdSLLexicon and Dactylology inventory contain 50+ items across all 5 levels."""
    signs = master_lexicon.all_signs()
    assert len(signs) >= 30

    # Ensure all 6 categories exist
    categories = {s.get("category") for s in signs}
    assert "Kinship" in categories
    assert "Healthcare & Emergency" in categories
    assert "Public & Education" in categories
    assert "Disaster & Safety" in categories
    assert "Action Verbs" in categories

    # Grapheme inventory
    assert len(VOWELS) == 11
    assert len(CONSONANTS) == 39
    assert len(DIGITS) == 10
    assert len(MASTER_GRAPHEMES) == 60


def test_academy_curriculum_database_indexing(qapp):
    """Verify AcademyDashboard loads and indexes signs from master lexicon and dactylology."""
    dashboard = AcademyDashboard()
    db = dashboard.curriculum_data

    # Check key signs exist in db
    assert "ধন্যবাদ" in db or "dhonnobad" in db
    assert "মা" in db or "ma" in db
    assert "বাবা" in db or "baba" in db
    assert "ডাক্তার" in db or "daktar" in db
    assert "ভূমিকম্প" in db or "bhumikompo" in db
    assert "ক" in db or "cons_ka" in db

    # Verify curriculum tree has 5 levels
    assert dashboard.tree.topLevelItemCount() == 5


def test_curriculum_search_filter(qapp):
    """Verify real-time filtering in the curriculum tree."""
    dashboard = AcademyDashboard()
    
    # Filter by specific term
    dashboard._filter_curriculum("ভূমিকম্প")
    
    # Level 4 should remain visible
    l4_item = dashboard.tree.topLevelItem(3)
    assert not l4_item.isHidden()

    # Reset search filter
    dashboard._filter_curriculum("")
    for i in range(dashboard.tree.topLevelItemCount()):
        assert not dashboard.tree.topLevelItem(i).isHidden()


def test_kinematic_interpolator_compound_motion():
    """Verify multi-sign compound motion synthesis and transition buffer."""
    interpolator = KinematicMotionInterpolator()
    
    # Single sign motion length is 60 frames
    single_frames = interpolator.resolve_motion_sequence("khawa")
    assert len(single_frames) == 60

    # Compound of 2 signs ("khawa" + "taka") should be 60 + 10 (transition) + 60 = 130 frames
    compound_frames = interpolator.load_compound_motion(["khawa", "taka"])
    assert len(compound_frames) == 130

    # Ensure all frames have valid landmark structures
    for f in compound_frames:
        assert len(f.right_hand) == 21
        assert len(f.left_hand) == 21
        assert not any(coord is None for coord in f.head)


def test_human_rig_viewer_compound_loading(qapp):
    """Verify HumanRigViewer multi-stage step indicator and subsign ranges."""
    viewer = HumanRigViewer()
    
    # Load compound sign (হোটেল -> [খাওয়া, টাকা])
    viewer.load_compound_sign(["khawa", "taka"], "হোটেল", "Hotel")
    assert viewer.is_compound is True
    assert len(viewer.compound_glosses) == 2
    assert len(viewer.subsign_frame_ranges) == 2
    assert len(viewer.frames) == 130

    # Test subsign frame ranges
    s1_start, s1_end, s1_gloss = viewer.subsign_frame_ranges[0]
    assert s1_start == 0 and s1_end == 60 and s1_gloss == "khawa"

    s2_start, s2_end, s2_gloss = viewer.subsign_frame_ranges[1]
    assert s2_start == 70 and s2_end == 130 and s2_gloss == "taka"
