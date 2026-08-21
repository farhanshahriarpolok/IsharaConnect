"""Unit tests for the Advanced Continuous BdSL Grammar Engine."""

import pytest
from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine


@pytest.fixture
def engine():
    return AdvancedBdSLGrammarEngine()


def test_empty_gloss_sequence(engine):
    """Test handling of empty or whitespace gloss input."""
    res = engine.generate_natural_sentence([])
    assert res["bengali"] == ""
    assert res["english"] == ""
    assert res["confidence"] == 0.0

    res_spaces = engine.generate_natural_sentence(["   ", ""])
    assert res_spaces["bengali"] == ""


def test_doctor_visit_inflection(engine):
    """Test morphological inflection and postposition for doctor visit."""
    glosses = ["আমি", "ডাক্তার", "যাওয়া", "প্রয়োজন"]
    res = engine.generate_natural_sentence(glosses)

    assert "ডাক্তারের কাছে" in res["bengali"]
    assert res["bengali"].endswith("।")
    assert "doctor" in res["english"].lower()
    assert "need" in res["english"].lower()
    assert res["confidence"] >= 0.9


def test_emergency_help_inflection(engine):
    """Test need-based pronoun shift (আমি -> আমার) with emergency."""
    glosses = ["আমি", "জরুরি", "সাহায্য", "প্রয়োজন"]
    res = engine.generate_natural_sentence(glosses)

    assert res["bengali"].startswith("আমার")
    assert "জরুরি" in res["bengali"]
    assert "সাহায্য" in res["bengali"]
    assert res["english"] == "I need emergency help."


def test_interrogative_hospital_location(engine):
    """Test question sentence generation and punctuation."""
    glosses = ["হাসপাতাল", "কোথায়"]
    res = engine.generate_natural_sentence(glosses)

    assert res["bengali"].endswith("?")
    assert "হাসপাতাল" in res["bengali"]
    assert res["english"].endswith("?")
    assert "hospital" in res["english"].lower()


def test_greeting_idioms(engine):
    """Test single greeting idiom processing and punctuation."""
    res_thanks = engine.generate_natural_sentence(["ধন্যবাদ"])
    assert res_thanks["bengali"] == "ধন্যবাদ!"
    assert res_thanks["english"] == "Thank you!"

    res_salam = engine.generate_natural_sentence(["আসসালামু আলাইকুম"])
    assert "আসসালামু আলাইকুম" in res_salam["bengali"]


def test_negation_understanding(engine):
    """Test negation sentence generation."""
    res_not_understood = engine.generate_natural_sentence(["আমি", "বুঝেছি", "না"])
    assert "পারিনি" in res_not_understood["bengali"] or "না" in res_not_understood["bengali"]
    assert "not" in res_not_understood["english"].lower() or "did not" in res_not_understood["english"].lower()


def test_backward_compatibility_translate_gloss_sequence(engine):
    """Test translate_gloss_sequence backward compatibility interface."""
    res = engine.translate_gloss_sequence(["আমি", "পানি", "খাওয়া", "ইচ্ছা"])
    assert "bengali_sentence" in res
    assert "english_sentence" in res
    assert "raw_glosses" in res
    assert "পানি" in res["bengali_sentence"]
