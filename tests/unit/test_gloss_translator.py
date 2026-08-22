"""Unit tests for Continuous Sign Language Translation (CSLR -> SLT) NLP Suite."""

import pytest

from core_engine.nlp.bengali_inflection import BengaliMorphologicalInflector
from core_engine.nlp.gloss_to_sentence import GlossToSentenceTranslator
from core_engine.nlp.gloss_translator import BdSLGlossTranslator
from core_engine.nlp.temporal_debouncer import TemporalGlossDebouncer


@pytest.fixture
def translator():
    return BdSLGlossTranslator()


@pytest.fixture
def gloss_to_sentence():
    return GlossToSentenceTranslator()


def test_translate_empty(translator):
    res = translator.translate_gloss_sequence([])
    assert res["bengali_sentence"] == ""


def test_pronoun_inflection(translator):
    glosses = ["আমি", "সাহায্য", "প্রয়োজন"]
    res = translator.translate_gloss_sequence(glosses)
    assert "আমার সাহায্য প্রয়োজন" in res["bengali_sentence"]
    assert "I need help" in res["english_sentence"]


def test_negation_attachment(translator):
    glosses = ["আমি", "সাহায্য", "প্রয়োজন", "না"]
    res = translator.translate_gloss_sequence(glosses)
    assert "আমার" in res["bengali_sentence"]
    assert "সাহায্য" in res["bengali_sentence"]
    assert "I" in res["english_sentence"]
    assert "need" in res["english_sentence"]


def test_basic_gloss(translator):
    glosses = ["ধন্যবাদ"]
    res = translator.translate_gloss_sequence(glosses)
    assert "ধন্যবাদ" in res["bengali_sentence"]
    assert "Thank you" in res["english_sentence"]


# ==========================================
# Sprint 23 Continuous SLT & Inflection Tests
# ==========================================

def test_temporal_debouncer_jitter_suppression():
    """Test debouncer filters out transient noise and requires min_consecutive frames."""
    debouncer = TemporalGlossDebouncer(min_consecutive=3, confidence_thresh=0.65)

    # Low confidence -> ignored
    assert debouncer.add_prediction("ভাত", confidence=0.4) is None

    # 1st and 2nd frame -> not yet emitted
    assert debouncer.add_prediction("ভাত", confidence=0.8) is None
    assert debouncer.add_prediction("ভাত", confidence=0.8) is None

    # 3rd consecutive frame -> emitted
    assert debouncer.add_prediction("ভাত", confidence=0.85) == "ভাত"

    # Consecutive repeat frames -> not emitted again
    assert debouncer.add_prediction("ভাত", confidence=0.9) is None

    # Next distinct word
    assert debouncer.add_prediction("খাওয়া", confidence=0.85) is None
    assert debouncer.add_prediction("খাওয়া", confidence=0.85) is None
    assert debouncer.add_prediction("খাওয়া", confidence=0.85) == "খাওয়া"

    assert debouncer.get_stable_tokens() == ["ভাত", "খাওয়া"]


def test_temporal_debouncer_pause_boundary():
    """Test pause duration triggers sentence boundary flag."""
    debouncer = TemporalGlossDebouncer(min_consecutive=2, pause_threshold_s=1.0)

    # Ingest tokens
    debouncer.add_prediction("আমি", 0.9, timestamp=100.0)
    debouncer.add_prediction("আমি", 0.9, timestamp=100.1)
    assert debouncer.get_stable_tokens() == ["আমি"]

    # Pause of 1.5 seconds (idle/rest)
    debouncer.add_prediction("IDLE", 0.1, timestamp=101.6)
    assert debouncer.is_sentence_boundary() is True

    # Flush clears buffer
    flushed = debouncer.flush()
    assert flushed == ["আমি"]
    assert len(debouncer.get_stable_tokens()) == 0


def test_bengali_inflection_verb_conjugation():
    """Test morphological root verb conjugation by person and tense."""
    inflector = BengaliMorphologicalInflector

    # 1st Person (আমি)
    assert inflector.conjugate_verb("আমি", "খাওয়া", "present_continuous") == "খাচ্ছি"
    assert inflector.conjugate_verb("আমি", "খাওয়া", "future") == "খাব"
    assert inflector.conjugate_verb("আমি", "যাওয়া", "present_continuous") == "যাচ্ছি"

    # 2nd Person Honorific (আপনি)
    assert inflector.conjugate_verb("আপনি", "খাওয়া", "present_continuous") == "খাচ্ছেন"
    assert inflector.conjugate_verb("আপনি", "যাওয়া", "future") == "যাবেন"

    # 2nd Person Familiar (তুমি)
    assert inflector.conjugate_verb("তুমি", "দেখা", "present_continuous") == "দেখছ"
    assert inflector.conjugate_verb("তুমি", "বলা", "future") == "বলবে"


def test_bengali_inflection_vibhakti():
    """Test case ending (vibhakti) transformations."""
    inflector = BengaliMorphologicalInflector

    # Locatives (-এ, -তে, -য়)
    assert inflector.apply_vibhakti("হাসপাতাল", "locative") == "হাসপাতালে"
    assert inflector.apply_vibhakti("স্কুল", "locative") == "স্কুলে"
    assert inflector.apply_vibhakti("বাড়ি", "locative") == "বাড়িতে"
    assert inflector.apply_vibhakti("থানা", "locative") == "থানায়"

    # Accusatives (-কে)
    assert inflector.apply_vibhakti("ডাক্তার", "accusative") == "ডাক্তারকে"
    assert inflector.apply_vibhakti("পুলিশ", "accusative") == "পুলিশকে"


def test_gloss_to_sentence_translator_continuous_sequences(gloss_to_sentence):
    """Test end-to-end continuous sign translation into natural sentences with correct punctuation."""
    # 1. Subject + Object + Verb root
    res1 = gloss_to_sentence.translate(["আমি", "ভাত", "খাওয়া"])
    assert res1["translated_text"] == "আমি ভাত খাচ্ছি।"
    assert res1["confidence"] >= 0.9
    assert res1["is_final"] is True

    # 2. Introduction Pattern
    res2 = gloss_to_sentence.translate(["আমার", "নাম", "রাকিব"])
    assert "আমার নাম রাকিব" in res2["translated_text"]
    assert res2["translated_text"].endswith("।")

    # 3. Interrogative Question Pattern
    res3 = gloss_to_sentence.translate(["আপনি", "কেমন", "আছেন"])
    assert "আপনি কেমন আছেন" in res3["translated_text"]
    assert res3["translated_text"].endswith("?")


def test_gloss_to_sentence_stream_processing(gloss_to_sentence):
    """Test real-time stream ingestion and sentence generation."""
    gloss_to_sentence.reset()

    # Stream frame predictions for "আমি" (3 frames)
    gloss_to_sentence.process_stream("আমি", 0.9, timestamp=10.0)
    gloss_to_sentence.process_stream("আমি", 0.9, timestamp=10.05)
    r = gloss_to_sentence.process_stream("আমি", 0.9, timestamp=10.1)

    assert r["is_final"] is False
    assert "আমি" in r["translated_text"]
