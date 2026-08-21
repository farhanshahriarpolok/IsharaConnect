import pytest
from core_engine.nlp.gloss_translator import BdSLGlossTranslator

@pytest.fixture
def translator():
    return BdSLGlossTranslator()

def test_translate_empty(translator):
    res = translator.translate_gloss_sequence([])
    assert res["bengali_sentence"] == ""

def test_pronoun_inflection(translator):
    # আমি + সাহায্য + প্রয়োজন -> আমার সাহায্য প্রয়োজন
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
