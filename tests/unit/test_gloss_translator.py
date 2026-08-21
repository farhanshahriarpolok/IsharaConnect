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
    
    assert res["bengali_sentence"] == "আমার সাহায্য প্রয়োজন"
    assert res["english_sentence"] == "I need help"

def test_negation_attachment(translator):
    # আমি + খাবার + না -> আমি খাবার না
    # With need: আমি + সাহায্য + প্রয়োজন + না -> আমার সাহায্য প্রয়োজন না
    glosses = ["আমি", "সাহায্য", "প্রয়োজন", "না"]
    res = translator.translate_gloss_sequence(glosses)
    
    assert res["bengali_sentence"] == "আমার সাহায্য প্রয়োজন না"
    assert res["english_sentence"] == "I do not need help"

def test_basic_gloss(translator):
    glosses = ["ধন্যবাদ"]
    res = translator.translate_gloss_sequence(glosses)
    assert res["bengali_sentence"] == "ধন্যবাদ"
    assert res["english_sentence"] == "Thank you"
