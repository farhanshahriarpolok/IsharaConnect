import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class BdSLGlossTranslator:
    """
    Translates a sequence of isolated BdSL glosses into a grammatically correct Bengali and English sentence.
    Uses basic SOV rules and morphological adaptations.
    """
    
    def __init__(self):
        # Basic mappings
        self.pronoun_inflections = {
            "আমি": "আমার",
            "তুমি": "তোমার",
            "সে": "তার",
            "আমরা": "আমাদের",
        }
        
        self.english_dict = {
            "আমি": "I",
            "আমার": "My",
            "তুমি": "You",
            "তোমার": "Your",
            "সে": "He/She",
            "তার": "His/Her",
            "আমরা": "We",
            "আমাদের": "Our",
            "সাহায্য": "help",
            "প্রয়োজন": "need",
            "ধন্যবাদ": "thank you",
            "কেমন": "how",
            "আছেন": "are",
            "ভালো": "good",
            "খারাপ": "bad",
            "নাম": "name",
            "পানি": "water",
            "খাবার": "food",
            "ডাক্তার": "doctor",
            "ওষুধ": "medicine",
            "হাসপাতাল": "hospital",
            "হ্যাঁ": "yes",
            "না": "no",
            "বুঝেছি": "understood",
            "বুঝিনি": "not understood",
            "ঠিক আছে": "okay",
            "স্বাগতম": "welcome"
        }

    def translate_gloss_sequence(self, gloss_list: List[str]) -> Dict[str, str]:
        """
        Translates a list of raw glosses into Bengali and English sentences.
        Example: ['আমি', 'সাহায্য', 'প্রয়োজন'] -> "আমার সাহায্য প্রয়োজন", "I need help"
        """
        if not gloss_list:
            return {"bengali_sentence": "", "english_sentence": "", "raw_glosses": []}

        bn_words = []
        en_words = []
        
        has_need = "প্রয়োজন" in gloss_list
        is_negated = "না" in gloss_list or "নেই" in gloss_list
        
        # Filter out negations from the main loop to append them at the end (SOV syntax)
        core_glosses = [g for g in gloss_list if g not in ["না", "নেই"]]
        
        for i, gloss in enumerate(core_glosses):
            # Morphological rule: If "আমি" is followed by a need/possession, it becomes "আমার"
            if gloss in self.pronoun_inflections and has_need:
                inflected = self.pronoun_inflections[gloss]
                bn_words.append(inflected)
                if inflected == "আমার" and has_need:
                    # In English: "I need help" vs "My need help" -> Handle specifically
                    en_words.append("I")
                else:
                    en_words.append(self.english_dict.get(inflected, inflected))
            else:
                bn_words.append(gloss)
                en_words.append(self.english_dict.get(gloss, gloss))
                
        # Append negation at the end for Bengali
        if is_negated:
            bn_words.append("না")
            # For English, simplified structural negation insertion
            if "I" in en_words and len(en_words) > 1:
                en_words.insert(1, "do not")
            else:
                en_words.append("not")

        # Basic English reordering (e.g., "I help need" -> "I need help")
        if "I" in en_words and "need" in en_words and "help" in en_words:
            en_words = ["I", "need", "help"]
            if is_negated:
                en_words = ["I", "do not", "need", "help"]

        bengali_sentence = " ".join(bn_words)
        # Sentence casing for English
        english_sentence = " ".join(en_words).capitalize()

        return {
            "bengali_sentence": bengali_sentence,
            "english_sentence": english_sentence,
            "raw_glosses": gloss_list
        }
