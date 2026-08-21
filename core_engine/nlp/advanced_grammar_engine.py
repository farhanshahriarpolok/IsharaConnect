"""Advanced Continuous BdSL Grammar & Natural Language Processing Engine.

Handles morphological inflections (নামবিভক্তি, কাল ও পুরুষ প্রত্যয়),
contextual word smoothing, postpositions, and dual Bengali-English synthesis.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AdvancedBdSLGrammarEngine:
    """Production NLP Grammar Engine for translating BdSL gloss sequences into natural sentences."""

    def __init__(self):
        # 1. Pronoun Inflections (Nominative -> Genitive/Possessive & Accusative)
        self.pronoun_possessive = {
            "আমি": "আমার",
            "তুমি": "তোমার",
            "আপনি": "আপনার",
            "সে": "তার",
            "তিনি": "তাঁর",
            "আমরা": "আমাদের",
            "তোমরা": "তোমাদের",
            "আপনারা": "আপনাদের",
            "তারা": "তাদের",
            "এরা": "এদের",
            "ওরা": "ওদের"
        }

        self.pronoun_accusative = {
            "আমি": "আমাকে",
            "তুমি": "তোমাকে",
            "আপনি": "আপনাকে",
            "সে": "তাকে",
            "তিনি": "তাঁকে",
            "আমরা": "আমাদেরকে",
            "তোমরা": "তোমাদেরকে",
            "আপনারা": "আপনাদেরকে",
            "তারা": "তাদেরকে"
        }

        # 2. Case Endings (বিভক্তি) for Nouns & Entities
        self.noun_genitive_map = {
            "ডাক্তার": "ডাক্তারের",
            "হাসপাতাল": "হাসপাতালের",
            "পুলিশ": "পুলিশের",
            "থানা": "থানার",
            "মা": "মায়ের",
            "বাবা": "বাবার",
            "ভাই": "ভাইয়ের",
            "বোন": "বোনের",
            "বন্ধু": "বন্ধুর",
            "সাহায্য": "সাহায্যের",
            "ওষুধ": "ওষুধের",
            "পানি": "পানির",
            "খাবার": "খাবারের",
            "বাড়ি": "বাড়ির",
            "স্কুল": "স্কুলের",
            "ব্যাংক": "ব্যাংকের"
        }

        self.noun_locative_map = {
            "হাসপাতাল": "হাসপাতালে",
            "বাড়ি": "বাড়িতে",
            "স্কুল": "স্কুলে",
            "ব্যাংক": "ব্যাংকে",
            "থানা": "থানায়",
            "বাজার": "বাজারে",
            "দোকান": "দোকানে",
            "ঢাকা": "ঢাকায়",
            "অফিস": "অফিসে",
            "রুম": "রুমে"
        }

        # 3. Persons / Humans that require "কাছে" when visited/approached
        self.human_entities = {
            "ডাক্তার", "পুলিশ", "মা", "বাবা", "ভাই", "বোন", "বন্ধু", "শিক্ষক", "উকিল"
        }

        # 4. English Dictionary Mapping
        self.english_lexicon = {
            "আমি": "I",
            "আমার": "my",
            "আমাকে": "me",
            "তুমি": "you",
            "তোমার": "your",
            "তোমাকে": "you",
            "আপনি": "you",
            "আপনার": "your",
            "আপনাকে": "you",
            "সে": "he/she",
            "তার": "his/her",
            "তাকে": "him/her",
            "আমরা": "we",
            "আমাদের": "our",
            "সাহায্য": "help",
            "প্রয়োজন": "need",
            "দরকার": "need",
            "জরুরি": "emergency",
            "ধন্যবাদ": "thank you",
            "স্বাগতম": "welcome",
            "আসসালামু আলাইকুম": "peace be upon you",
            "কেমন": "how",
            "আছেন": "are you",
            "আছ": "are you",
            "ভালো": "good",
            "খারাপ": "bad",
            "নাম": "name",
            "পানি": "water",
            "খাবার": "food",
            "ডাক্তার": "doctor",
            "ওষুধ": "medicine",
            "হাসপাতাল": "hospital",
            "পুলিশ": "police",
            "থানা": "police station",
            "ব্যাংক": "bank",
            "টাকা": "money",
            "ব্যথা": "pain",
            "অসুস্থ": "sick",
            "যাওয়া": "go",
            "আসা": "come",
            "খাওয়া": "eat",
            "পান": "drink",
            "দেখা": "see",
            "বলা": "say",
            "চাওয়া": "want",
            "ইচ্ছা": "want",
            "হ্যাঁ": "yes",
            "না": "no",
            "নেই": "no/not available",
            "কোথায়": "where",
            "কখন": "when",
            "কেন": "why",
            "কি": "what",
            "বুঝেছি": "understood",
            "বুঝিনি": "did not understand",
            "ঠিক আছে": "all right",
            "বিদায়": "goodbye",
            "শুভ সকাল": "good morning",
            "শুভ রাত্রি": "good night"
        }

        # 5. Sadhu -> Cholit Standardization Rules
        self.sadhu_to_cholit = {
            "হইয়াছে": "হয়েছে",
            "করিতেছি": "করছি",
            "করিয়াছি": "করেছি",
            "যাইতেছি": "যাচ্ছি",
            "যাইব": "যাব",
            "আসিয়াছি": "এসেছি",
            "দেখিয়াছি": "দেখেছি",
            "তাহাদিগকে": "তাদেরকে",
            "উহাদের": "ওদের"
        }

    def standardize_bengali(self, text: str) -> str:
        """Standardizes Sadhu words to modern Cholit form."""
        words = text.split()
        converted = [self.sadhu_to_cholit.get(w, w) for w in words]
        return " ".join(converted)

    def apply_noun_inflection(self, noun: str, case_type: str = "genitive") -> str:
        """Applies Bengali noun case ending (বিভক্তি)."""
        if case_type == "genitive":
            if noun in self.noun_genitive_map:
                return self.noun_genitive_map[noun]
            if noun.endswith(('া', 'ি', 'ী', 'ু', 'ূ', 'ে', 'ো')):
                return noun + "র"
            return noun + "ের"

        elif case_type == "locative":
            if noun in self.noun_locative_map:
                return self.noun_locative_map[noun]
            if noun.endswith(('া', 'ে', 'ো')):
                return noun + "য়"
            return noun + "ে"

        elif case_type == "accusative":
            if noun.endswith(('া', 'ি', 'ী', 'ু', 'ূ', 'ে', 'ো')):
                return noun + "কে"
            return noun + "কে"

        return noun

    def generate_natural_sentence(self, gloss_sequence: List[str]) -> Dict[str, Any]:
        """Translates a raw sequence of BdSL glosses into fluent, grammatically accurate sentences.

        Args:
            gloss_sequence: List of strings representing recognized BdSL gloss tokens.

        Returns:
            dict containing:
                - 'bengali': Grammatically punctuated Bengali sentence.
                - 'english': Formatted natural English translation.
                - 'glosses': The original input list.
                - 'confidence': Heuristic confidence score (0.0 to 1.0).
        """
        if not gloss_sequence:
            return {
                "bengali": "",
                "english": "",
                "glosses": [],
                "confidence": 0.0
            }

        # Clean glosses
        cleaned_glosses = [g.strip() for g in gloss_sequence if g and g.strip()]
        if not cleaned_glosses:
            return {"bengali": "", "english": "", "glosses": [], "confidence": 0.0}

        # 1. Check exact phrase templates / idioms
        idiom_result = self._match_idioms_and_patterns(cleaned_glosses)
        if idiom_result:
            return idiom_result

        # 2. General Rule-Based Grammatical Smoothing
        return self._rule_based_synthesis(cleaned_glosses)

    def _match_idioms_and_patterns(self, glosses: List[str]) -> Optional[Dict[str, Any]]:
        """Handles high-frequency communicative patterns and idioms."""
        g_set = set(glosses)
        g_str = " ".join(glosses)

        # Pattern: [আমি, ডাক্তার, যাওয়া, প্রয়োজন / দরকার] or variations
        if ("আমি" in g_set or "আমার" in g_set) and "ডাক্তার" in g_set and ("যাওয়া" in g_set or "যেতে" in g_set or "প্রয়োজন" in g_set or "দরকার" in g_set):
            return {
                "bengali": "আমার ডাক্তারের কাছে যাওয়া প্রয়োজন।",
                "english": "I need to visit a doctor.",
                "glosses": glosses,
                "confidence": 0.98
            }

        # Pattern: [আমি / আমার, জরুরি, সাহায্য, প্রয়োজন]
        if ("আমি" in g_set or "আমার" in g_set) and "সাহায্য" in g_set and ("জরুরি" in g_set or "প্রয়োজন" in g_set):
            is_emergency = "জরুরি" in g_set
            bn = "আমার জরুরি সাহায্য প্রয়োজন।" if is_emergency else "আমার সাহায্য প্রয়োজন।"
            en = "I need emergency help." if is_emergency else "I need help."
            return {
                "bengali": bn,
                "english": en,
                "glosses": glosses,
                "confidence": 0.98
            }

        # Pattern: [তুমি / আপনি, কেমন, আছেন / আছ]
        if "কেমন" in g_set and ("আছেন" in g_set or "আছ" in g_set or "তুমি" in g_set or "আপনি" in g_set):
            if "আপনি" in g_set or "আছেন" in g_set:
                return {
                    "bengali": "আপনি কেমন আছেন?",
                    "english": "How are you?",
                    "glosses": glosses,
                    "confidence": 0.99
                }
            else:
                return {
                    "bengali": "তুমি কেমন আছ?",
                    "english": "How are you?",
                    "glosses": glosses,
                    "confidence": 0.99
                }

        # Pattern: [হাসপাতাল / থানা / ব্যাংক, কোথায়]
        loc_words = [w for w in ["হাসপাতাল", "থানা", "ব্যাংক", "টয়লেট", "অফিস", "ডাক্তারখানা"] if w in g_set]
        if loc_words and "কোথায়" in g_set:
            target = loc_words[0]
            en_target = self.english_lexicon.get(target, target)
            return {
                "bengali": f"{target}টি কোথায়?",
                "english": f"Where is the {en_target}?",
                "glosses": glosses,
                "confidence": 0.96
            }

        # Pattern: [আমি, পানি / খাবার / ওষুধ, খাওয়া / পান / চাওয়া / ইচ্ছা]
        if ("আমি" in g_set or "আমার" in g_set) and ("পানি" in g_set or "খাবার" in g_set or "ওষুধ" in g_set) and ("খাওয়া" in g_set or "পান" in g_set or "চাওয়া" in g_set or "ইচ্ছা" in g_set):
            item = "পানি" if "পানি" in g_set else ("ওষুধ" if "ওষুধ" in g_set else "খাবার")
            action = "খেতে" if item != "পানি" else "পান করতে"
            en_item = self.english_lexicon.get(item, item)
            en_verb = "drink" if item == "পানি" else "take" if item == "ওষুধ" else "eat"
            return {
                "bengali": f"আমি {item} {action} চাই।",
                "english": f"I want to {en_verb} {en_item}.",
                "glosses": glosses,
                "confidence": 0.97
            }

        # Pattern: Single Greetings / Farewells
        if len(glosses) == 1:
            g = glosses[0]
            if g in ["ধন্যবাদ", "স্বাগতম", "শুভ সকাল", "শুভ রাত্রি", "আসসালামু আলাইকুম", "বিদায়"]:
                punct = "!" if g in ["ধন্যবাদ", "স্বাগতম"] else "।"
                en_trans = self.english_lexicon.get(g, g).capitalize() + ("!" if punct == "!" else ".")
                return {
                    "bengali": f"{g}{punct}",
                    "english": en_trans,
                    "glosses": glosses,
                    "confidence": 0.99
                }

        # Pattern: Understanding check
        if "বুঝিনি" in g_set or ("বুঝেছি" in g_set and "না" in g_set):
            return {
                "bengali": "আমি বুঝতে পারিনি।",
                "english": "I did not understand.",
                "glosses": glosses,
                "confidence": 0.95
            }
        elif "বুঝেছি" in g_set or "বুঝেছি" == g_str:
            return {
                "bengali": "আমি বুঝেছি।",
                "english": "I understood.",
                "glosses": glosses,
                "confidence": 0.95
            }

        return None

    def _rule_based_synthesis(self, glosses: List[str]) -> Dict[str, Any]:
        """Synthesizes Bengali & English based on structural grammatical rules."""
        bn_tokens = []
        en_tokens = []

        has_need = any(w in glosses for w in ["প্রয়োজন", "দরকার"])
        is_negated = any(w in glosses for w in ["না", "নেই", "নয়", "নাই"])
        is_question = any(w in glosses for w in ["কোথায়", "কেমন", "কখন", "কেন", "কি", "কার"])

        core_glosses = [g for g in glosses if g not in ["না", "নেই", "নয়", "নাই"]]

        for i, gloss in enumerate(core_glosses):
            # Check pronoun inflection
            if gloss in self.pronoun_possessive and has_need:
                inflected_bn = self.pronoun_possessive[gloss]
                bn_tokens.append(inflected_bn)
                en_tokens.append("I" if gloss == "আমি" else self.english_lexicon.get(gloss, gloss))
                continue

            # Check if next word is 'যাওয়া' and current word is a human entity -> add genitive + 'কাছে'
            if i < len(core_glosses) - 1:
                next_word = core_glosses[i + 1]
                if next_word in ["যাওয়া", "যেতে", "আসা", "আসতে"] and gloss in self.human_entities:
                    inflected_noun = self.apply_noun_inflection(gloss, "genitive")
                    bn_tokens.append(f"{inflected_noun} কাছে")
                    en_noun = self.english_lexicon.get(gloss, gloss)
                    en_tokens.append(f"visit {en_noun}")
                    continue

            # Check standard noun/word mapping
            bn_tokens.append(gloss)
            en_tokens.append(self.english_lexicon.get(gloss, gloss))

        # Handle Bengali Negation
        if is_negated:
            bn_tokens.append("না")
            if "I" in en_tokens:
                idx = en_tokens.index("I")
                en_tokens.insert(idx + 1, "do not")
            else:
                en_tokens.append("not")

        # English sentence construction heuristics
        if has_need and "need" not in en_tokens:
            en_tokens.insert(1 if len(en_tokens) > 1 else 0, "need")

        # Clean duplicates in English
        dedup_en = []
        for w in en_tokens:
            if not dedup_en or dedup_en[-1].lower() != w.lower():
                dedup_en.append(w)

        # Assemble Bengali
        raw_bn = " ".join(bn_tokens)
        standardized_bn = self.standardize_bengali(raw_bn)
        
        # Punctuation
        if is_question:
            if not standardized_bn.endswith("?"):
                standardized_bn += "?"
        else:
            if not standardized_bn.endswith(("।", "!", "?")):
                standardized_bn += "।"

        # Assemble English
        raw_en = " ".join(dedup_en).strip()
        if raw_en:
            raw_en = raw_en[0].upper() + raw_en[1:]
            if is_question and not raw_en.endswith("?"):
                raw_en += "?"
            elif not raw_en.endswith((".", "!", "?")):
                raw_en += "."

        return {
            "bengali": standardized_bn,
            "english": raw_en,
            "glosses": glosses,
            "confidence": 0.88
        }

    def translate_gloss_sequence(self, gloss_list: List[str]) -> Dict[str, Any]:
        """Backward-compatible adapter matching the interface of BdSLGlossTranslator."""
        result = self.generate_natural_sentence(gloss_list)
        return {
            "bengali_sentence": result["bengali"],
            "english_sentence": result["english"],
            "raw_glosses": result["glosses"],
            "confidence": result.get("confidence", 0.9)
        }
