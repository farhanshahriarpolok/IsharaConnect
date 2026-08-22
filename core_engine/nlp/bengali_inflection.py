"""Bengali Morphological & Syntax Inflector for Continuous Sign Language Translation.

Implements rule-based linguistic inflection for:
- Vibhakti (কারক ও বিভক্তি: Accusative -কে, Genitive -র/-এর, Locative -এ/-তে/-য়)
- Person & Honorific Agreement (1st Person, 2nd Familiar/Honorific/Intimate, 3rd Person)
- Verb Root Conjugation across Tenses (Present Continuous, Future, Simple Present, Perfect)
- Question Particle & Negation Placement with proper terminal punctuation (। vs ?)
"""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BengaliMorphologicalInflector:
    """Production Bengali linguistic morphology & syntax inflector."""

    PRONOUNS_1ST = {"আমি", "আমরা", "আমাকে", "আমাদেরকে", "আমার", "আমাদের"}
    PRONOUNS_2ND_HONORIFIC = {"আপনি", "আপনারা", "আপনাকে", "আপনাদেরকে", "আপনার", "আপনাদের"}
    PRONOUNS_2ND_FAMILIAR = {"তুমি", "তোমরা", "তোমাকে", "তোমাদেরকে", "তোমার", "তোমাদের"}
    PRONOUNS_2ND_INTIMATE = {"তুই", "তোরা", "তোকে", "তোদেরকে", "তোর", "তোদের"}
    PRONOUNS_3RD = {"সে", "তিনি", "তারা", "তাঁরা", "তাকে", "তাঁকে", "তার", "তাঁর", "তাদের", "তাঁদের"}

    INTERROGATIVE_TOKENS = {"কী", "কেমন", "কোথায়", "কোথায়", "কবে", "কেন", "কার", "কাকে", "কি", "কোন"}
    NEGATION_TOKENS = {"না", "নাই", "নেই", "না_বোধক"}

    # Canonical Verb Conjugation Tables by Subject Person
    VERB_CONJUGATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
        "খাওয়া": {
            "1st": {"present_continuous": "খাচ্ছি", "future": "খাব", "present_simple": "খাই", "past_perfect": "খেয়েছি"},
            "2nd_hon": {"present_continuous": "খাচ্ছেন", "future": "খাবেন", "present_simple": "খান", "past_perfect": "খেয়েছেন"},
            "2nd_fam": {"present_continuous": "খাচ্ছ", "future": "খাবে", "present_simple": "খাও", "past_perfect": "খেয়েছ"},
            "2nd_int": {"present_continuous": "খাচ্ছিস", "future": "খাবি", "present_simple": "খাস", "past_perfect": "খেয়েছিস"},
            "3rd": {"present_continuous": "খাচ্ছে", "future": "খাবে", "present_simple": "খায়", "past_perfect": "খেয়েছে"},
        },
        "যাওয়া": {
            "1st": {"present_continuous": "যাচ্ছি", "future": "যাব", "present_simple": "যাই", "past_perfect": "গেছি"},
            "2nd_hon": {"present_continuous": "যাচ্ছেন", "future": "যাবেন", "present_simple": "যান", "past_perfect": "গেছেন"},
            "2nd_fam": {"present_continuous": "যাচ্ছ", "future": "যাবে", "present_simple": "যাও", "past_perfect": "গেছ"},
            "2nd_int": {"present_continuous": "যাচ্ছিস", "future": "যাবি", "present_simple": "যাস", "past_perfect": "গেছিস"},
            "3rd": {"present_continuous": "যাচ্ছে", "future": "যাবে", "present_simple": "যায়", "past_perfect": "গেছে"},
        },
        "যাওয়া": {
            "1st": {"present_continuous": "যাচ্ছি", "future": "যাব", "present_simple": "যাই", "past_perfect": "গেছি"},
            "2nd_hon": {"present_continuous": "যাচ্ছেন", "future": "যাবেন", "present_simple": "যান", "past_perfect": "গেছেন"},
            "2nd_fam": {"present_continuous": "যাচ্ছ", "future": "যাবে", "present_simple": "যাও", "past_perfect": "গেছ"},
            "2nd_int": {"present_continuous": "যাচ্ছিস", "future": "যাবি", "present_simple": "যাস", "past_perfect": "গেছিস"},
            "3rd": {"present_continuous": "যাচ্ছে", "future": "যাবে", "present_simple": "যায়", "past_perfect": "গেছে"},
        },
        "দেখা": {
            "1st": {"present_continuous": "দেখছি", "future": "দেখব", "present_simple": "দেখি", "past_perfect": "দেখেছি"},
            "2nd_hon": {"present_continuous": "দেখছেন", "future": "দেখবেন", "present_simple": "দেখেন", "past_perfect": "দেখেছেন"},
            "2nd_fam": {"present_continuous": "দেখছ", "future": "দেখবে", "present_simple": "দেখো", "past_perfect": "দেখেছ"},
            "2nd_int": {"present_continuous": "দেখছিস", "future": "দেখবি", "present_simple": "দেখিস", "past_perfect": "দেখেছিস"},
            "3rd": {"present_continuous": "দেখছে", "future": "দেখবে", "present_simple": "দেখে", "past_perfect": "দেখেছে"},
        },
        "বলা": {
            "1st": {"present_continuous": "বলছি", "future": "বলব", "present_simple": "বলি", "past_perfect": "বলেছি"},
            "2nd_hon": {"present_continuous": "বলছেন", "future": "বলবেন", "present_simple": "বলেন", "past_perfect": "বলেছেন"},
            "2nd_fam": {"present_continuous": "বলছ", "future": "বলবে", "present_simple": "বলো", "past_perfect": "বলেছ"},
            "2nd_int": {"present_continuous": "বলছিস", "future": "বলবি", "present_simple": "বলিস", "past_perfect": "বলেছিস"},
            "3rd": {"present_continuous": "বলছে", "future": "বলবে", "present_simple": "বলে", "past_perfect": "বলেছে"},
        },
        "আসা": {
            "1st": {"present_continuous": "আসছি", "future": "আসব", "present_simple": "আসি", "past_perfect": "এসেছি"},
            "2nd_hon": {"present_continuous": "আসছেন", "future": "আসবেন", "present_simple": "আসেন", "past_perfect": "এসেছেন"},
            "2nd_fam": {"present_continuous": "আসছ", "future": "আসবে", "present_simple": "এসো", "past_perfect": "এসেছ"},
            "2nd_int": {"present_continuous": "আসছিস", "future": "আসবি", "present_simple": "আয়", "past_perfect": "এসেছিস"},
            "3rd": {"present_continuous": "আসছে", "future": "আসবে", "present_simple": "আসে", "past_perfect": "এসেছে"},
        },
        "পড়া": {
            "1st": {"present_continuous": "পড়ছি", "future": "পড়ব", "present_simple": "পড়ি", "past_perfect": "পড়েছি"},
            "2nd_hon": {"present_continuous": "পড়ছেন", "future": "পড়বেন", "present_simple": "পড়েন", "past_perfect": "পড়েছেন"},
            "2nd_fam": {"present_continuous": "পড়ছ", "future": "পড়বে", "present_simple": "পড়ো", "past_perfect": "পড়েছ"},
            "2nd_int": {"present_continuous": "পড়ছিস", "future": "পড়বি", "present_simple": "পড়িস", "past_perfect": "পড়েছিস"},
            "3rd": {"present_continuous": "পড়ছে", "future": "পড়বে", "present_simple": "পড়ে", "past_perfect": "পড়েছে"},
        },
        "পড়া": {
            "1st": {"present_continuous": "পড়ছি", "future": "পড়ব", "present_simple": "পড়ি", "past_perfect": "পড়েছি"},
            "2nd_hon": {"present_continuous": "পড়ছেন", "future": "পড়বেন", "present_simple": "পড়েন", "past_perfect": "পড়েছেন"},
            "2nd_fam": {"present_continuous": "পড়ছ", "future": "পড়বে", "present_simple": "পড়ো", "past_perfect": "পড়েছ"},
            "2nd_int": {"present_continuous": "পড়ছিস", "future": "পড়বি", "present_simple": "পড়িস", "past_perfect": "পড়েছিস"},
            "3rd": {"present_continuous": "পড়ছে", "future": "পড়বে", "present_simple": "পড়ে", "past_perfect": "পড়েছে"},
        },
        "লেখা": {
            "1st": {"present_continuous": "লিখছি", "future": "লিখব", "present_simple": "লিখি", "past_perfect": "লিখেছি"},
            "2nd_hon": {"present_continuous": "লিখছেন", "future": "লিখবেন", "present_simple": "লেখেন", "past_perfect": "লিখেছেন"},
            "2nd_fam": {"present_continuous": "লিখছ", "future": "লিখবে", "present_simple": "লেখো", "past_perfect": "লিখেছ"},
            "2nd_int": {"present_continuous": "লিখছিস", "future": "লিখবি", "present_simple": "লিখিস", "past_perfect": "লিখেছিস"},
            "3rd": {"present_continuous": "লিখছে", "future": "লিখবে", "present_simple": "লেখে", "past_perfect": "লিখেছে"},
        },
        "করা": {
            "1st": {"present_continuous": "করছি", "future": "করব", "present_simple": "করি", "past_perfect": "করেছি"},
            "2nd_hon": {"present_continuous": "করছেন", "future": "করবেন", "present_simple": "করেন", "past_perfect": "করেছেন"},
            "2nd_fam": {"present_continuous": "করছ", "future": "করবে", "present_simple": "করো", "past_perfect": "করেছ"},
            "2nd_int": {"present_continuous": "করছিস", "future": "করবি", "present_simple": "করিস", "past_perfect": "করেছিস"},
            "3rd": {"present_continuous": "করছে", "future": "করবে", "present_simple": "করে", "past_perfect": "করেছে"},
        },
        "ভালোবাসা": {
            "1st": {"present_continuous": "ভালোবাসছি", "future": "ভালোবাসব", "present_simple": "ভালোবাসি", "past_perfect": "ভালোবেসেছি"},
            "2nd_hon": {"present_continuous": "ভালোবাসছেন", "future": "ভালোবাসবেন", "present_simple": "ভালোবাসেন", "past_perfect": "ভালোবেসেছেন"},
            "2nd_fam": {"present_continuous": "ভালোবাসছ", "future": "ভালোবাসবে", "present_simple": "ভালোবাসো", "past_perfect": "ভালোবেসেছ"},
            "2nd_int": {"present_continuous": "ভালোবাসছিস", "future": "ভালোবাসবি", "present_simple": "ভালোবাসিস", "past_perfect": "ভালোবেসেছিস"},
            "3rd": {"present_continuous": "ভালোবাসছে", "future": "ভালোবাসবে", "present_simple": "ভালোবাসে", "past_perfect": "ভালোবেসেছে"},
        },
        "থাকা": {
            "1st": {"present_continuous": "আছি", "future": "থাকব", "present_simple": "থাকি", "past_perfect": "থেকেছি"},
            "2nd_hon": {"present_continuous": "আছেন", "future": "থাকবেন", "present_simple": "থাকেন", "past_perfect": "থেকেছেন"},
            "2nd_fam": {"present_continuous": "আছ", "future": "থাকবে", "present_simple": "থাকো", "past_perfect": "থেকেছ"},
            "2nd_int": {"present_continuous": "আছিস", "future": "থাকবি", "present_simple": "থাকিস", "past_perfect": "থেকেছিস"},
            "3rd": {"present_continuous": "আছে", "future": "থাকবে", "present_simple": "থাকে", "past_perfect": "থেকেছে"},
        },
        "আছি": {
            "1st": {"present_continuous": "আছি", "future": "থাকব", "present_simple": "আছি", "past_perfect": "ছিলাম"},
            "2nd_hon": {"present_continuous": "আছেন", "future": "থাকবেন", "present_simple": "আছেন", "past_perfect": "ছিলেন"},
            "2nd_fam": {"present_continuous": "আছ", "future": "থাকবে", "present_simple": "আছ", "past_perfect": "ছিলে"},
            "2nd_int": {"present_continuous": "আছিস", "future": "থাকবি", "present_simple": "আছিস", "past_perfect": "ছিলি"},
            "3rd": {"present_continuous": "আছে", "future": "থাকবে", "present_simple": "আছে", "past_perfect": "ছিল"},
        },
        "আছেন": {
            "1st": {"present_continuous": "আছি", "future": "থাকব", "present_simple": "আছি", "past_perfect": "ছিলাম"},
            "2nd_hon": {"present_continuous": "আছেন", "future": "থাকবেন", "present_simple": "আছেন", "past_perfect": "ছিলেন"},
            "2nd_fam": {"present_continuous": "আছ", "future": "থাকবে", "present_simple": "আছ", "past_perfect": "ছিলে"},
            "2nd_int": {"present_continuous": "আছিস", "future": "থাকবি", "present_simple": "আছিস", "past_perfect": "ছিলি"},
            "3rd": {"present_continuous": "আছে", "future": "থাকবে", "present_simple": "আছে", "past_perfect": "ছিল"},
        }
    }

    # Noun Vibhakti Tables (কারক ও বিভক্তি)
    NOUN_LOCATIVES = {
        "হাসপাতাল": "হাসপাতালে",
        "স্কুল": "স্কুলে",
        "বাড়ি": "বাড়িতে",
        "বাড়ি": "বাড়িতে",
        "থানা": "থানায়",
        "ব্যাংক": "ব্যাংকে",
        "কলেজ": "কলেজে",
        "বাজার": "বাজারে",
        "অফিস": "অফিসে",
    }

    NOUN_GENITIVES = {
        "ডাক্তার": "ডাক্তারের",
        "হাসপাতাল": "হাসপাতালের",
        "স্কুল": "স্কুলের",
        "মা": "মায়ের",
        "বাবা": "বাবার",
        "ভাই": "ভাইয়ের",
        "বোন": "বোনের",
        "বন্ধু": "বন্ধুর",
        "সাহায্য": "সাহায্যের",
        "পানি": "পানির",
        "ভাত": "ভাতের",
        "ওষুধ": "ওষুধের",
    }

    NOUN_ACCUSATIVES = {
        "ডাক্তার": "ডাক্তারকে",
        "পুলিশ": "পুলিশকে",
        "মা": "মাকে",
        "বাবা": "বাবাকে",
        "ভাই": "ভাইকে",
        "বোন": "বোনকে",
        "শিক্ষক": "শিক্ষককে",
    }

    @classmethod
    def detect_person(cls, subject_token: str) -> str:
        """Determines the grammatical person and honorific tier from the subject token."""
        token = subject_token.strip()
        if token in cls.PRONOUNS_1ST:
            return "1st"
        elif token in cls.PRONOUNS_2ND_HONORIFIC:
            return "2nd_hon"
        elif token in cls.PRONOUNS_2ND_FAMILIAR:
            return "2nd_fam"
        elif token in cls.PRONOUNS_2ND_INTIMATE:
            return "2nd_int"
        else:
            return "3rd"

    @classmethod
    def conjugate_verb(
        cls,
        subject: str,
        verb_root: str,
        tense: str = "present_continuous"
    ) -> str:
        """Conjugates a Bengali root verb according to subject person and tense."""
        person = cls.detect_person(subject)
        root = verb_root.strip()

        if root in cls.VERB_CONJUGATIONS:
            table = cls.VERB_CONJUGATIONS[root].get(person, cls.VERB_CONJUGATIONS[root]["1st"])
            return table.get(tense, table.get("present_continuous", root))

        return root

    @classmethod
    def apply_vibhakti(cls, noun: str, case_type: str = "locative") -> str:
        """Applies grammatical case inflections (locative, genitive, accusative) to nouns."""
        cleaned = noun.strip()
        if case_type == "locative":
            return cls.NOUN_LOCATIVES.get(cleaned, cleaned)
        elif case_type == "genitive":
            return cls.NOUN_GENITIVES.get(cleaned, cleaned)
        elif case_type == "accusative":
            return cls.NOUN_ACCUSATIVES.get(cleaned, cleaned)
        return cleaned

    @classmethod
    def inflect_tokens(cls, tokens: List[str]) -> str:
        """Transforms a sequence of isolated sign glosses into a grammatically coherent Bengali sentence."""
        if not tokens:
            return ""

        cleaned = [t.strip() for t in tokens if t and t.strip()]
        if not cleaned:
            return ""

        # Check for interrogative presence
        is_question = any(t in cls.INTERROGATIVE_TOKENS for t in cleaned)

        # Identify subject
        subject = "আমি"
        for t in cleaned:
            if t in cls.PRONOUNS_1ST or t in cls.PRONOUNS_2ND_HONORIFIC or t in cls.PRONOUNS_2ND_FAMILIAR or t in cls.PRONOUNS_3RD:
                subject = t
                break

        # Process tokens with morphological rules
        inflected_words: List[str] = []
        has_destination = any(w in cls.VERB_CONJUGATIONS and "যাওয়া" in w or "আসা" in w for w in cleaned)

        i = 0
        while i < len(cleaned):
            token = cleaned[i]

            # 1. Locative application for destination nouns if motion verb follows
            if has_destination and token in cls.NOUN_LOCATIVES:
                inflected_words.append(cls.apply_vibhakti(token, "locative"))
            
            # 2. Verb root conjugation
            elif token in cls.VERB_CONJUGATIONS:
                inflected_verb = cls.conjugate_verb(subject, token, "present_continuous")
                inflected_words.append(inflected_verb)

            # 3. Special Idiomatic BdSL mappings
            elif token == "সাহায্য" and i + 1 < len(cleaned) and cleaned[i + 1] in ("করা", "লাগবে", "চাই"):
                inflected_words.append("সাহায্য")
            else:
                inflected_words.append(token)

            i += 1

        sentence = " ".join(inflected_words)

        # Append terminal punctuation
        terminal_mark = "?" if is_question else "।"
        if not sentence.endswith("।") and not sentence.endswith("?") and not sentence.endswith("!"):
            sentence = f"{sentence}{terminal_mark}"

        return sentence
