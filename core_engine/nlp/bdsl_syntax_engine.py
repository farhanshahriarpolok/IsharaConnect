"""Tier 3 & 4: BdSL Visual-Spatial Syntax & Grammar Transformation Engine.

Implements the 4 foundational BdSL transformation rules:
1. Post-Nominal Adjective Inversion (Adj + N -> N + Adj)
2. Terminal Interrogative Displacement (Wh-words placed at terminal position)
3. Conjunction and Particle Deletion (omits 'এবং', 'কিন্তু', 'অথবা', etc.)
4. Semantic Compounding / Unpacking (e.g. হোটেল -> [খাওয়া, টাকা], অ্যাম্বুলেন্স -> [হাসপাতাল, গাড়ি])

Provides bidirectional transformations: Natural Bengali <-> Syntactic BdSL Gloss Sequence.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Lexical inventories for syntax parsing
ADJECTIVES: Set[str] = {
    "ভালো", "খারাপ", "বড়", "ছোট", "সুন্দর", "নতুন", "পুরাতন", "লাল", "নীল", "সবুজ",
    "সাদা", "কালো", "গরম", "ঠান্ডা", "ধনী", "গরীব", "অসুস্থ", "জরুরি", "তাড়াতাড়ি",
    "দেরি", "কঠিন", "সহজ", "বেশি", "কম", "অনেক", "একটু", "উঁচু", "নিচু"
}

NOUNS: Set[str] = {
    "ছেলে", "মেয়ে", "মানুষ", "বাড়ি", "ঘর", "স্কুল", "বই", "গাড়ি", "ফুল", "পানি",
    "টাকা", "ডাক্তার", "হাসপাতাল", "মা", "বাবা", "ভাই", "বোন", "বন্ধু", "শিক্ষক",
    "পুলিশ", "ব্যাংক", "আগুন", "ভাত", "চা", "নাম", "খাবার", "ওষুধ", "দেশ", "শহর"
}

WH_WORDS: Set[str] = {
    "কি", "কী", "কেন", "কোথায়", "কখন", "কে", "কাকে", "কার", "কিভাবে", "কেমন"
}

CONJUNCTIONS_AND_PARTICLES: Set[str] = {
    "এবং", "কিন্তু", "অথবা", "বা", "ও", "আর", "যে", "সুতরাং", "তবে", "নাকি",
    "তাহলে", "তো", "ই", "ওটা", "এটা", "টি", "টা"
}

# Semantic Compounding & Unpacking Dictionary (Bengali Compound -> BdSL Constituent Glosses)
SEMANTIC_COMPOUNDS: Dict[str, List[str]] = {
    "হোটেল": ["খাওয়া", "টাকা"],
    "অ্যাম্বুলেন্স": ["হাসপাতাল", "গাড়ি"],
    "ডাক্তারখানা": ["ডাক্তার", "ঘর"],
    "পাঠশালা": ["পড়া", "ঘর"],
    "চিকিৎসা": ["ডাক্তার", "সাহায্য"],
    "রেস্তোরাঁ": ["খাওয়া", "ঘর"],
    "রান্নাঘর": ["খাওয়া", "ঘর"],
    "বিদ্যুৎ": ["আগুন", "আলো"],
}

# Reverse Unpacking Dictionary for Gloss -> Bengali Synthesis
REVERSE_COMPOUNDS: Dict[Tuple[str, ...], str] = {
    ("খাওয়া", "টাকা"): "হোটেল",
    ("হাসপাতাল", "গাড়ি"): "অ্যাম্বুলেন্স",
    ("ডাক্তার", "ঘর"): "ডাক্তারখানা",
    ("পড়া", "ঘর"): "পাঠশালা",
    ("ডাক্তার", "সাহায্য"): "চিকিৎসা",
}

# Lemma / Root Stem Normalizer
STEM_DICTIONARY: Dict[str, str] = {
    "খাচ্ছি": "খাওয়া", "খেয়েছি": "খাওয়া", "খাবো": "খাওয়া", "খায়": "খাওয়া", "খেয়েছে": "খাওয়া", "খাচ্ছো": "খাওয়া", "খেয়েছেন": "খাওয়া",
    "যাচ্ছি": "যাওয়া", "গিয়েছি": "যাওয়া", "যাবো": "যাওয়া", "যায়": "যাওয়া", "গেছে": "যাওয়া", "যাচ্ছো": "যাওয়া", "গিয়েছেন": "যাওয়া",
    "আসছি": "আসা", "এসেছি": "আসা", "আসবো": "আসা", "আসে": "আসা", "এসেছে": "আসা", "এসেছো": "আসা", "এসেছেন": "আসা", "আসুন": "আসা",
    "পড়ছি": "পড়া", "পড়েছি": "পড়া", "পড়বো": "পড়া", "পড়ে": "পড়া", "পড়াশোনা": "পড়া", "পড়ছেন": "পড়া", "পড়ছো": "পড়া",
    "ঘুমাচ্ছি": "ঘুমানো", "ঘুমিয়েছি": "ঘুমানো", "ঘুমাবো": "ঘুমানো", "ঘুমায়": "ঘুমানো", "ঘুমিয়েছেন": "ঘুমানো",
    "করছি": "করা", "করেছি": "করা", "করবো": "করা", "করে": "করা", "করছেন": "করা", "করছো": "করা",
    "দিচ্ছি": "দেওয়া", "দিয়েছি": "দেওয়া", "দেবো": "দেওয়া", "দেয়": "দেওয়া", "দিয়েছেন": "দেওয়া",
    "নিচ্ছি": "নেওয়া", "নিয়েছি": "নেওয়া", "নেবো": "নেওয়া", "নেয়": "নেওয়া", "নিয়েছেন": "নেওয়া",
}


class BdSLSyntaxEngine:
    """Tier 3 & 4 Visual-Spatial Grammar Engine."""

    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        """Cleans and tokenizes natural Bengali text."""
        cleaned = re.sub(r"[।?!,;:\"'\(\)\[\]]", " ", text)
        return [t.strip() for t in cleaned.split() if t.strip()]

    def text_to_bdsl_gloss(self, bengali_text: str) -> Dict[str, Any]:
        """Transforms natural spoken Bengali sentence into syntactically aligned BdSL gloss sequence.

        Applies:
        1. Semantic Unpacking
        2. Conjunction & Particle Deletion
        3. Lemma Stemming
        4. Post-Nominal Adjective Inversion (Adj + N -> N + Adj)
        5. Terminal Interrogative Displacement (Wh-movement to sentence end)
        """
        raw_tokens = self.tokenize(bengali_text)
        if not raw_tokens:
            return {
                "glosses": [],
                "facs_nmm": [],
                "applied_rules": [],
                "estimated_duration_s": 0.0,
                "is_interrogative": False
            }

        applied_rules: List[str] = []
        is_interrogative = any(t in WH_WORDS for t in raw_tokens) or "?" in bengali_text

        # Step 1: Semantic Compounding / Unpacking
        unpacked_tokens: List[str] = []
        for tok in raw_tokens:
            if tok in SEMANTIC_COMPOUNDS:
                constituents = SEMANTIC_COMPOUNDS[tok]
                unpacked_tokens.extend(constituents)
                applied_rules.append(f"SemanticUnpacking({tok} -> {'+'.join(constituents)})")
            else:
                unpacked_tokens.append(tok)

        # Step 2: Conjunction and Particle Deletion
        filtered_tokens: List[str] = []
        for tok in unpacked_tokens:
            if tok in CONJUNCTIONS_AND_PARTICLES:
                applied_rules.append(f"ParticleDeletion({tok})")
            else:
                filtered_tokens.append(tok)

        # Step 3: Verb Stemming / Normalization
        stemmed_tokens: List[str] = []
        for tok in filtered_tokens:
            stem = STEM_DICTIONARY.get(tok, tok)
            stemmed_tokens.append(stem)

        # Step 4: Post-Nominal Adjective Inversion (Adj + N -> N + Adj)
        i = 0
        inverted_tokens: List[str] = []
        while i < len(stemmed_tokens):
            curr = stemmed_tokens[i]
            if i + 1 < len(stemmed_tokens):
                nxt = stemmed_tokens[i + 1]
                if curr in ADJECTIVES and (nxt in NOUNS or nxt not in ADJECTIVES):
                    inverted_tokens.append(nxt)
                    inverted_tokens.append(curr)
                    applied_rules.append(f"PostNominalAdjectiveInversion({curr} {nxt} -> {nxt} {curr})")
                    i += 2
                    continue
            inverted_tokens.append(curr)
            i += 1

        # Step 5: Terminal Interrogative Displacement
        wh_found: List[str] = []
        non_wh_tokens: List[str] = []
        for tok in inverted_tokens:
            if tok in WH_WORDS:
                wh_found.append(tok)
            else:
                non_wh_tokens.append(tok)

        if wh_found:
            final_glosses = non_wh_tokens + wh_found
            applied_rules.append(f"TerminalInterrogativeDisplacement({','.join(wh_found)})")
        else:
            final_glosses = non_wh_tokens

        # Step 6: Generate Facial Action Unit (NMM) sequence
        facs_nmm: List[Dict[str, float]] = []
        for gloss in final_glosses:
            if is_interrogative:
                # Wh-question facial marker (Brow Lowerer AU04 + Head Tilt)
                facs_nmm.append({"AU01": 0.0, "AU02": 0.0, "AU04": 0.6, "AU12": 0.0, "AU25": 0.2, "head_pitch": -4.0})
            elif gloss in ["ভালো", "ধন্যবাদ", "স্বাগতম", "সুন্দর"]:
                # Positive marker (Smile AU12 + Cheek Raiser AU06)
                facs_nmm.append({"AU01": 0.0, "AU02": 0.0, "AU04": 0.0, "AU06": 0.4, "AU12": 0.7, "head_pitch": 3.0})
            elif gloss in ["অসুস্থ", "ব্যথা", "আগুন", "বন্যা", "সাবধান"]:
                # Negative / Alert marker (Brow furrow AU04)
                facs_nmm.append({"AU01": 0.3, "AU02": 0.0, "AU04": 0.7, "AU12": 0.0, "head_pitch": -2.0})
            else:
                facs_nmm.append({"AU01": 0.0, "AU02": 0.0, "AU04": 0.0, "AU12": 0.2, "head_pitch": 0.0})

        # Calculate estimated duration (approx 650ms per gloss)
        est_duration = max(0.65, len(final_glosses) * 0.65)

        return {
            "glosses": final_glosses,
            "facs_nmm": facs_nmm,
            "applied_rules": applied_rules,
            "estimated_duration_s": round(est_duration, 2),
            "is_interrogative": is_interrogative
        }

    def bdsl_gloss_to_text(self, gloss_list: List[str]) -> Dict[str, Any]:
        """Converts raw BdSL gloss stream back into natural, inflected Bengali sentences.

        Re-synthesizes compound constituents, restores standard SOV order, and adds appropriate punctuation.
        """
        if not gloss_list:
            return {"bengali": "", "confidence": 0.0, "is_interrogative": False}

        cleaned = [g.strip() for g in gloss_list if g and g.strip()]
        if not cleaned:
            return {"bengali": "", "confidence": 0.0, "is_interrogative": False}

        # 1. Reverse Compound Packing
        packed_tokens: List[str] = []
        idx = 0
        while idx < len(cleaned):
            if idx + 1 < len(cleaned):
                pair = (cleaned[idx], cleaned[idx + 1])
                if pair in REVERSE_COMPOUNDS:
                    packed_tokens.append(REVERSE_COMPOUNDS[pair])
                    idx += 2
                    continue
            packed_tokens.append(cleaned[idx])
            idx += 1

        is_question = any(t in WH_WORDS for t in packed_tokens)

        # 2. Re-order Adjective + Noun if Noun + Adj occurred
        reordered: List[str] = []
        i = 0
        while i < len(packed_tokens):
            curr = packed_tokens[i]
            if i + 1 < len(packed_tokens):
                nxt = packed_tokens[i + 1]
                if curr in NOUNS and nxt in ADJECTIVES:
                    reordered.append(nxt)
                    reordered.append(curr)
                    i += 2
                    continue
            reordered.append(curr)
            i += 1

        # 3. Simple inflection generation
        # e.g., ["আমি", "ভাত", "খাওয়া"] -> "আমি ভাত খাচ্ছি।"
        sentence_words: List[str] = []
        for t in reordered:
            if t == "খাওয়া":
                sentence_words.append("খাচ্ছি" if "আমি" in reordered else "খাচ্ছে")
            elif t == "যাওয়া":
                sentence_words.append("যাবো" if "আমি" in reordered else "যাচ্ছে")
            elif t == "আসা":
                sentence_words.append("এসেছি" if "আমি" in reordered else "এসেছে")
            elif t == "পড়া":
                sentence_words.append("পড়ছি" if "আমি" in reordered else "পড়ছে")
            elif t == "ঘুমানো":
                sentence_words.append("ঘুমাচ্ছি" if "আমি" in reordered else "ঘুমাচ্ছে")
            else:
                sentence_words.append(t)

        mark = "?" if is_question else "।"
        sentence = " ".join(sentence_words) + mark

        return {
            "bengali": sentence,
            "tokens": sentence_words,
            "confidence": 0.95,
            "is_interrogative": is_question
        }


# Module singleton
bdsl_syntax_engine = BdSLSyntaxEngine()
