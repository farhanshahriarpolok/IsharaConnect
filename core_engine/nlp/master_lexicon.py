"""Tier 2: Master BdSL Lexicon Query & Schema Engine.

Provides unified runtime querying, category indexing, and kinematic profile retrieval
for standardized Bangladesh Sign Language (BdSL) signs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEXICON_JSON_PATH = Path(__file__).resolve().parents[2] / "dataset" / "lexicon" / "master_bdsl_lexicon.json"


class MasterBdSLLexicon:
    """In-memory indexing and query interface for the Master BdSL Lexical Database."""

    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = json_path or LEXICON_JSON_PATH
        self.signs_by_slug: Dict[str, Dict[str, Any]] = {}
        self.signs_by_bn: Dict[str, Dict[str, Any]] = {}
        self.signs_by_category: Dict[str, List[Dict[str, Any]]] = {}
        self._load_lexicon()

    def _load_lexicon(self):
        """Loads and indexes the master lexicon from JSON."""
        if not self.json_path.exists():
            logger.warning("Lexicon file %s not found. Initializing empty fallback.", self.json_path)
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sign in data.get("signs", []):
                slug = sign.get("slug", "")
                bn = sign.get("label_bn", "").strip()
                cat = sign.get("category", "General")

                if slug:
                    self.signs_by_slug[slug] = sign
                if bn:
                    self.signs_by_bn[bn] = sign

                if cat not in self.signs_by_category:
                    self.signs_by_category[cat] = []
                self.signs_by_category[cat].append(sign)

            logger.info("Loaded %d master BdSL signs from %s", len(self.signs_by_slug), self.json_path)
        except Exception as e:
            logger.error("Failed to load master BdSL lexicon: %s", e)

    def get_sign_by_gloss(self, gloss: str) -> Optional[Dict[str, Any]]:
        """Resolves sign metadata by Bengali gloss, slug, or English label."""
        if not gloss:
            return None
        clean = gloss.strip()
        
        # 1. Exact Bengali match
        if clean in self.signs_by_bn:
            return self.signs_by_bn[clean]
        
        # 2. Exact slug match
        clean_slug = clean.lower().replace(" ", "_")
        if clean_slug in self.signs_by_slug:
            return self.signs_by_slug[clean_slug]

        # 3. Fuzzy search in English names or slugs
        for s in self.signs_by_slug.values():
            if s.get("label_en", "").lower() == clean.lower():
                return s
            if clean in s.get("label_bn", "") or s.get("label_bn", "") in clean:
                return s

        return None

    def get_signs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Returns all signs in a given domain category."""
        return self.signs_by_category.get(category, [])

    def get_kinematic_profile(self, gloss: str) -> Optional[Dict[str, Any]]:
        """Returns the 3D Bézier anchors, contact physics, and FACS vectors for a sign."""
        sign = self.get_sign_by_gloss(gloss)
        if not sign:
            return None

        return {
            "slug": sign.get("slug"),
            "label_bn": sign.get("label_bn"),
            "handshape": sign.get("handshape"),
            "stokoe_notation": sign.get("stokoe_notation"),
            "bezier_anchors_3d": sign.get("bezier_anchors_3d", {}),
            "facs_action_units": sign.get("facs_action_units", {}),
            "contact_physics": sign.get("contact_physics", {}),
            "timing_ms": sign.get("timing_ms", {})
        }

    def all_signs(self) -> List[Dict[str, Any]]:
        """Returns list of all signs in the database."""
        return list(self.signs_by_slug.values())

    def search_signs(self, query: str) -> List[Dict[str, Any]]:
        """Searches signs across Bengali, English, and category fields."""
        q = query.strip().lower()
        results = []
        for s in self.signs_by_slug.values():
            if (
                q in s.get("label_bn", "").lower()
                or q in s.get("label_en", "").lower()
                or q in s.get("slug", "").lower()
                or q in s.get("category", "").lower()
            ):
                results.append(s)
        return results


# Module-level singleton
master_lexicon = MasterBdSLLexicon()
