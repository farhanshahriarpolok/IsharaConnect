"""BdSL Gloss Translator powered by AdvancedBdSLGrammarEngine."""

import logging
from typing import List, Dict, Any
from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine

logger = logging.getLogger(__name__)


class BdSLGlossTranslator(AdvancedBdSLGrammarEngine):
    """Translates a sequence of isolated BdSL glosses into grammatically correct Bengali and English sentences.

    Extends AdvancedBdSLGrammarEngine to provide full morphological inflections and contextual smoothing.
    """
    pass
