"""CTC Beam Search Decoder with Bengali Language Model Rescoring.

Implements prefix-tree beam search decoding over CTC logit outputs with:
  - Full prefix merging (blank-split and non-blank merge)
  - Configurable beam width (default W=10)
  - Language model rescoring via insertion bonus (β) and LM weight (α)
  - Bengali N-gram dictionary transition scorer
  - Drop-in replacement for greedy argmax in CSLROnnxEngine

References:
  Graves et al. 2006 — Connectionist Temporal Classification
  Hannun et al. 2014 — Deep Speech: Scaling up end-to-end speech recognition

Usage:
    from core_engine.inference.ctc_beam_decoder import CTCBeamSearchDecoder, ctc_beam_search_decode

    decoder = CTCBeamSearchDecoder(vocab=vocab, beam_width=10, alpha=0.6, beta=1.2)
    gloss, conf = decoder.decode(logits_T_C)
"""

import math
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Bengali CSLR vocabulary aligned with CSLROnnxEngine
# ---------------------------------------------------------------------------
DEFAULT_BDSL_VOCAB: List[str] = [
    "<blank>", "আমি", "তুমি", "আপনি", "স্কুল", "যাওয়া", "আসা", "খাওয়া",
    "পানি", "চা", "কফি", "দুধ", "ধন্যবাদ", "সালাম", "ডাক্তার", "হাসপাতাল",
    "অসুস্থ", "জরুরি", "সাহায্য", "ভূমিকম্প", "যানজট", "মা", "বাবা", "ভাই", "বোন",
    "কেমন", "আছেন", "হ্যাঁ", "না", "ভালো", "খারাপ",
    "বড়", "ছোট", "গরম", "ঠান্ডা", "বাড়ি", "বাজার", "কাজ",
]

# ---------------------------------------------------------------------------
# Bengali N-gram Language Model Scorer
# ---------------------------------------------------------------------------

# Bigram log-probability table (based on common Bengali sign sequences).
# Only high-frequency transitions are listed; unseen transitions use a small
# negative log-prob (uniform backoff).
_BENGALI_BIGRAM_LOG_PROBS: Dict[Tuple[str, str], float] = {
    ("আমি", "স্কুল"): -0.5,
    ("আমি", "যাওয়া"): -0.6,
    ("আমি", "খাওয়া"): -0.7,
    ("আমি", "আসা"):   -0.8,
    ("আমি", "ভালো"):  -0.9,
    ("স্কুল", "যাওয়া"): -0.4,
    ("জরুরি", "সাহায্য"): -0.3,
    ("সাহায্য", "ডাক্তার"): -0.4,
    ("ডাক্তার", "হাসপাতাল"): -0.5,
    ("মা", "বাবা"): -0.4,
    ("বাবা", "ভালো"): -0.6,
    ("ধন্যবাদ",): -0.2,   # unigram bonus — always a legal terminal
    ("কেমন", "আছেন"): -0.3,
    ("পানি", "খাওয়া"): -0.5,
    ("চা", "খাওয়া"): -0.5,
}
_BIGRAM_BACKOFF: float = -3.5   # log-prob for unseen transitions


class BengaliNgramLMScorer:
    """Lightweight Bengali N-gram language model scorer for CTC beam rescoring.

    Scores a proposed next token given the current prefix sequence using
    a static bigram log-probability table with uniform backoff.
    """

    def __init__(
        self,
        bigram_table: Optional[Dict] = None,
        backoff_log_prob: float = _BIGRAM_BACKOFF,
        vocab_set: Optional[set] = None
    ):
        self.bigram_table = bigram_table or _BENGALI_BIGRAM_LOG_PROBS
        self.backoff = backoff_log_prob
        self.vocab_set = vocab_set or set()

    def score(self, prefix_tokens: List[str], next_token: str) -> float:
        """Returns log P(next_token | last token in prefix) using bigram backoff.

        Args:
            prefix_tokens: Current decoded token sequence (may be empty).
            next_token: Proposed next gloss token.

        Returns:
            Log-probability (negative float). Closer to 0 = more probable.
        """
        if not prefix_tokens:
            # Unigram score
            key = (next_token,)
            return self.bigram_table.get(key, self.backoff)
        last = prefix_tokens[-1]
        key = (last, next_token)
        return self.bigram_table.get(key, self.backoff)

    def word_insertion_bonus(self, token: str) -> float:
        """Returns a small bonus for each successfully inserted word (β factor)."""
        return 0.0 if token == "<blank>" else 1.0


# ---------------------------------------------------------------------------
# Beam State
# ---------------------------------------------------------------------------

class _BeamState:
    """Represents a single beam hypothesis during CTC beam search."""

    __slots__ = ("tokens", "p_blank", "p_nblank", "lm_score")

    def __init__(
        self,
        tokens: Tuple[str, ...] = (),
        p_blank: float = 0.0,      # log-prob of ending in blank
        p_nblank: float = -1e38,   # log-prob of ending in non-blank
        lm_score: float = 0.0
    ):
        self.tokens = tokens
        self.p_blank = p_blank
        self.p_nblank = p_nblank
        self.lm_score = lm_score

    @property
    def total_log_prob(self) -> float:
        """Log-sum-exp of blank and non-blank path probabilities."""
        a, b = self.p_blank, self.p_nblank
        if a == -1e38:
            return b
        if b == -1e38:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    def score_with_lm(self, alpha: float, beta: float) -> float:
        """Total score = acoustic log-prob + α * LM log-prob + β * |tokens|."""
        return self.total_log_prob + alpha * self.lm_score + beta * len(self.tokens)

    def __repr__(self) -> str:
        return f"Beam({' '.join(self.tokens) or '<empty>'}, p={self.total_log_prob:.3f})"


# ---------------------------------------------------------------------------
# Core Beam Search
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log-domain addition: log(exp(a) + exp(b))."""
    if a == -1e38:
        return b
    if b == -1e38:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_beam_search_decode(
    logits: np.ndarray,
    vocab: List[str],
    beam_width: int = 10,
    alpha: float = 0.6,
    beta: float = 1.2,
    lm_scorer: Optional[BengaliNgramLMScorer] = None,
    blank_idx: int = 0
) -> Tuple[str, float]:
    """CTC Beam Search Decoder with optional LM rescoring.

    Args:
        logits: Raw network output of shape (T, C) — unnormalized log-probs or probs.
                If values appear to be probabilities (max ≤ 1, min ≥ 0), they are used
                directly; otherwise log-softmax is applied.
        vocab: Vocabulary list where vocab[0] == '<blank>'.
        beam_width: Maximum number of active beams (W). Default 10.
        alpha: LM weight for rescoring. Default 0.6.
        beta: Word insertion bonus weight. Default 1.2.
        lm_scorer: Optional Bengali N-gram scorer. None = no LM rescoring.
        blank_idx: Index of the CTC blank token. Default 0.

    Returns:
        (best_gloss_sequence, confidence_score)
            best_gloss_sequence: Space-joined token string.
            confidence_score: Normalized beam probability in [0, 1].
    """
    T, C = logits.shape
    if C != len(vocab):
        logger.warning(
            "Vocab size mismatch: logits C=%d vs vocab len=%d. Truncating.", C, len(vocab)
        )
        C = min(C, len(vocab))
        logits = logits[:, :C]

    # Convert to log-probabilities
    if logits.max() > 1.0 or logits.min() < 0.0:
        # Assume unnormalized logits → log-softmax
        log_probs = logits - np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True) + logits.max(axis=-1, keepdims=True)
        # Simpler: just use scipy/manual log-softmax
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    else:
        log_probs = np.log(np.clip(logits, 1e-10, 1.0))

    # Initialize beam: single empty prefix
    beams: Dict[Tuple[str, ...], _BeamState] = {
        (): _BeamState(tokens=(), p_blank=0.0, p_nblank=-1e38)
    }

    for t in range(T):
        t_log_probs = log_probs[t]  # shape (C,)
        new_beams: Dict[Tuple[str, ...], _BeamState] = {}

        for prefix, state in beams.items():
            p_total = state.total_log_prob

            # --- Extend with blank ---
            blank_lp = float(t_log_probs[blank_idx])
            if prefix not in new_beams:
                new_beams[prefix] = _BeamState(
                    tokens=prefix, p_blank=-1e38, p_nblank=-1e38, lm_score=state.lm_score
                )
            new_beams[prefix].p_blank = _log_add(
                new_beams[prefix].p_blank,
                p_total + blank_lp
            )

            # --- Extend with each non-blank token ---
            for c in range(C):
                if c == blank_idx:
                    continue
                token = vocab[c]
                lp = float(t_log_probs[c])

                new_prefix = prefix
                last_token = prefix[-1] if prefix else None

                if token == last_token:
                    # Same as last token: only extend from blank-ending path
                    new_p = state.p_blank + lp
                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = _BeamState(
                            tokens=new_prefix, p_blank=-1e38, p_nblank=-1e38,
                            lm_score=state.lm_score
                        )
                    new_beams[new_prefix].p_nblank = _log_add(
                        new_beams[new_prefix].p_nblank, new_p
                    )
                else:
                    # New token: extend prefix
                    extended = prefix + (token,)
                    lm_delta = 0.0
                    if lm_scorer is not None:
                        lm_delta = (alpha * lm_scorer.score(list(prefix), token)
                                    + beta * lm_scorer.word_insertion_bonus(token))

                    new_lm = state.lm_score + lm_delta

                    if extended not in new_beams:
                        new_beams[extended] = _BeamState(
                            tokens=extended, p_blank=-1e38, p_nblank=-1e38,
                            lm_score=new_lm
                        )
                    else:
                        # Merge LM scores (take max as approximation)
                        new_beams[extended].lm_score = max(new_beams[extended].lm_score, new_lm)

                    new_beams[extended].p_nblank = _log_add(
                        new_beams[extended].p_nblank, p_total + lp
                    )

        # Prune to top beam_width by combined score
        sorted_beams = sorted(
            new_beams.values(),
            key=lambda s: s.score_with_lm(alpha, beta),
            reverse=True
        )
        beams = {s.tokens: s for s in sorted_beams[:beam_width]}

    if not beams:
        return "", 0.0

    # Select best beam
    best = max(beams.values(), key=lambda s: s.score_with_lm(alpha, beta))
    gloss = " ".join(best.tokens) if best.tokens else ""

    # Normalize confidence to [0, 1] via sigmoid of total log-prob
    raw_log_prob = best.total_log_prob
    confidence = float(1.0 / (1.0 + math.exp(-raw_log_prob / max(T, 1))))
    confidence = min(max(confidence, 0.0), 1.0)

    return gloss, confidence


# ---------------------------------------------------------------------------
# High-level decoder class (used by CSLROnnxEngine)
# ---------------------------------------------------------------------------

class CTCBeamSearchDecoder:
    """High-level CTC Beam Search Decoder with integrated Bengali LM rescoring.

    Drop-in replacement for the greedy argmax decoder inside CSLROnnxEngine.

    Args:
        vocab: Vocabulary list (vocab[0] == '<blank>').
        beam_width: Number of active beams. Default 10.
        alpha: LM log-probability weight. Default 0.6.
        beta: Word insertion bonus weight. Default 1.2.
        use_lm: Whether to activate Bengali N-gram LM rescoring. Default True.
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        beam_width: int = 10,
        alpha: float = 0.6,
        beta: float = 1.2,
        use_lm: bool = True
    ):
        self.vocab = vocab or DEFAULT_BDSL_VOCAB
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_scorer = BengaliNgramLMScorer(vocab_set=set(self.vocab)) if use_lm else None

    def decode(self, logits_T_C: np.ndarray) -> Tuple[str, float]:
        """Decodes a (T, C) logit array into a gloss sequence with confidence.

        Args:
            logits_T_C: Shape (T, C). May be logits or probabilities.

        Returns:
            (gloss_sequence, confidence): str, float in [0, 1].
        """
        if logits_T_C.ndim != 2:
            raise ValueError(f"Expected 2D logits (T, C), got shape {logits_T_C.shape}")
        return ctc_beam_search_decode(
            logits=logits_T_C,
            vocab=self.vocab,
            beam_width=self.beam_width,
            alpha=self.alpha,
            beta=self.beta,
            lm_scorer=self.lm_scorer
        )

    def decode_batch(
        self,
        logits_B_T_C: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Decodes a batch of (B, T, C) logit sequences.

        Returns:
            List of (gloss, confidence) tuples, one per batch item.
        """
        if logits_B_T_C.ndim == 2:
            return [self.decode(logits_B_T_C)]
        results = []
        for b in range(logits_B_T_C.shape[0]):
            results.append(self.decode(logits_B_T_C[b]))
        return results

    def update_vocab(self, vocab: List[str]) -> None:
        """Hot-swaps the vocabulary and refreshes the LM scorer."""
        self.vocab = vocab
        if self.lm_scorer is not None:
            self.lm_scorer.vocab_set = set(vocab)
