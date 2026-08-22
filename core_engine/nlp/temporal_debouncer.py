"""Temporal Gloss Debouncer & Continuous Gesture Stream Segmenter.

Suppresses transitional jitter and duplicate frame glitches using sliding windows,
minimum consecutive frame voting, and pause duration phrase boundary detection.
"""

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TemporalGlossDebouncer:
    """Stabilizes real-time continuous sign recognition output streams."""

    IDLE_TOKENS = {"", "none", "idle", "rest", "background", "null", "noise"}

    def __init__(
        self,
        window_size: int = 20,
        min_consecutive: int = 3,
        confidence_thresh: float = 0.65,
        pause_threshold_s: float = 1.2,
    ):
        self.window_size = window_size
        self.min_consecutive = min_consecutive
        self.confidence_thresh = confidence_thresh
        self.pause_threshold_s = pause_threshold_s

        self._window: Deque[Tuple[str, float, float]] = deque(maxlen=window_size)
        self.stable_tokens: List[str] = []
        self.last_active_time: float = time.time()
        self.last_emitted_token: Optional[str] = None
        self._sentence_boundary_triggered = False

    def add_prediction(
        self,
        sign_slug: str,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> Optional[str]:
        """Ingests a single-frame sign prediction, debounces jitter, and detects stable tokens.

        Returns newly stabilized token if threshold reached, else None.
        """
        now = timestamp if timestamp is not None else time.time()
        cleaned_slug = sign_slug.strip().lower() if sign_slug else ""

        # Check for pause / idle boundary
        if cleaned_slug in self.IDLE_TOKENS or confidence < self.confidence_thresh:
            if self.stable_tokens and (now - self.last_active_time) >= self.pause_threshold_s:
                self._sentence_boundary_triggered = True
            return None

        self._window.append((sign_slug.strip(), confidence, now))
        self.last_active_time = now
        self._sentence_boundary_triggered = False

        # Check for min_consecutive occurrences of the same token in recent window
        recent_items = list(self._window)[-self.min_consecutive:]
        if len(recent_items) == self.min_consecutive:
            target_slug = recent_items[0][0]
            if all(item[0] == target_slug and item[1] >= self.confidence_thresh for item in recent_items):
                if target_slug != self.last_emitted_token:
                    self.last_emitted_token = target_slug
                    self.stable_tokens.append(target_slug)
                    logger.debug("Debounced new stable token: %s", target_slug)
                    return target_slug

        return None

    def is_sentence_boundary(self) -> bool:
        """Returns True if user has paused long enough to trigger a phrase/sentence flush."""
        return self._sentence_boundary_triggered

    def get_stable_tokens(self) -> List[str]:
        """Returns the list of stabilized tokens accumulated in the current phrase buffer."""
        return list(self.stable_tokens)

    def flush(self) -> List[str]:
        """Flushes and returns stabilized tokens, resetting current phrase buffer."""
        flushed = list(self.stable_tokens)
        self.reset()
        return flushed

    def reset(self):
        """Resets the debouncer internal state."""
        self._window.clear()
        self.stable_tokens.clear()
        self.last_emitted_token = None
        self._sentence_boundary_triggered = False
        self.last_active_time = time.time()
