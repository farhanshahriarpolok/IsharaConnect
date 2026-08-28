"""Coarticulated Sentence Synthesizer.

Consumes a CoarticulatedSentencePlan and produces a single continuous
kinematic frame stream suitable for driving the ToonAvatarRenderer or
any downstream 3D avatar engine.

Key responsibilities:
  - Per-transition blend timing (variable blend_ms from coarticulation map)
  - Spatial pause hold frames between non-slerp transitions
  - Per-frame NMM FACS AU injection from the sentence expression timeline
  - Global frame-index and timestamp reassignment after stitching
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core_engine.dsl.procedural_synthesizer import (
    HyperKinematicSynthesizer,
    MultiSignSequenceBlender,
)
from core_engine.nlp.sentence_to_gloss_pipeline import (
    CoarticulatedSentencePlan,
    GlossTransitionSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal per-gloss sign stub
# ---------------------------------------------------------------------------

def _make_stub_sign_spec(gloss: str, duration_ms: int = 550) -> Dict[str, Any]:
    """Build a minimal BdSL v3-compatible sign spec dict for a gloss token.

    In production this is replaced by a full MasterBdSLLexicon lookup.
    The stub keeps the synthesizer pipeline runnable for glosses that are not
    yet in the lexicon (e.g. temporal markers, spatial adverbs).
    """
    return {
        "sign_id": f"STUB_{gloss}",
        "gloss_bn": gloss,
        "gloss_en": gloss,
        "phonetics": {"handshape_code": "HS_FLAT_BENT_THUMB"},
        "kinematics": {
            "start_anchor": {"body_part": "MID_CHEST", "offset_cm": [0.0, 0.0, 5.0]},
            "end_anchor":   {"body_part": "MID_CHEST", "offset_cm": [0.0, 2.0, 20.0]},
            "velocity_profile": {"peak_velocity_ms": 1.0, "ease_type": "CUBIC_OUT"},
        },
        "facial_action_units": {
            "AU12_lip_corner_puller": 0.0,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
        "temporal_phases_ms": {"total_ms": max(200, duration_ms)},
    }


# ---------------------------------------------------------------------------
# Pause-frame generator
# ---------------------------------------------------------------------------

def _generate_pause_frames(
    reference_frame: Dict[str, Any],
    pause_ms: int,
    fps: int,
) -> List[Dict[str, Any]]:
    """Repeat the last frame N times to realise a spatial hold."""
    pause_frames = max(1, int((pause_ms / 1000.0) * fps))
    result = []
    for _ in range(pause_frames):
        f = dict(reference_frame)
        f["is_transition"] = False
        f["is_pause"] = True
        result.append(f)
    return result


# ---------------------------------------------------------------------------
# Main synthesizer
# ---------------------------------------------------------------------------

class CoarticulatedSentenceSynthesizer:
    """Renders a CoarticulatedSentencePlan into a continuous kinematic frame stream.

    Parameters
    ----------
    fps : int
        Output frame rate (default 60).
    lexicon : optional callable
        A function ``(gloss: str) -> Optional[Dict]`` that returns a BdSL v3
        sign spec for the given gloss token. Falls back to stub specs when None.
    """

    def __init__(
        self,
        fps: int = 60,
        lexicon: Optional[Any] = None,
    ) -> None:
        self.fps = fps
        self.lexicon = lexicon
        self._synth = HyperKinematicSynthesizer(fps=fps)
        self._blender = MultiSignSequenceBlender(fps=fps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, plan: CoarticulatedSentencePlan) -> List[Dict[str, Any]]:
        """Convert a CoarticulatedSentencePlan into a frame-by-frame motion stream.

        Returns
        -------
        List[Dict]
            Each dict contains:
                frame_idx    : int
                timestamp_ms : int
                right_wrist  : [x, y, z]
                right_hand   : [[x,y,z] × 21]
                facs         : {AU_key: float, ...}
                is_transition: bool
                is_pause     : bool   (True during spatial holds)
        """
        if not plan.gloss_sequence:
            logger.warning("Empty gloss sequence — returning empty frame stream.")
            return []

        all_frames: List[Dict[str, Any]] = []
        glosses = plan.gloss_sequence

        for i, gloss in enumerate(glosses):
            # ---- Generate frames for this sign -------------------------
            sign_spec = self._resolve_sign(gloss)
            sign_frames = self._synth.generate_trajectory_frames(sign_spec)

            # ---- Blend transition INTO this sign -----------------------
            if all_frames and sign_frames:
                transition_spec = plan.get_transition(
                    glosses[i - 1], gloss
                ) if i > 0 else None
                blend_ms = transition_spec.blend_ms if transition_spec else 150
                trans_frames = self._blend_transition(
                    all_frames[-1], sign_frames[0], blend_ms
                )
                all_frames.extend(trans_frames)

            # ---- Append sign frames ------------------------------------
            for sf in sign_frames:
                sf_copy = dict(sf)
                sf_copy["is_transition"] = False
                sf_copy["is_pause"] = False
                all_frames.append(sf_copy)

            # ---- Spatial pause AFTER this sign (if specified) ----------
            if i < len(glosses) - 1:
                next_gloss = glosses[i + 1]
                transition_spec = plan.get_transition(gloss, next_gloss)
                if transition_spec and transition_spec.spatial_pause_ms > 0 and all_frames:
                    pause_frames = _generate_pause_frames(
                        all_frames[-1], transition_spec.spatial_pause_ms, self.fps
                    )
                    all_frames.extend(pause_frames)

        # ---- Inject NMM FACS from expression timeline ------------------
        self._inject_nmm_facs(all_frames, plan)

        # ---- Re-index all frames globally ------------------------------
        for idx, frame in enumerate(all_frames):
            frame["frame_idx"] = idx
            frame["timestamp_ms"] = int((idx / self.fps) * 1000)

        logger.debug(
            "CoarticulatedSentenceSynthesizer: '%s' → %d frames (~%.2fs) for %d glosses",
            plan.template_id,
            len(all_frames),
            len(all_frames) / self.fps,
            len(glosses),
        )
        return all_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sign(self, gloss: str) -> Dict[str, Any]:
        """Look up a BdSL v3 sign spec from the lexicon or fall back to stub."""
        if self.lexicon:
            try:
                spec = self.lexicon(gloss)
                if spec:
                    return spec
            except Exception as exc:
                logger.warning("Lexicon lookup failed for '%s': %s", gloss, exc)
        return _make_stub_sign_spec(gloss)

    def _blend_transition(
        self,
        last_frame: Dict[str, Any],
        first_next_frame: Dict[str, Any],
        blend_ms: int,
    ) -> List[Dict[str, Any]]:
        """Generate Hermite-blended transition frames between two sign boundary frames."""
        blend_frames = max(2, int((blend_ms / 1000.0) * self.fps))
        p_start = np.array(last_frame["right_wrist"], dtype=np.float32)
        p_end = np.array(first_next_frame["right_wrist"], dtype=np.float32)
        v = (p_end - p_start) * 0.25

        hand_start = np.array(last_frame["right_hand"], dtype=np.float32)
        hand_end = np.array(first_next_frame["right_hand"], dtype=np.float32)

        facs_start = last_frame.get("facs", {})
        facs_end = first_next_frame.get("facs", {})

        frames = []
        for i in range(blend_frames):
            t = (i + 1) / (blend_frames + 1)
            t_s = 3 * t**2 - 2 * t**3  # smooth-step ease

            # Hermite position
            h00 = 2 * t_s**3 - 3 * t_s**2 + 1
            h10 = t_s**3 - 2 * t_s**2 + t_s
            h01 = -2 * t_s**3 + 3 * t_s**2
            h11 = t_s**3 - t_s**2
            wrist = h00 * p_start + h10 * v + h01 * p_end + h11 * v

            hand = (1 - t_s) * hand_start + t_s * hand_end

            # Lerp FACS
            au_keys = set(facs_start) | set(facs_end)
            facs_interp: Dict[str, float] = {
                k: round(
                    (1 - t_s) * facs_start.get(k, 0.0) + t_s * facs_end.get(k, 0.0),
                    3,
                )
                for k in au_keys
            }

            frames.append({
                "right_wrist": [round(float(v_), 4) for v_ in wrist.tolist()],
                "right_hand": [[round(float(c), 4) for c in pt] for pt in hand.tolist()],
                "facs": facs_interp,
                "is_transition": True,
                "is_pause": False,
            })

        return frames

    def _inject_nmm_facs(
        self,
        frames: List[Dict[str, Any]],
        plan: CoarticulatedSentencePlan,
    ) -> None:
        """Overwrite or merge per-frame FACS values from the NMM expression timeline."""
        for frame in frames:
            ts = frame.get("timestamp_ms", 0)
            nmm_overrides = plan.get_nmm_at(ts)
            if nmm_overrides:
                current_facs = frame.get("facs", {})
                # NMM timeline takes precedence; merge with existing AU values
                merged = {**current_facs, **nmm_overrides}
                frame["facs"] = merged
