"""Unit tests for MultiSignSequenceBlender pipeline."""

import json
from pathlib import Path
import pytest
from core_engine.dsl.sequence_blender import MultiSignSequenceBlender


def test_sequence_blender_pipeline():
    """Test full multi-sign sequence blending pipeline."""
    spec_path = Path(__file__).resolve().parents[2] / "data" / "signs" / "BDSL_V3_00104_dhonnobad.json"
    assert spec_path.exists()

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    blender = MultiSignSequenceBlender(fps=30, transition_ms=150)
    stream = blender.blend_sentence_stream([spec, spec])

    assert len(stream) > 0
    # ট্রানজিশন ফ্রেমের উপস্থিতি পরীক্ষা
    transitions = [f for f in stream if f.get("is_transition")]
    assert len(transitions) == blender.transition_frames

    # ধারাবাহিক ইনডেক্সিং এবং হ্যান্ড জয়েন্ট ভেরিফিকেশন
    for idx, frame in enumerate(stream):
        assert frame["frame_idx"] == idx
        assert "right_wrist" in frame
        assert len(frame["right_hand"]) == 21
        assert "facs" in frame
