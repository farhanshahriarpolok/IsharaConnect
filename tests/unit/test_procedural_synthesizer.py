"""Unit tests for Procedural HyperKinematicSynthesizer."""

import json
from pathlib import Path
import pytest
from core_engine.dsl.procedural_synthesizer import (
    HyperKinematicSynthesizer,
    MultiSignSequenceBlender,
)


@pytest.fixture
def dhonnobad_schema():
    spec_path = Path(__file__).resolve().parents[2] / "data" / "signs" / "BDSL_V3_00104_dhonnobad.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hyper_kinematic_synthesizer_initialization():
    """Test synthesizer initialization with custom FPS."""
    synthesizer = HyperKinematicSynthesizer(fps=60)
    assert synthesizer.fps == 60


def test_trajectory_frame_generation(dhonnobad_schema):
    """Test generating frame-by-frame 3D skeletal landmarks from schema."""
    synthesizer = HyperKinematicSynthesizer(fps=30)
    frames = synthesizer.generate_trajectory_frames(dhonnobad_schema)

    # 950ms at 30 fps ~= 28 frames
    assert len(frames) >= 20
    assert frames[0]["frame_idx"] == 0
    assert frames[0]["timestamp_ms"] == 0

    first_frame = frames[0]
    assert "right_wrist" in first_frame
    assert len(first_frame["right_wrist"]) == 3
    assert "right_hand" in first_frame
    assert len(first_frame["right_hand"]) == 21

    # FACS Action Unit checks
    assert "facs" in first_frame
    assert "AU12" in first_frame["facs"]
    assert "head_pitch" in first_frame["facs"]

    # Final frame checks
    last_frame = frames[-1]
    assert last_frame["frame_idx"] == len(frames) - 1
    assert last_frame["facs"]["AU12"] > first_frame["facs"]["AU12"]


def test_multi_sign_sequence_blender_initialization():
    """Test blender initialization with fps and transition parameters."""
    blender = MultiSignSequenceBlender(fps=60, transition_ms=150)
    assert blender.fps == 60
    assert blender.transition_ms == 150
    assert blender.transition_frames >= 2


def test_multi_sign_sequence_blender_transitions(dhonnobad_schema):
    """Test transition generation between consecutive frames."""
    blender = MultiSignSequenceBlender(fps=30, transition_ms=100)
    synthesizer = HyperKinematicSynthesizer(fps=30)
    frames = synthesizer.generate_trajectory_frames(dhonnobad_schema)

    transitions = blender.generate_transition_frames(frames[-1], frames[0])
    assert len(transitions) == blender.transition_frames
    for t_frame in transitions:
        assert t_frame["is_transition"] is True
        assert "right_wrist" in t_frame
        assert len(t_frame["right_wrist"]) == 3
        assert len(t_frame["right_hand"]) == 21
        assert "facs" in t_frame


def test_blend_sentence_stream(dhonnobad_schema):
    """Test full multi-sign sequence blending into continuous motion stream."""
    blender = MultiSignSequenceBlender(fps=30, transition_ms=100)

    # Blend two identical signs in sequence
    stream = blender.blend_sentence_stream([dhonnobad_schema, dhonnobad_schema])
    assert len(stream) > 0

    # Ensure continuous indexing and timestamps
    for i, frame in enumerate(stream):
        assert frame["frame_idx"] == i
        assert frame["timestamp_ms"] == int((i / 30.0) * 1000)

    # Check that transition frames exist in the middle
    transition_frames = [f for f in stream if f.get("is_transition")]
    assert len(transition_frames) == blender.transition_frames


def test_blend_empty_stream():
    """Test blending empty sequence returns empty list."""
    blender = MultiSignSequenceBlender()
    assert blender.blend_sentence_stream([]) == []
