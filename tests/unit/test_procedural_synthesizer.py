"""Unit tests for Procedural HyperKinematicSynthesizer."""

import json
from pathlib import Path
import pytest
from core_engine.dsl.procedural_synthesizer import HyperKinematicSynthesizer


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
