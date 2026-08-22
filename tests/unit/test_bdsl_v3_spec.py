"""Unit tests for BdSL v3 Parametric Kinematics, Phonetics & FACS Schema."""

import json
from pathlib import Path
import pytest
from core_engine.dsl.bdsl_v3_spec import BdSLV3SignSpec


def test_bdsl_v3_sign_spec_validation():
    """Test loading and validating complete BdSL v3 sign specification."""
    spec_path = Path(__file__).resolve().parents[2] / "data" / "signs" / "BDSL_V3_00104_dhonnobad.json"
    assert spec_path.exists()

    with open(spec_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    spec = BdSLV3SignSpec(**raw_data)
    assert spec.sign_id == "BDSL_V3_00104"
    assert spec.gloss_bn == "ধন্যবাদ"
    assert spec.phonetics.stokoe_notation == "⫸𝄆√"
    assert spec.phonetics.handshape_code == "HS_FLAT_BENT_THUMB"
    assert spec.kinematics.start_anchor.body_part == "CHIN"
    assert spec.kinematics.end_anchor.body_part == "MID_CHEST"
    assert spec.facial_action_units.AU12_lip_corner_puller == 0.85
    assert spec.contact_physics.has_contact is True
    assert spec.temporal_phases_ms.total_ms == 950
    assert spec.morphosyntax.pos == "INTERJECTION"


def test_bdsl_v3_stokoe_summary():
    """Test phonetic representation string format."""
    spec_path = Path(__file__).resolve().parents[2] / "data" / "signs" / "BDSL_V3_00104_dhonnobad.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = BdSLV3SignSpec(**json.load(f))

    summary = spec.get_stokoe_summary()
    assert "ধন্যবাদ" in summary
    assert "⫸𝄆√" in summary
    assert "HS_FLAT_BENT_THUMB" in summary


def test_bdsl_v3_bezier_trajectory_computation():
    """Test calculating 3D Bézier trajectory curve points."""
    spec_path = Path(__file__).resolve().parents[2] / "data" / "signs" / "BDSL_V3_00104_dhonnobad.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = BdSLV3SignSpec(**json.load(f))

    trajectory = spec.compute_bezier_trajectory(num_samples=30)
    assert len(trajectory) == 30
    assert len(trajectory[0]) == 3  # [x, y, z]

    # Start point matches start_anchor offset
    assert trajectory[0] == spec.kinematics.start_anchor.offset_cm
    # End point matches end_anchor offset
    assert trajectory[-1] == spec.kinematics.end_anchor.offset_cm
