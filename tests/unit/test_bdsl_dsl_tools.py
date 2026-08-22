"""Unit tests for BdSL Parametric DSL and Landmark Retrieval Agent Tools."""

import pytest
from core_engine.dsl.bdsl_tools import (
    load_bdsl_dictionary,
    get_sign_dsl_tool,
    get_sign_landmarks_tool,
)


def test_load_bdsl_dictionary():
    """Test loading dictionary into memory."""
    db = load_bdsl_dictionary()
    assert isinstance(db, dict)
    assert len(db) > 0
    assert "ধন্যবাদ" in db or "dhonnobad" in db


def test_get_sign_dsl_tool_success():
    """Test looking up sign DSL configuration."""
    res = get_sign_dsl_tool("ধন্যবাদ")
    assert res["status"] == "success"
    assert res["gloss"] == "ধন্যবাদ"
    assert "dsl" in res["data"]
    assert res["data"]["dsl"]["hand"] in ["right", "both", "left"]


def test_get_sign_dsl_tool_not_found():
    """Test looking up non-existent sign."""
    res = get_sign_dsl_tool("unknown_nonexistent_sign_xyz")
    assert res["status"] == "not_found"


def test_get_sign_landmarks_tool_success():
    """Test retrieving landmark data for a known sign."""
    res = get_sign_landmarks_tool("dhonnobad")
    assert res["status"] == "success"
    assert "landmarks" in res


def test_get_sign_landmarks_tool_not_found():
    """Test retrieving landmark data for non-existent sign."""
    res = get_sign_landmarks_tool("invalid_sign_12345")
    assert res["status"] == "not_found"
