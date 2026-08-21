"""Unit tests for Windows Executable Spec and Build Packaging Scripts."""

from pathlib import Path
import pytest

from scripts.build_windows_dist import generate_pyinstaller_spec
from scripts.build_windows_exe import build_executable


def test_pyinstaller_spec_generation(tmp_path):
    """Spec generation creates valid PyInstaller .spec file with correct added data."""
    spec_path = generate_pyinstaller_spec(tmp_path, app_name="IsharaConnectTest")
    assert spec_path.exists()

    content = spec_path.read_text(encoding="utf-8")
    assert "dataset" in content
    assert "models" in content
    assert "desktop_app" in content
    assert "PyQt6" in content
    assert "onnxruntime" in content


def test_build_executable_dry_run(tmp_path):
    """build_executable in dry-run mode validates spec and creates placeholder executable scaffold."""
    dist_dir = tmp_path / "test_dist"
    work_dir = tmp_path / "test_build"

    exe_path = build_executable(
        dist_dir=str(dist_dir),
        work_dir=str(work_dir),
        clean=True,
        dry_run=True
    )
    assert exe_path.exists()
    assert "IsharaConnect" in str(exe_path)
