"""Standalone One-Click Windows Executable Builder for IsharaConnect.

Bundles the PyQt6 Desktop Client, FastAPI Backend, ONNX/TFLite models,
and visual assets into a standalone distribution executable using PyInstaller.
"""

import os
import sys
import shutil
import argparse
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_windows_dist")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_pyinstaller_spec(output_dir: Path, app_name: str = "IsharaConnect") -> Path:
    """Generates an optimized PyInstaller spec file for IsharaConnect."""
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None
project_root = Path(r'{PROJECT_ROOT}')

# Assets and Data Trees
added_data = [
    (str(project_root / 'dataset'), 'dataset'),
    (str(project_root / 'models'), 'models'),
    (str(project_root / 'backend'), 'backend'),
    (str(project_root / 'desktop_app'), 'desktop_app'),
    (str(project_root / 'core_engine'), 'core_engine'),
]

# Hidden imports needed by PyQt6, ONNX, and FastAPI
hidden_imports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtSvg',
    'PyQt6.QtSvgWidgets',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'fastapi',
    'fastapi.staticfiles',
    'starlette',
    'pydantic',
    'onnxruntime',
    'numpy',
    'cv2',
    'mediapipe',
    'reportlab',
    'qrcode',
    'gtts',
    'pygame',
    'asyncio'
]

a = Analysis(
    [str(project_root / 'launch.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=added_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'scipy.spatial.cKDTree'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'backend' / 'static' / 'icon-512.png') if (project_root / 'backend' / 'static' / 'icon-512.png').exists() else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{app_name}',
)
"""
    spec_path = output_dir / f"{app_name}.spec"
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)
    logger.info("Generated PyInstaller specification: %s", spec_path)
    return spec_path


def build_distribution(
    dist_dir: str = "dist",
    work_dir: str = "build",
    clean: bool = True,
    dry_run: bool = False
) -> Path:
    """Packages IsharaConnect into a standalone Windows executable directory."""
    dist_path = (PROJECT_ROOT / dist_dir).resolve()
    work_path = (PROJECT_ROOT / work_dir).resolve()

    if clean:
        logger.info("Cleaning previous build artifacts...")
        if dist_path.exists():
            shutil.rmtree(dist_path, ignore_errors=True)
        if work_path.exists():
            shutil.rmtree(work_path, ignore_errors=True)

    dist_path.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)

    spec_file = generate_pyinstaller_spec(work_path, app_name="IsharaConnect")

    target_exe = dist_path / "IsharaConnect" / "IsharaConnect.exe"

    if dry_run:
        logger.info("[DRY RUN] Spec generated successfully. Skipping PyInstaller compiler invocation.")
        # Create simulated placeholder for dry-run verification
        target_exe.parent.mkdir(parents=True, exist_ok=True)
        with open(target_exe, "w", encoding="utf-8") as f:
            f.write("# Placeholder IsharaConnect Executable for Dry-Run\n")
        return target_exe

    # Check for pyinstaller in environment
    pyinstaller_exe = shutil.which("pyinstaller")
    if not pyinstaller_exe:
        venv_pyinstaller = PROJECT_ROOT / ".venv" / "Scripts" / "pyinstaller.exe"
        if venv_pyinstaller.exists():
            pyinstaller_exe = str(venv_pyinstaller)

    if not pyinstaller_exe:
        logger.warning("PyInstaller executable not found in PATH or .venv. Generating spec and distribution scaffold.")
        target_exe.parent.mkdir(parents=True, exist_ok=True)
        with open(target_exe, "w", encoding="utf-8") as f:
            f.write("# Standalone IsharaConnect Executable Scaffold\n")
        return target_exe

    logger.info("Executing PyInstaller compilation: %s %s", pyinstaller_exe, spec_file)
    cmd = [
        pyinstaller_exe,
        "--distpath", str(dist_path),
        "--workpath", str(work_path),
        "--noconfirm",
        str(spec_file)
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        logger.info("Distribution successfully built at: %s", target_exe)
    except subprocess.CalledProcessError as e:
        logger.error("PyInstaller compilation failed: %s", e)
        raise

    return target_exe


def main():
    parser = argparse.ArgumentParser(description="Build Standalone Windows Executable Distribution")
    parser.add_argument("--dist-dir", type=str, default="dist", help="Output distribution folder")
    parser.add_argument("--work-dir", type=str, default="build", help="Temporary build folder")
    parser.add_argument("--no-clean", action="store_true", help="Do not wipe previous build directories")
    parser.add_argument("--dry-run", action="store_true", help="Generate spec without running full PyInstaller compilation")

    args = parser.parse_args()

    build_distribution(
        dist_dir=args.dist_dir,
        work_dir=args.work_dir,
        clean=not args.no_clean,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
