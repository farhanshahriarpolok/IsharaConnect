"""IsharaConnect Windows Desktop App Installer & Standalone Packager."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def build_installer():
    """Compiles IsharaConnect into a standalone distribution directory using PyInstaller."""
    try:
        import PyInstaller.__main__
    except ImportError:
        print("PyInstaller is not installed in the active environment.")
        return

    icon_path = BASE_DIR / "assets" / "icons" / "icon.ico"
    args = [
        str(BASE_DIR / "desktop_app" / "main.py"),
        "--name=IsharaConnect",
        "--onedir",
        "--windowed",
        f"--add-data={BASE_DIR / 'dataset' / 'lexicon'};dataset/lexicon",
        f"--add-data={BASE_DIR / 'backend' / 'static'};backend/static",
        f"--add-data={BASE_DIR / 'backend' / 'templates'};backend/templates",
        f"--add-data={BASE_DIR / 'backend' / 'models'};backend/models",
        "--hidden-import=uvicorn",
        "--hidden-import=aiortc",
        "--hidden-import=onnxruntime",
        "--hidden-import=PyQt6.QtWebEngineWidgets",
        "-y"
    ]

    if icon_path.exists():
        args.insert(-1, f"--icon={icon_path}")

    print(f"Executing PyInstaller with args: {args}")
    PyInstaller.__main__.run(args)


if __name__ == "__main__":
    build_installer()
