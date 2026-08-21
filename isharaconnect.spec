# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None
project_root = Path(__file__).resolve().parent

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
    hooksconfig={},
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
    name='IsharaConnect',
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
    name='IsharaConnect',
)
