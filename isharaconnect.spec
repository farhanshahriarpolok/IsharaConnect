# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None
project_root = Path(SPECPATH).resolve() if 'SPECPATH' in globals() else Path('.').resolve()

# Ensure runtime directories exist
(project_root / 'certificates').mkdir(exist_ok=True)
(project_root / 'dataset' / 'audio_cache').mkdir(parents=True, exist_ok=True)
(project_root / 'models').mkdir(exist_ok=True)

# Assets and Data Trees
added_data = [
    (str(project_root / 'dataset'), 'dataset'),
    (str(project_root / 'models'), 'models'),
    (str(project_root / 'backend' / 'templates'), 'backend/templates'),
    (str(project_root / 'backend' / 'static'), 'backend/static'),
    (str(project_root / 'backend'), 'backend'),
    (str(project_root / 'desktop_app'), 'desktop_app'),
    (str(project_root / 'core_engine'), 'core_engine'),
    (str(project_root / 'certificates'), 'certificates'),
]

# Hidden imports needed by PyQt6, ONNX, FastAPI, SQLAlchemy, Redis, and Cryptography
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
    'fastapi.templating',
    'jinja2',
    'jinja2.ext',
    'starlette',
    'pydantic',
    'pydantic_settings',
    'sqlalchemy',
    'sqlalchemy.ext.asyncio',
    'sqlalchemy.dialects.sqlite.aiosqlite',
    'aiosqlite',
    'passlib',
    'passlib.handlers.bcrypt',
    'bcrypt',
    'jose',
    'jose.jwt',
    'redis',
    'redis.asyncio',
    'websockets',
    'websockets.legacy',
    'onnxruntime',
    'numpy',
    'cv2',
    'mediapipe',
    'reportlab',
    'reportlab.lib.pagesizes',
    'reportlab.platypus',
    'reportlab.lib.colors',
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
