# -*- mode: python ; coding: utf-8 -*-
"""
Qwen3 TTS — PyInstaller .spec for Windows .exe

Build:
    pip install pyinstaller pywebview flask torch transformers soundfile
    pyinstaller Qwen3TTS.spec

Output: dist/Qwen3 TTS.exe
"""

import sys
from pathlib import Path

block_cipher = None

# Collect all data files
datas = [
    ("qwen3_tts_server.py", "."),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    "qwen_tts",
    "qwen_tts.inference.qwen3_tts_model",
    "qwen_tts.inference.qwen3_tts_tokenizer",
    "qwen_tts.core.models",
    "flask",
    "werkzeug",
    "soundfile",
    "numpy",
    "torch",
    "torchaudio",
    "transformers",
    "huggingface_hub",
    "accelerate",
    "einops",
    "librosa",
    "scipy",
    "soxr",
    "safetensors",
    "webview",
    "ctypes",
    "queue",
    "threading",
    "webbrowser",
    "logging",
    "json",
    "time",
    "tempfile",
    "pathlib",
    "shutil",
    "typing",
    "urllib",
]

a = Analysis(
    ['win_launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Qwen3 TTS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
)
