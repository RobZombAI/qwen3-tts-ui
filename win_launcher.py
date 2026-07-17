#!/usr/bin/env python3
"""
Qwen3 TTS — Windows Launcher
==============================
Standalone Windows .exe entry point.
Starts the Flask server and opens a native WebView window.

For PyInstaller packaging:
    pip install pywebview pyinstaller flask
    pyinstaller --windowed --onefile --name "Qwen3 TTS" ^
        --add-data "qwen3_tts_server.py;." ^
        --hidden-import "qwen_tts" ^
        --hidden-import "soundfile" ^
        --hidden-import "flask" ^
        --hidden-import "webview" ^
        win_launcher.py
"""

import os
import sys
import json
import time
import threading
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

# Quiet noisy libs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WERKZEUG_RUN_MAIN"] = "false"

import webview
import logging

# ── Config ───────────────────────────────────────────────────────────────────

PORT = 8765
WINDOW_TITLE = "Qwen3 TTS — CustomVoice + Voice Clone"
WINDOW_WIDTH = 840
WINDOW_HEIGHT = 920
WINDOW_MIN_WIDTH = 700
WINDOW_MIN_HEIGHT = 700

# On Windows, user data goes to %APPDATA%/Qwen3TTS
if sys.platform == "win32":
    _appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
    CONFIG_DIR = Path(_appdata) / "Qwen3TTS"
else:
    CONFIG_DIR = Path.home() / "qwen3-tts-ui"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path(tempfile.gettempdir()) / "qwen3tts_clone"
TEMP_DIR.mkdir(exist_ok=True)
PROFILES_DIR = CONFIG_DIR / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)


# ── Compatibility Check ──────────────────────────────────────────────────────

def check_compatibility() -> dict:
    """Check if Windows system can run the model. Returns report dict."""
    import torch
    report = {
        "platform": sys.platform,
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "ram_gb": 0,
        "verdict": "",
        "recommended_model": "cv_0b6",  # conservative default
    }

    # Check RAM
    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        report["ram_gb"] = 0

    ram = report["ram_gb"]
    cuda = report["cuda_available"]

    if cuda and ram >= 16:
        report["verdict"] = "✅ Full support — CUDA GPU + 16GB+ RAM"
        report["recommended_model"] = "cv_1b7"
    elif cuda and ram >= 8:
        report["verdict"] = "⚠️ GPU available but limited RAM — use Fast Mode (0.6B)"
        report["recommended_model"] = "cv_0b6"
    elif not cuda and ram >= 32:
        report["verdict"] = "⚠️ No GPU detected — CPU mode. 32GB+ RAM OK but slow (~30× RTF)"
        report["recommended_model"] = "cv_0b6"
    elif not cuda and ram >= 16:
        report["verdict"] = "⚠️ No GPU, limited RAM — Fast Mode only (0.6B), expect ~50× RTF"
        report["recommended_model"] = "cv_0b6"
    else:
        report["verdict"] = "❌ Device may be too weak. Minimum: 8GB RAM, GPU recommended"
        report["recommended_model"] = None

    return report


# ── Server Management ────────────────────────────────────────────────────────

def start_server():
    """Start the Flask server in background thread."""
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    import importlib.util
    server_path = os.path.join(os.path.dirname(__file__), "qwen3_tts_server.py")
    
    # PyInstaller: when frozen, files are in _MEIPASS
    if getattr(sys, 'frozen', False):
        server_path = os.path.join(sys._MEIPASS, "qwen3_tts_server.py")

    spec = importlib.util.spec_from_file_location("qwen_tts_server", server_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Override TEMP_DIR and PROFILES_DIR for Windows
    mod.TEMP_DIR = str(TEMP_DIR)
    mod.PROFILES_DIR = str(PROFILES_DIR)
    os.makedirs(mod.TEMP_DIR, exist_ok=True)
    os.makedirs(mod.PROFILES_DIR, exist_ok=True)

    t = threading.Thread(
        target=lambda: mod.app.run(
            host="127.0.0.1", port=PORT,
            debug=False, threaded=True, use_reloader=False,
        ),
        daemon=True,
    )
    t.start()
    return mod


def stop_server():
    """Graceful shutdown."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/shutdown",
            method="POST", data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass
    time.sleep(0.5)
    os._exit(0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Check compatibility first
    print("🔍 Qwen3 TTS — Windows Launcher")
    print()
    print("   Checking system compatibility…")
    report = check_compatibility()

    print(f"   Platform: {report['platform']}")
    print(f"   PyTorch:  {report['torch_version']}")
    print(f"   CUDA:     {'✅ Available' if report['cuda_available'] else '❌ Not available'}")
    print(f"   RAM:      {report['ram_gb']} GB")
    print(f"   Verdict:  {report['verdict']}")
    print()

    if report["recommended_model"] is None:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0,
            "Your system may not have enough resources to run Qwen3 TTS models.\n\n"
            f"Detected: {report['ram_gb']}GB RAM\n"
            f"CUDA GPU: {'Yes' if report['cuda_available'] else 'No'}\n\n"
            "Minimum requirements: 8GB RAM, CUDA GPU recommended.\n\n"
            "You can still try with Fast Mode (0.6B) on CPU.",
            "Qwen3 TTS — Compatibility Warning", 0x30)

    print("   Starting server…")
    mod = start_server()

    # Wait for server
    for i in range(30):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2)
            break
        except Exception:
            time.sleep(1)

    print(f"   Opening window ({WINDOW_WIDTH}x{WINDOW_HEIGHT})…")
    print("   Close the window to quit.")
    print()

    api = type('WebViewAPI', (), {'pick_folder': lambda self: _pick_folder()})()

    webview.create_window(
        title=WINDOW_TITLE,
        url=f"http://127.0.0.1:{PORT}",
        width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
        min_size=(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT),
        resizable=True, js_api=api, confirm_close=True,
    )

    webview.start(debug=False, http_server=False)
    stop_server()


def _pick_folder() -> str:
    """Native Windows folder picker."""
    try:
        import ctypes
        from ctypes import wintypes
        # Use folder dialog via shell32
        folder = ctypes.create_unicode_buffer(260)
        pidl = ctypes.windll.shell32.SHBrowseForFolderW(None, None, "Choose output folder for generated audio:", 0, None, None)
        if pidl:
            ctypes.windll.shell32.SHGetPathFromIDListW(pidl, folder)
            return folder.value
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    main()
