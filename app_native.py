#!/usr/bin/env python3
"""
Qwen3 TTS — Native macOS App
==============================
Starts the TTS web server in a background thread and opens
a native macOS window (WebKit) — no browser tab needed.

Clean shutdown: when the window closes, MPS memory is freed,
the server stops, and all resources are released.

Usage:
    source venv/bin/activate && python app_native.py

For .app bundle: used as entry point by Qwen3 TTS.app
"""

import os
import sys
import threading
import time
import signal

# ── Imports ──────────────────────────────────────────────────────────────────

# Quiet noisy libraries before importing anything
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WERKZEUG_RUN_MAIN"] = "false"

import webview
import logging
import subprocess

# ── Config ───────────────────────────────────────────────────────────────────

PORT = 8765
WINDOW_TITLE = "Qwen3 TTS — CustomVoice + Voice Clone"
WINDOW_WIDTH = 840
WINDOW_HEIGHT = 920
WINDOW_MIN_WIDTH = 700
WINDOW_MIN_HEIGHT = 700

SHUTDOWN_TIMEOUT = 5  # max seconds to wait for graceful shutdown


# ── Server management ────────────────────────────────────────────────────────

_server_thread = None
_server_mod = None


def start_server():
    """Start the Flask server in a background thread."""
    global _server_thread, _server_mod

    # Suppress Flask's default output
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Import the Flask app from qwen3_tts_server.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen_tts_server",
        os.path.join(os.path.dirname(__file__), "qwen3_tts_server.py")
    )
    _server_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_server_mod)
    app = _server_mod.app

    _server_thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1",
            port=PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        ),
        daemon=True,
    )
    _server_thread.start()


def stop_server():
    """Graceful shutdown: free MPS memory, stop server, exit cleanly.

    The OS reclaims all memory (RAM + GPU/Metal) when the process exits,
    but we proactively clear MPS cache to make it instant.
    """
    import urllib.request

    print("   Shutting down gracefully…")

    # 1. Tell the Flask server to shutdown (this also clears MPS cache)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/shutdown",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass

    # 2. Give it a moment to clean up
    time.sleep(0.3)

    # 3. Final MPS cleanup from our side too
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass

    # 4. Exit cleanly — os._exit(0) is necessary here because:
    #    - Flask's dev server doesn't have a clean shutdown API
    #    - The daemon thread for Flask would keep the process alive
    #    - os._exit(0) reclaims ALL memory (RAM, MPS, GPU) immediately
    print("   Memory released. Goodbye!")
    os._exit(0)


# ── Native JS API (exposed to frontend via pywebview) ────────────────────────


class WebViewAPI:
    """Methods callable from JavaScript via pywebview.api.*()"""

    def pick_folder(self) -> str:
        """Open native macOS folder picker. Returns selected path or empty string."""
        try:
            # Use osascript for native folder picker dialog
            script = """
            tell application "System Events"
                activate
                set theFolder to choose folder with prompt "Choose output folder for generated audio:"
                return POSIX path of theFolder
            end tell
            """
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    return path
        except Exception as e:
            print(f"   Folder picker error: {e}")
        return ""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"🎙️  Qwen3 TTS — Native macOS App")
    print(f"   Starting server on http://localhost:{PORT}")
    print()

    # Start the Flask server in background
    start_server()

    # Wait for server to be fully ready (health endpoint)
    import urllib.request
    import json as _json
    for i in range(60):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
            data = _json.loads(r.read())
            if data.get("status") == "alive":
                print(f"   Server started (PID: {os.getpid()})")
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        print("   ⚠️ Server may not have started — opening window anyway…")

    print(f"   Opening native window ({WINDOW_WIDTH}x{WINDOW_HEIGHT})…")
    print(f"   Close the window to quit — memory is freed automatically.")
    print()

    # Create native window with webview (blocks until window closes)
    api = WebViewAPI()

    webview.create_window(
        title=WINDOW_TITLE,
        url=f"http://127.0.0.1:{PORT}",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT),
        resizable=True,
        fullscreen=False,
        js_api=api,
        confirm_close=True,
    )

    # Run the native event loop (blocks until window closes)
    webview.start(debug=False, http_server=False)

    # Cleanup after window closes
    stop_server()


if __name__ == "__main__":
    main()
