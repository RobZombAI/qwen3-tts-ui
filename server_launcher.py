#!/usr/bin/env python3
"""
Qwen3 TTS — Server Launcher (reliable browser mode)
Starts the server and opens the default browser.
100% reliable — no WebKit/WebView dependencies.

Usage:
    source venv/bin/activate && python3 server_launcher.py
"""

import os
import sys
import time
import threading
import webbrowser
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PORT = 8765


def main():
    print(f"\n🎙️  Qwen3 TTS — Starting server...")
    print(f"   URL: http://localhost:{PORT}")
    print(f"   Close this window to stop the server.")
    print()

    # Import and start Flask server
    import importlib.util
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    spec = importlib.util.spec_from_file_location(
        "qwen_tts_server",
        os.path.join(os.path.dirname(__file__), "qwen3_tts_server.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = mod.app

    t = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1", port=PORT,
            debug=False, threaded=True, use_reloader=False,
        ),
        daemon=True,
    )
    t.start()

    # Wait for server
    import urllib.request
    import json as _json
    for i in range(30):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
            d = _json.loads(r.read())
            if d.get("status") == "alive":
                print(f"   ✅ Server ready!")
                break
        except Exception:
            pass
        time.sleep(0.5)

    # Open browser
    webbrowser.open(f"http://localhost:{PORT}")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print("   Shutting down...")


if __name__ == "__main__":
    main()
