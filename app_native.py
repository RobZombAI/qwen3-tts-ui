#!/usr/bin/env python3
"""
Qwen3 TTS — Native macOS App (WebKit window, no browser)
=========================================================
Truly standalone: one window, no browser tab.
Close the window → app shuts down cleanly → memory freed.

Usage:
    cd /Applications/Qwen3 TTS.app/Contents/Resources
    ./venv/bin/python3 app_native.py
"""

import os, sys, time, threading, logging, json, urllib.request

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PORT = 8765

def start_server():
    """Start Flask in background, wait for it to be ready."""
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "srv", os.path.join(os.path.dirname(__file__), "server_native.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t = threading.Thread(
        target=lambda: mod.app.run(host="127.0.0.1", port=PORT,
                                   debug=False, threaded=True, use_reloader=False),
        daemon=True
    )
    t.start()

    # Wait for server
    for i in range(60):
        try:
            r = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/api/health", timeout=2).read())
            if r.get("status") == "alive":
                return mod
        except: pass
        time.sleep(0.5)
    return mod

def main():
    print("🎙️  Qwen3 TTS — Native App")
    print("   Starting server...")

    mod = start_server()
    print("   ✅ Server ready, opening window...")

    import webview

    # API exposed to JS
    class API:
        def shutdown(self):
            """Called from JS when user clicks Quit."""
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/api/shutdown", timeout=2)
            except: pass
            time.sleep(0.3)
            os._exit(0)

    api = API()

    # Create native window
    window = webview.create_window(
        title="Qwen3 TTS — CustomVoice + Voice Clone",
        url=f"http://127.0.0.1:{PORT}",
        width=860, height=920,
        min_size=(700, 700),
        resizable=True,
        js_api=api,
        confirm_close=True,
    )

    webview.start(debug=False, http_server=False)

    # Window closed — cleanup
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/shutdown", timeout=2)
    except: pass
    time.sleep(0.5)
    # Free MPS
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except: pass
    os._exit(0)


if __name__ == "__main__":
    main()
