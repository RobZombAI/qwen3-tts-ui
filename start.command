#!/bin/bash
# Qwen3 TTS — Native App Launcher (no browser needed)
# Starts the app in a native macOS WebKit window.
# Works with or without the .app bundle.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source "$DIR/venv/bin/activate"
python3 app_native.py
