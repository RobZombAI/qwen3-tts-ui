# Qwen3 TTS — macOS Setup Guide

## Prerequisites

- macOS 14+ (Sonoma/Sequoia)
- Apple Silicon (M1–M5) preferred; Intel supported
- Python 3.10 – 3.12
- 10 GB free storage
- 8 GB+ RAM (16 GB+ recommended)

## Installation

### 1. Download the App

**Option A: Use the pre-built .app bundle**
```
~/Desktop/Qwen3 TTS.app  →  drag to /Applications
```

**Option B: Run from source**
```bash
cd ~/qwen3-tts-ui
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -U qwen-tts
pip install flask pywebview soundfile pillow
brew install sox
```

### 2. Launch

**From the .app:**
Double-click `Qwen3 TTS.app` in /Applications or ~/Desktop.

**From terminal:**
```bash
cd ~/qwen3-tts-ui && source venv/bin/activate && python3 app_native.py
```

**From Finder (standalone script):**
Double-click `start.command`.

### 3. First Launch

On first launch, the app checks your system:
```
✅ MPS (Apple Silicon) — full speed
```

Then click a model name to load it:
- **CustomVoice** → loads 1.7B HQ or 0.6B Fast depending on toggle
- **Voice Clone** → loads Base model for voice cloning

Model weights (~3.5 GB each) are downloaded from HuggingFace automatically.

### Uninstall

```bash
rm -rf ~/qwen3-tts-ui
rm -rf ~/.qwen3tts_config.json
rm -rf ~/.qwen3tts_profiles
rm -rf /Applications/Qwen3\ TTS.app
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't open | Right-click → Open (Gatekeeper override) |
| Window is blank | Check `http://localhost:8765` in Safari |
| Model download slow | First load needs ~3.5 GB; subsequent loads instant |
| Voice cloning fails | Use shorter audio (3-10s) with clear speech |
| "flash-attn not installed" | Normal on MPS; SDPA fallback is used automatically |
| Tcl/Tk crash | The native app uses WebKit, not Tk. Use `app_native.py`. |
