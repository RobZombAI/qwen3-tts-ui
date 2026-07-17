# Qwen3 TTS — Desktop Application

**🎤 Multi-platform TTS desktop app powered by Qwen3-TTS models.  
Custom voices, voice cloning, style control — all running locally on your machine.**

---

## ✨ Features

- **🎤 CustomVoice TTS** — 9 premium voices across Chinese, English, Japanese, and Korean, with natural-language instruction control for tone, emotion, and speaking rate
- **🧬 Voice Cloning** — Clone any voice from a 3-second audio sample (x-vector mode) or optionally pair with a transcript for higher fidelity (ICL mode)
- **⚡ Fast Mode** — 0.6B parameter model for 3–4× faster generation, available as a toggle
- **🌍 10 Languages** — Italian, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish (each speaker can speak any language)
- **🎯 Style Control** — Natural-language instructions like *"Speak happily and excitedly"* or *"Use a calm, professional tone"*
- **✂️ Audio Trimmer** — Upload reference audio up to 60+ seconds; interactive waveform with slider to select the best segment for cloning
- **💾 Voice Profiles** — Save cloned voices for reuse across sessions (audio, settings, and metadata persisted to disk)
- **📁 Configurable Output** — Choose where generated audio is saved; preference persists across sessions
- **🖥️ Native Desktop** — macOS native app (WebKit window, no browser needed); Windows support via PyInstaller .exe

---

## 📸 Screenshots

<p align="center">
  <img src="screenshots/main_window.png" width="700" alt="Main application window">
</p>
<p align="center">
  <img src="screenshots/voice_clone.png" width="700" alt="Voice cloning interface">
</p>

---

## 🚀 Quick Start

### macOS

```bash
# Clone or download
cd ~/qwen3-tts-ui

# Set up environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -U pip
pip install -U qwen-tts
pip install flask pywebview soundfile pillow

# Install audio tools
brew install sox

# Launch the native app
python3 app_native.py
```

Or double-click **`start.command`** from Finder.

> **First launch:** The app downloads model weights (~3.5 GB each) from HuggingFace. Models are cached after the first download.

### Windows

```bat
cd C:\Users\YOURNAME\qwen3-tts-ui
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -U qwen-tts
pip install flask pywebview soundfile pillow
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python win_launcher.py
```

See [Windows Setup Guide](docs/WINDOWS_SETUP.md) for CUDA setup and standalone .exe builds.

---

## 🎯 Usage

### Select a Model

| Model | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| **CustomVoice 1.7B** | 🌟 Best | ~14× RTF | Production quality, maximum fidelity |
| **Fast Mode 0.6B** | 👍 Excellent | ~4× RTF | Quick iterations, lower-resource machines |
| **Voice Clone** | 🌟 Best | ~15× RTF | Cloning any voice from an audio sample |

### Generate Speech with CustomVoice

1. Click **CustomVoice** in the model selector
2. Toggle **Fast Mode** for the 0.6B model if desired
3. Select **Language** and **Speaker** (hover to see descriptions)
4. Enter text in the synthesis area
5. Optionally add a **Style / Emotion** instruction
6. Click **⚡ Generate Speech**

### Clone a Voice

1. Click **Voice Clone** in the model selector
2. Upload a reference audio file (3–10 seconds recommended)
3. If the audio is longer than 60 seconds, use the **trim slider** to select a segment
4. Optionally enable **ICL Mode** and provide a transcript for better quality
5. Enter the text to synthesize
6. Click **⚡ Generate Speech**
7. After generation, click **💾 Save Voice** to keep the profile for reuse

---

## 🧪 Running Tests

```bash
cd ~/qwen3-tts-ui
source venv/bin/activate
python test_suite.py
```

The test suite covers:
- Server startup and graceful shutdown
- All API endpoints
- Model loading (all 3 variants)
- Speech generation (multiple languages, speakers)
- Voice cloning (x-vector mode)
- Audio upload and waveform generation
- Profile save/load/delete
- Output directory configuration
- Device compatibility checking

---

## 📁 Project Structure

```
qwen3-tts-ui/
├── qwen3_tts_server.py      # Flask server + embedded HTML/JS UI
├── app_native.py             # macOS native window launcher (pywebview)
├── win_launcher.py           # Windows entry point
├── test_suite.py             # Comprehensive test suite
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # Apache 2.0
├── CONTRIBUTING.md           # Contribution guidelines
├── CHANGELOG.md              # Version history
├── docs/
│   ├── macOS_SETUP.md        # Detailed macOS setup
│   ├── WINDOWS_SETUP.md      # Windows + CUDA + .exe build
│   ├── ANDROID_GUIDE.md      # Android compatibility analysis
│   ├── API_REFERENCE.md      # Full API documentation
│   └── ARCHITECTURE.md       # System architecture
├── scripts/
│   ├── start.command          # macOS Finder launcher
│   ├── start.bat              # Windows launcher
│   └── build_exe.bat          # Windows PyInstaller build
├── screenshots/               # App screenshots
└── icon.icns                  # macOS app icon
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Native Window                      │
│  ┌───────────────────────────────────────────────┐  │
│  │            WebKit / WebView                   │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │         HTML + CSS + JavaScript UI       │  │  │
│  │  └─────────────────────┬───────────────────┘  │  │
│  └────────────────────────┼──────────────────────┘  │
└───────────────────────────┼─────────────────────────┘
                            │ HTTP localhost:8765
┌───────────────────────────┼─────────────────────────┐
│              Flask Server (background thread)       │
│  ┌────────────────────┐  │  ┌────────────────────┐  │
│  │  REST API          │  │  │  Model Manager     │  │
│  │  • /api/generate   │──┼─→│  • cv_1b7 (1.7B)   │  │
│  │  • /api/speakers   │  │  │  • cv_0b6 (0.6B)   │  │
│  │  • /api/profiles   │  │  │  • base (1.7B)     │  │
│  │  • /api/output_dir │  │  └────────┬───────────┘  │
│  │  • /api/check_device│  │           │              │
│  └────────────────────┘  │  ┌────────┴───────────┐  │
│                          │  │  PyTorch + MPS/CUDA │  │
│                          │  └────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Prerequisites

| Platform | Requirements |
|----------|--------------|
| **macOS** | Apple Silicon (M1/M2/M3/M4/M5) or Intel, macOS 14+ recommended |
| **Windows** | 64-bit, Windows 10+, CUDA-capable GPU recommended |
| **RAM** | 8 GB minimum, 16 GB+ recommended |
| **Storage** | 10 GB free for models + dependencies |
| **Python** | 3.10 – 3.12 |

---

## 📜 License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).

The underlying Qwen3-TTS models by Alibaba Cloud are also Apache 2.0 licensed.

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🙏 Acknowledgments

- **[Qwen Team (Alibaba Cloud)](https://huggingface.co/Qwen)** — for the Qwen3-TTS models
- **[HuggingFace](https://huggingface.co)** — model hosting and transformers library
- **[pywebview](https://pywebview.flowrl.com)** — native WebView window for Python
- **[CustomTkinter](https://customtkinter.tomschimansky.com)** — initial GUI prototyping

---

<p align="center">
  <sub>Built with ❤️ for the open-source community</sub>
</p>
