# Qwen3 TTS — Windows (.exe) Setup Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Any (CPU works) | NVIDIA CUDA (RTX 2060+) |
| **Python** | 3.10+ | 3.12 |
| **Storage** | 10 GB free | 15 GB+ |
| **OS** | Windows 10 64-bit | Windows 11 |

## Compatibility Check

Before downloading models, `win_launcher.py` automatically checks:

```
🔍 Qwen3 TTS — Windows Launcher

   Checking system compatibility…
   Platform: win32
   PyTorch:  2.13.0
   CUDA:     ✅ Available (RTX 3060)
   RAM:      32 GB
   Verdict:  ✅ Full support — CUDA GPU + 16GB+ RAM
```

If your system is below minimum, it shows a warning dialog before proceeding.

### What each spec means

| RAM | CUDA GPU | Verdict | Model |
|-----|----------|---------|-------|
| 16GB+ | ✅ Yes | ✅ Full speed | 1.7B HQ |
| 8-16GB | ✅ Yes | ⚠️ Limited RAM | 0.6B Fast |
| 32GB+ | ❌ No | ⚠️ CPU only | 0.6B Fast |
| 8-16GB | ❌ No | ⚠️ Slow CPU | 0.6B Fast |
| <8GB | Either | ❌ Not recommended | — |

## Installation

### Quick start (no build)

```cmd
cd C:\Users\YOURNAME\qwen3-tts-ui

python -m venv venv
venv\Scripts\activate

pip install -U pip
pip install -U qwen-tts
pip install flask pywebview pyinstaller
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

python win_launcher.py
```

### Build standalone .exe

```cmd
pip install pyinstaller
pyinstaller Qwen3TTS.spec
```

Output: `dist/Qwen3 TTS.exe`

You can copy `dist/Qwen3 TTS.exe` anywhere and run it. No Python needed on target machine.

> **Note:** The .exe includes the Python environment but NOT the model weights.
> Model weights (~3.5 GB each) are downloaded from HuggingFace on first use.
> First launch will need internet and ~5 minutes for download.

## CUDA Setup (for GPU acceleration)

1. **Check if you have an NVIDIA GPU**:
   ```cmd
   nvidia-smi
   ```
   If not found, install drivers from https://www.nvidia.com/drivers/

2. **Install PyTorch with CUDA**:
   ```cmd
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Verify**:
   ```cmd
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

## Performance Comparison

| Device | Model | Typical RTF |
|--------|-------|-------------|
| RTX 4090 | 1.7B HQ | ~2-3× RTF |
| RTX 3060 | 1.7B HQ | ~5-8× RTF |
| RTX 3060 | 0.6B Fast | ~2-3× RTF |
| CPU (16 cores) | 0.6B Fast | ~15-25× RTF |
| CPU (8 cores) | 0.6B Fast | ~30-50× RTF |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No module named 'qwen_tts'" | Run `pip install -U qwen-tts` |
| CUDA out of memory | Use Fast Mode (0.6B) or reduce max_new_tokens |
| Stuck on "Loading model" | Check internet; first download is ~3.5 GB |
| Window is blank | Check `http://localhost:8765` in browser; restart app |
| Audio distorted/static | Lower text length; instruct mode may cause artifacts |

## Folder Structure (Windows)

```
%APPDATA%/Qwen3TTS/
├── profiles/          # Saved voice clone profiles
└── config.json        # Settings (output dir, etc.)

%TEMP%/qwen3tts_clone/ # Temp uploaded audio
~/Desktop/             # Default output for generated files
```
