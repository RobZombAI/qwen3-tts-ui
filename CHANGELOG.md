# Changelog

All notable changes to this project will be documented in this file.

## [2.2.0] — 2026-07-18

### Added
- Native macOS app (pywebview) — no browser needed
- Proper "Close App" button with clean shutdown
- MPS memory cleanup on exit
- Live progress logs inside the app during model loading

### Changed
- Complete codebase cleanup — removed 6 legacy files
- `.gitignore` updated for professional release
- Server split into `server_native.py` (pywebview) and `server_minimal.py` (browser fallback)

### Removed
- `qwen3_tts_server.py` (replaced by server_native.py + server_minimal.py)
- `win_launcher.py` (Windows build now uses PyInstaller from `scripts/`)
- `app.py`, `app_launcher.py`, `setup.py`, `server_launcher.py`
- `Qwen3TTS.spec`, `Qwen3TTS.launcher`

## [2.1.0] — 2026-07-18

### Added
- "⏻ Quit App" button in browser UI with graceful shutdown
- Live progress tracking for model loading
- `server_minimal.py` — lightweight browser-based mode

## [2.0.0] — 2026-07-18

### Changed
- Complete rewrite: minimal, reliable server
- Browser mode instead of pywebview (reliability)
- Self-contained `.app` bundle with embedded Python venv

## [1.0.0] — 2026-07-17

### Added
- Initial public release
- CustomVoice TTS with 9 premium voices (1.7B and 0.6B models)
- Voice cloning from audio sample (x-vector and ICL modes)
- 10 language support (Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- Natural-language style/emotion instruction control
- Interactive waveform display for uploaded audio
- Audio trimming for files over 60 seconds
- Voice profile save/load/delete for cloned voices
- Configurable output directory with persistence
- Native macOS app bundle (WebKit window, no browser required)
- Windows launcher with PyInstaller .exe build support
- macOS Finder launcher (start.command)
- Device compatibility check before model download
- Comprehensive test suite (24+ tests)
- Full REST API with JSON responses
- Thread-safe model management (lazy loading, on-demand)
- Graceful shutdown with MPS memory cleanup
