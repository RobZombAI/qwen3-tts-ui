# Changelog

All notable changes to this project will be documented in this file.

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
