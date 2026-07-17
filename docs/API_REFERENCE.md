# Qwen3 TTS REST API Reference

All endpoints return JSON. The server runs on `http://localhost:8765`.

## Endpoints

### `GET /api/models`
List available models with metadata.
```json
{
  "models": [
    {"id": "cv_1b7", "name": "CustomVoice 1.7B", "short": "🎤 High Quality", "desc": "Best quality, ~14× RTF", "size": "1.7B"},
    {"id": "cv_0b6", "name": "CustomVoice 0.6B", "short": "⚡ Fast Mode", "desc": "3-4× faster, excellent quality", "size": "0.6B"},
    {"id": "base",  "name": "Voice Clone 1.7B", "short": "🧬 Voice Clone", "desc": "Clone from audio sample", "size": "1.7B"}
  ],
  "meta": {...},
  "current": "cv_1b7"
}
```

### `POST /api/load_model`
Load a specific model. Models are cached after first load.
```json
{"model": "cv_1b7"} → {"status": "loading", "model": "cv_1b7"}
```

### `GET /api/status`
Get loading status for all models.
```json
{
  "cv_1b7": {"status": "ready", "name": "CustomVoice 1.7B", "speakers": [...]},
  "cv_0b6": {"status": "not_loaded"},
  "base": {"status": "loading"},
  "active": "cv_1b7"
}
```

### `GET /api/speakers`
List all speakers with detailed metadata (language, gender, age, style, dialect).

### `GET /api/languages`
List all supported languages.
```json
["Auto", "Chinese", "English", "Italian", "Japanese", ...]
```

### `POST /api/generate`
Generate speech from text.

**CustomVoice (cv_1b7 / cv_0b6):**
```json
{
  "model": "cv_1b7",
  "text": "Ciao, mondo!",
  "language": "Italian",
  "speaker": "Ryan",
  "instruct": "Speak naturally"
}
```

**Voice Clone (base):**
```json
{
  "model": "base",
  "text": "This is a voice clone test.",
  "language": "English",
  "ref_audio_file": "ref_123456_test.wav",
  "ref_text": "Original transcript (for ICL mode)",
  "x_vector_only": true,
  "trim_start": 0,
  "trim_end": 60
}
```

**Response:**
```json
{
  "success": true,
  "filename": "qwen3tts_1234567890.wav",
  "duration": 3.2,
  "sample_rate": 24000,
  "model": "cv_1b7"
}
```

### `POST /api/upload_ref_audio`
Upload reference audio for voice cloning. Multipart form-data, field name `file`.
```json
{
  "success": true,
  "filename": "ref_123456_test.wav",
  "duration": 5.2,
  "sample_rate": 24000,
  "waveform": [1.0, 0.98, 0.95, ...],
  "needs_trim": false,
  "trim_max": 5.2
}
```

### `GET /api/output_dir`
Get current output directory.
```json
{"output_dir": "/Users/name/Desktop", "exists": true}
```

### `POST /api/output_dir`
Set output directory.
```json
{"output_dir": "~/Desktop"} → {"output_dir": "/Users/name/Desktop", "exists": true}
```

### `GET /api/profiles/list`
List saved voice clone profiles.

### `POST /api/profiles/save`
Save current voice clone settings as a named profile.

### `POST /api/profiles/delete`
Delete a saved profile by name.

### `GET /api/check_device`
Check system compatibility for running models.
```json
{
  "platform": "darwin",
  "torch_version": "2.13.0",
  "mps_available": true,
  "cuda_available": false,
  "ram_gb": 128.0,
  "verdict": "✅ MPS (Apple Silicon) — full speed",
  "recommended_model": "cv_1b7"
}
```

### `POST /api/shutdown`
Gracefully shut down the server (frees MPS/CUDA memory before exit).

## Audio Output

Generated WAV files are:
- Format: 24 kHz, mono, 16-bit PCM
- Sample rate: 24000 Hz
- Duration: varies based on text length
- Location: configurable via `/api/output_dir`
