# Qwen3 TTS — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Desktop Application                    │
│                                                          │
│  ┌──────────────┐    HTTP localhost:8765   ┌──────────┐  │
│  │  Native Window│◄────────────────────────►│  Flask   │  │
│  │  (WebKit/WV) │     JSON API + WAV      │  Server  │  │
│  └──────────────┘                          └─────┬────┘  │
│                                                  │       │
│                                        ┌─────────▼─────┐ │
│                                        │  Model Manager │ │
│                                        │  (thread-safe) │ │
│                                        └────────┬──────┘ │
│                                                 │        │
│                                  ┌──────────────┼──────┐ │
│                                  │  cv_1b7  │  cv_0b6│ │ │
│                                  │  1.7B    │  0.6B  │ │ │
│                                  │ Custom   │ Custom │ │ │
│                                  │ Voice    │ Voice  │ │ │
│                                  ├──────────┼────────┤ │ │
│                                  │  base    │        │ │ │
│                                  │  1.7B    │        │ │ │
│                                  │  Voice   │        │ │ │
│                                  │  Clone   │        │ │ │
│                                  └──────────┴────────┘ │ │
└──────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Local Server + Thin Client
- Flask runs on localhost:8765 in a background thread
- The UI is served as HTML/CSS/JS from the same process
- Audio files are served via HTTP (no file system coupling)
- No external dependencies for the client except a WebView

### 2. Lazy Model Loading
- Models are loaded on demand, not at startup
- Multiple models can coexist in memory (tested with 3 simultaneous on 128GB Mac)
- Each model is ~3-4 GB in bfloat16
- Loading is thread-safe with lock + event synchronization

### 3. Dual-Mode UI
- **Native app:** pywebview creates a WebKit window (macOS) or MS Edge WebView2 (Windows)
- **Browser mode:** Open `http://localhost:8765` in any browser for the same experience
- Both modes share 100% of the UI code

### 4. Voice Profile Storage
- Profiles stored as JSON in `~/.qwen3tts_profiles/`
- Reference audio copied to the same directory
- Settings persisted: name, ref_text, x_vector_only, trim boundaries

### 5. Graceful Shutdown
- Close window → POST `/api/shutdown` → clear MPS cache → exit
- OS kernel reclaims all memory (RAM + GPU/Metal) immediately
- No orphan processes, no memory leaks

## Threading Model

```
Main Thread (pywebview)
  │
  ├── WebView event loop (GUI)
  │
  └── Background Thread (daemon)
      └── Flask server
          ├── Request handlers (threaded)
          └── Model inference (blocks request thread)
```

- Flask uses `threaded=True` for concurrent request handling
- Model inference is CPU/GPU-bound and runs in the request thread
- File uploads are handled synchronously (fast I/O)
- Generation requests can take 30-90 seconds depending on text length

## Model Specifications

| Model | Parameters | Size (bf16) | RTF (MPS) | RTF (CUDA) | RTF (CPU) |
|-------|-----------|-------------|-----------|------------|-----------|
| cv_1b7 | 1.7B | ~3.4 GB | ~14× | ~3-5× | ~30-50× |
| cv_0b6 | 0.6B | ~1.2 GB | ~4× | ~1-2× | ~10-15× |
| base | 1.7B | ~3.4 GB | ~15× | ~3-5× | ~30-50× |

RTF = Real-Time Factor. 10 seconds of audio at 14× RTF = 140 seconds generation time.

## API Design Principles

- All endpoints return JSON
- Errors return appropriate HTTP status codes (400, 404, 500, 503)
- File uploads use multipart/form-data
- Audio is served as static files via `/output/<filename>`
- Models are identified by string IDs (`cv_1b7`, `cv_0b6`, `base`)
