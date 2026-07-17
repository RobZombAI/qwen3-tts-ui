#!/usr/bin/env python3
"""
Qwen3 TTS — Web UI Server (Dual Model)
========================================
Supports:
  - CustomVoice: 9 preset voices with instruction control
  - Base: 3-second voice cloning from audio sample

Usage:
    source venv/bin/activate && python qwen3_tts_server.py
"""

import os
import sys
import json
import time
import threading
import webbrowser
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import flask
from werkzeug.utils import secure_filename

# ── Simple logger ──
import logging
_log = logging.getLogger("qwen3tts")


def log(msg: str):
    _log.info(msg)
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

# ── Config ───────────────────────────────────────────────────────────────────

PORT = 8765
CONFIG_FILE = os.path.expanduser("~/.qwen3tts_config.json")
TEMP_DIR = os.path.join(tempfile.gettempdir(), "qwen3tts_clone")
PROFILES_DIR = os.path.expanduser("~/.qwen3tts_profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)


def load_config() -> dict:
    """Load persistent config from ~/.qwen3tts_config.json"""
    defaults = {"output_dir": os.path.expanduser("~/Desktop")}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
                defaults.update(cfg)
    except Exception:
        pass
    return defaults


def save_config(cfg: dict):
    """Save persistent config to ~/.qwen3tts_config.json"""
    try:
        existing = load_config()
        existing.update(cfg)
        with open(CONFIG_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        log(f"Failed to save config: {e}")


def get_output_dir() -> str:
    """Get the current output directory, creating it if needed."""
    cfg = load_config()
    od = os.path.expanduser(cfg.get("output_dir", "~/Desktop"))
    os.makedirs(od, exist_ok=True)
    return od


os.makedirs(TEMP_DIR, exist_ok=True)

MODEL_IDS = {
    "cv_1b7": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "cv_0b6": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

MODEL_META = {
    "cv_1b7": {"name": "CustomVoice 1.7B", "short": "🎤 High Quality", "desc": "Best quality, ~14× RTF", "size": "1.7B"},
    "cv_0b6": {"name": "CustomVoice 0.6B", "short": "⚡ Fast Mode", "desc": "3-4× faster, excellent quality", "size": "0.6B"},
    "base":  {"name": "Voice Clone 1.7B", "short": "🧬 Voice Clone", "desc": "Clone from audio sample", "size": "1.7B"},
}

# Speaker info — detailed
SPEAKER_INFO = {
    "Vivian": {
        "short": "Bright, slightly edgy young female",
        "language": "Chinese",
        "dialect": "Standard",
        "gender": "Female",
        "age": "Young",
        "style": "Bright, slightly edgy",
        "emoji": "👩",
        "best_for": "Chinese (native), any language"
    },
    "Serena": {
        "short": "Warm, gentle young female",
        "language": "Chinese",
        "dialect": "Standard",
        "gender": "Female",
        "age": "Young",
        "style": "Warm, gentle",
        "emoji": "👩",
        "best_for": "Chinese (native), any language"
    },
    "Uncle_Fu": {
        "short": "Seasoned male with low, mellow timbre",
        "language": "Chinese",
        "dialect": "Standard",
        "gender": "Male",
        "age": "Mature",
        "style": "Deep, mellow, authoritative",
        "emoji": "👨",
        "best_for": "Chinese (native), any language"
    },
    "Dylan": {
        "short": "Youthful Beijing male, clear natural",
        "language": "Chinese",
        "dialect": "Beijing",
        "gender": "Male",
        "age": "Young",
        "style": "Clear, natural, youthful",
        "emoji": "👦",
        "best_for": "Chinese (Beijing dialect), any language"
    },
    "Eric": {
        "short": "Lively Chengdu male, slightly husky",
        "language": "Chinese",
        "dialect": "Sichuan",
        "gender": "Male",
        "age": "Young",
        "style": "Lively, slightly husky, bright",
        "emoji": "👦",
        "best_for": "Chinese (Sichuan dialect), any language"
    },
    "Ryan": {
        "short": "Dynamic male with strong rhythmic drive",
        "language": "English",
        "dialect": "Standard",
        "gender": "Male",
        "age": "Adult",
        "style": "Dynamic, energetic, rhythmic",
        "emoji": "👨",
        "best_for": "English (native), Italian, any language"
    },
    "Aiden": {
        "short": "Sunny American male with clear midrange",
        "language": "English",
        "dialect": "American",
        "gender": "Male",
        "age": "Young adult",
        "style": "Sunny, clear, midrange",
        "emoji": "👦",
        "best_for": "English (native), Italian, any language"
    },
    "Ono_Anna": {
        "short": "Playful Japanese female, light nimble",
        "language": "Japanese",
        "dialect": "Standard",
        "gender": "Female",
        "age": "Young",
        "style": "Playful, light, nimble",
        "emoji": "👩",
        "best_for": "Japanese (native), any language"
    },
    "Sohee": {
        "short": "Warm Korean female with rich emotion",
        "language": "Korean",
        "dialect": "Standard",
        "gender": "Female",
        "age": "Young",
        "style": "Warm, emotional, rich",
        "emoji": "👩",
        "best_for": "Korean (native), any language"
    },
}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# ── Model management (thread-safe, lazy-loaded) ──────────────────────────────

_models: Dict[str, Any] = {}
_models_lock = threading.Lock()
_models_loaded: Dict[str, bool] = {}
_models_error: Dict[str, Optional[str]] = {}
_loading_events: Dict[str, threading.Event] = {}

_model_status_listeners: list = []  # callbacks for status changes


def notify_status(msg: str):
    for cb in _model_status_listeners:
        try:
            cb(msg)
        except Exception:
            pass


def get_model(model_type: str, progress_callback=None) -> Tuple[Any, Optional[str]]:
    """Load a model on first call, return (model, error)."""
    global _models, _models_loaded, _models_error

    if _models_loaded.get(model_type) and model_type in _models:
        return _models[model_type], None

    if _models_error.get(model_type):
        return None, _models_error[model_type]

    with _models_lock:
        if _models_loaded.get(model_type) and model_type in _models:
            return _models[model_type], None
        if _models_error.get(model_type):
            return None, _models_error[model_type]

        # Mark as loading
        if model_type not in _loading_events:
            _loading_events[model_type] = threading.Event()

        event = _loading_events[model_type]

        try:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            import torch
            from qwen_tts import Qwen3TTSModel

            model_id = MODEL_IDS[model_type]
            meta = MODEL_META.get(model_type, {})
            name = meta.get("name", model_type)

            if progress_callback:
                progress_callback(f"Loading {name}…")

            if torch.backends.mps.is_available():
                device = "mps"
                model_dtype = torch.bfloat16
                attn_impl = "sdpa"
            else:
                device = "cpu"
                model_dtype = torch.float32
                attn_impl = "eager"

            if progress_callback:
                progress_callback(f"Downloading {name} ({device.upper()})…")

            model = Qwen3TTSModel.from_pretrained(
                model_id,
                dtype=model_dtype,
                attn_implementation=attn_impl,
            )

            if progress_callback:
                progress_callback("Moving to device…")

            if device != "cpu":
                model.model = model.model.to(device)
            model.device = torch.device(device)

            if progress_callback:
                progress_callback("Warming up…")

            # Warm-up
            try:
                if model_type.startswith("cv_"):
                    model.generate_custom_voice(
                        text="Hello.", language="English",
                        speaker="Ryan", max_new_tokens=8,
                    )
                else:
                    model.generate_voice_clone(
                        text="Hello.", language="English",
                        x_vector_only_mode=True,
                        ref_audio=(None, 16000),  # dummy — will fail but warm up main model
                        max_new_tokens=8,
                    )
            except Exception:
                pass

            _models[model_type] = model
            _models_loaded[model_type] = True
            _models_error[model_type] = None
            event.set()

            if progress_callback:
                progress_callback(f"✅ {name} ready")
            return model, None

        except Exception as e:
            err = str(e)
            _models_error[model_type] = err
            _models_loaded[model_type] = False
            event.set()
            if progress_callback:
                progress_callback(f"❌ {name} error: {err}")
            return None, err


# ── Flask App ───────────────────────────────────────────────────────────────

app = flask.Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload


# ── Static Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return flask.render_template_string(HTML_TEMPLATE)


@app.route("/output/<filename>")
def serve_output(filename):
    safe = os.path.basename(filename)
    path = os.path.join(get_output_dir(), safe)
    if not os.path.exists(path):
        return flask.jsonify({"error": "File not found"}), 404
    return flask.send_file(path, mimetype="audio/wav")


# ── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/models")
def api_models():
    return flask.jsonify({
        "models": [
            {"id": "cv_1b7", **MODEL_META["cv_1b7"]},
            {"id": "cv_0b6", **MODEL_META["cv_0b6"]},
            {"id": "base", **MODEL_META["base"]},
        ],
        "meta": MODEL_META,
        "current": next((m for m in ["cv_1b7", "cv_0b6", "base"] if _models_loaded.get(m)), None),
    })


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = flask.request.get_json(force=True)
    model_type = data.get("model", "custom_voice")
    if model_type not in MODEL_IDS:
        return flask.jsonify({"error": f"Unknown model: {model_type}"}), 400

    if _models_loaded.get(model_type):
        return flask.jsonify({"status": "already_loaded", "model": model_type})

    # Start loading in background
    def load_bg():
        get_model(model_type, progress_callback=lambda msg: notify_status(msg))

    thread = threading.Thread(target=load_bg, daemon=True)
    thread.start()

    return flask.jsonify({"status": "loading", "model": model_type})


@app.route("/api/speakers")
def api_speakers():
    """Return detailed speaker info."""
    return flask.jsonify(SPEAKER_INFO)


@app.route("/api/languages")
def api_languages():
    return flask.jsonify(LANGUAGES)


@app.route("/api/output_dir", methods=["GET", "POST"])
def api_output_dir():
    """Get or set the output directory for generated audio files."""
    if flask.request.method == "GET":
        od = get_output_dir()
        return flask.jsonify({"output_dir": od, "exists": os.path.isdir(od)})

    data = flask.request.get_json(force=True)
    new_dir = data.get("output_dir", "").strip()
    if not new_dir:
        return flask.jsonify({"error": "Path is required"}), 400

    new_dir = os.path.expanduser(new_dir)
    if not os.path.isdir(new_dir):
        try:
            os.makedirs(new_dir, exist_ok=True)
        except Exception as e:
            return flask.jsonify({"error": f"Cannot create directory: {e}"}), 400

    save_config({"output_dir": new_dir})
    log(f"Output directory changed to: {new_dir}")
    return flask.jsonify({"output_dir": new_dir, "exists": True})


# ── Voice Profile Management ─────────────────────────────────────────────────


@app.route("/api/profiles/list")
def api_profiles_list():
    """List all saved voice clone profiles."""
    profiles = []
    if not os.path.isdir(PROFILES_DIR):
        return flask.jsonify({"profiles": []})

    for fname in sorted(os.listdir(PROFILES_DIR)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(PROFILES_DIR, fname)) as f:
                    profile = json.load(f)
                    profiles.append(profile)
            except Exception:
                continue

    # Sort by timestamp descending (newest first)
    profiles.sort(key=lambda p: p.get("created", 0), reverse=True)
    return flask.jsonify({"profiles": profiles})


@app.route("/api/profiles/save", methods=["POST"])
def api_profiles_save():
    """Save a voice clone profile for reuse."""
    data = flask.request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return flask.jsonify({"error": "Profile name is required"}), 400

    # Sanitize name for filename
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    if not safe_name:
        return flask.jsonify({"error": "Invalid profile name"}), 400

    ref_file = data.get("ref_audio_file") or ""
    if not ref_file:
        return flask.jsonify({"error": "No reference audio to save"}), 400

    # Find the source audio file
    src_path = os.path.join(TEMP_DIR, os.path.basename(ref_file))
    if not os.path.exists(src_path):
        return flask.jsonify({"error": "Reference audio file no longer available"}), 400

    # Copy audio to profiles dir
    import shutil
    audio_dst = os.path.join(PROFILES_DIR, f"{safe_name}_audio.wav")
    try:
        shutil.copy2(src_path, audio_dst)
    except Exception as e:
        return flask.jsonify({"error": f"Cannot copy audio: {e}"}), 500

    # Save profile metadata
    profile = {
        "name": safe_name,
        "created": int(time.time() * 1000),
        "ref_audio_file": f"{safe_name}_audio.wav",
        "ref_text": data.get("ref_text", ""),
        "x_vector_only": data.get("x_vector_only", True),
        "trim_start": data.get("trim_start", 0),
        "trim_end": data.get("trim_end", 0),
    }

    profile_path = os.path.join(PROFILES_DIR, f"{safe_name}.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    log(f"Voice profile saved: {safe_name}")
    return flask.jsonify({"success": True, "profile": profile})


@app.route("/api/profiles/delete", methods=["POST"])
def api_profiles_delete():
    """Delete a saved voice profile."""
    data = flask.request.get_json(force=True)
    name = (data.get("name") or "").strip()
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    if not safe_name:
        return flask.jsonify({"error": "Invalid profile name"}), 400

    # Delete metadata
    json_path = os.path.join(PROFILES_DIR, f"{safe_name}.json")
    if os.path.exists(json_path):
        os.unlink(json_path)

    # Delete audio
    audio_path = os.path.join(PROFILES_DIR, f"{safe_name}_audio.wav")
    if os.path.exists(audio_path):
        os.unlink(audio_path)

    log(f"Voice profile deleted: {safe_name}")
    return flask.jsonify({"success": True})


_shutting_down = False


@app.route("/api/check_device")
def api_check_device():
    """Check if the device can run the models. Returns compatibility report."""
    report = {"platform": sys.platform, "verdict": "ok", "warnings": [], "recommended_model": "cv_1b7"}

    try:
        import torch
        report["torch_version"] = torch.__version__
        report["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        report["cuda_available"] = torch.cuda.is_available()
        report["cpu_count"] = os.cpu_count()
    except Exception as e:
        report["warnings"].append(f"Cannot check PyTorch: {e}")
        return flask.jsonify(report)

    # Check RAM
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / (1024**3), 1)
        report["ram_gb"] = ram

        if report.get("cuda_available"):
            report["verdict"] = "✅ CUDA GPU — full speed"
        elif report.get("mps_available"):
            report["verdict"] = "✅ MPS (Apple Silicon) — full speed"
        elif ram >= 32:
            report["verdict"] = "⚠️ CPU only — Fast Mode recommended (0.6B)"
            report["recommended_model"] = "cv_0b6"
        elif ram >= 16:
            report["verdict"] = "⚠️ CPU, limited RAM — use Fast Mode (0.6B)"
            report["recommended_model"] = "cv_0b6"
        else:
            report["verdict"] = "❌ May be too slow — minimum 16GB RAM, GPU recommended"
            report["recommended_model"] = "cv_0b6"
            report["warnings"].append("Device may not have enough resources for smooth operation")

    except ImportError:
        report["ram_gb"] = "unknown"
        report["warnings"].append("Cannot check RAM (psutil not installed)")

    return flask.jsonify(report)


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Gracefully shut down the server. Called by the native app on window close."""
    global _shutting_down
    _shutting_down = True
    log("Shutdown requested — stopping server…")

    # Free MPS memory if possible
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            log("MPS cache cleared")
    except Exception:
        pass

    # Stop Flask server in a background thread (Flask doesn't support
    # clean shutdown from within a request handler, so we do it async)
    def _do_shutdown():
        import time
        time.sleep(0.5)  # Let the response be sent
        os._exit(0)

    threading.Thread(target=_do_shutdown, daemon=True).start()

    return flask.jsonify({"shutdown": True, "message": "Server shutting down…"})


@app.route("/api/status")
def api_status():
    """Return status for all models."""
    result = {}
    all_mts = list(MODEL_IDS.keys())
    for mt in all_mts:
        meta = MODEL_META.get(mt, {})
        if _models_loaded.get(mt):
            info = {"status": "ready", "name": meta.get("name", mt), "short": meta.get("short", mt), "size": meta.get("size", "")}
            if mt.startswith("cv_"):
                info["speakers"] = list(SPEAKER_INFO.keys())
                info["model_type"] = "custom_voice"
            else:
                info["model_type"] = "base"
            result[mt] = info
        elif _models_error.get(mt):
            result[mt] = {"status": "error", "error": _models_error[mt]}
        else:
            result[mt] = {"status": "not_loaded"}

    # Also include any model currently loading
    for mt, evt in _loading_events.items():
        if not evt.is_set() and mt not in result:
            result[mt] = {"status": "loading"}
        elif evt.is_set():
            pass  # already handled above

    # Which model is currently active/loaded
    result["active"] = None
    for mt in all_mts:
        if _models_loaded.get(mt):
            result["active"] = mt
            break

    result["all_model_ids"] = all_mts
    result["model_meta"] = MODEL_META

    return flask.jsonify(result)


@app.route("/api/upload_ref_audio", methods=["POST"])
def api_upload_ref_audio():
    """Upload reference audio for voice cloning. Returns waveform data for UI."""
    if "file" not in flask.request.files:
        return flask.jsonify({"error": "No file uploaded"}), 400

    file = flask.request.files["file"]
    if file.filename == "":
        return flask.jsonify({"error": "Empty filename"}), 400

    # Save to temp dir
    safe = secure_filename(file.filename)
    ts = int(time.time() * 1000)
    filename = f"ref_{ts}_{safe}"
    filepath = os.path.join(TEMP_DIR, filename)
    file.save(filepath)

    # Verify and analyze audio
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(filepath)
        duration = len(data) / sr

        if duration < 1.0:
            os.unlink(filepath)
            return flask.jsonify({"error": "Audio too short (< 1 second)"}), 400

        # Generate waveform data (downsampled to ~300 bins for display)
        n_bins = 300
        if len(data) > n_bins:
            # Compute peak amplitude per bin
            bin_size = len(data) // n_bins
            waveform = []
            for i in range(n_bins):
                chunk = data[i * bin_size : (i + 1) * bin_size]
                if len(chunk) > 0:
                    # Use max absolute amplitude in each bin
                    amp = float(np.max(np.abs(chunk)))
                    waveform.append(amp)
                else:
                    waveform.append(0.0)
        else:
            waveform = [float(np.abs(x)) for x in data]

        # Normalize waveform to [0, 1]
        max_val = max(waveform) if waveform else 1.0
        if max_val > 0:
            waveform = [round(v / max_val, 4) for v in waveform]

        response = {
            "success": True,
            "filename": filename,
            "duration": round(duration, 1),
            "sample_rate": sr,
            "waveform": waveform,
            "needs_trim": duration > 60.0,
            "trim_max": min(60.0, duration),
        }

        if duration > 60.0:
            response["trim_warning"] = f"Audio is {duration:.0f}s. Please select a {60:.0f}s segment below."

        return flask.jsonify(response)

    except Exception as e:
        if os.path.exists(filepath):
            os.unlink(filepath)
        return flask.jsonify({"error": f"Invalid audio file: {e}"}), 400


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate speech from text. Supports both CustomVoice and Base models."""
    data = flask.request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return flask.jsonify({"error": "Text is required"}), 400

    model_type = data.get("model", "custom_voice")
    language = data.get("language", "Auto")
    instruct = (data.get("instruct") or "").strip()

    # Load model
    model, err = get_model(model_type)
    if err:
        return flask.jsonify({"error": f"Model error: {err}"}), 500
    if model is None:
        return flask.jsonify({"error": "Model not loaded yet"}), 503

    try:
        lang = language if language != "Auto" else None
        instruct_val = instruct if instruct else None

        if model_type.startswith("cv_"):
            speaker = data.get("speaker", "Ryan")
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=lang,
                speaker=speaker,
                instruct=instruct_val,
                max_new_tokens=min(2048, max(64, int(len(text) * 3))),
            )

        elif model_type == "base":
            ref_file = data.get("ref_audio_file") or ""
            ref_text = data.get("ref_text") or ""
            x_vector_only = data.get("x_vector_only", True)
            trim_start = data.get("trim_start", 0)
            trim_end = data.get("trim_end", 0)

            if not ref_file:
                return flask.jsonify({"error": "Reference audio required for voice cloning"}), 400

            ref_path = os.path.join(TEMP_DIR, os.path.basename(ref_file))
            if not os.path.exists(ref_path):
                return flask.jsonify({"error": "Reference audio file not found, please re-upload"}), 400

            # Trim audio if trim boundaries provided
            if trim_end > trim_start:
                try:
                    import soundfile as sf
                    import numpy as np
                    audio_data, audio_sr = sf.read(ref_path)
                    start_sample = int(trim_start * audio_sr)
                    end_sample = min(int(trim_end * audio_sr), len(audio_data))
                    if end_sample > start_sample:
                        trimmed = audio_data[start_sample:end_sample]
                        # Save trimmed version
                        trim_path = ref_path.replace(".wav", "_trimmed.wav").replace(".mp3", "_trimmed.wav")
                        sf.write(trim_path, trimmed, audio_sr)
                        ref_path = trim_path
                        log(f"Trimmed ref audio: {trim_start:.1f}s → {trim_end:.1f}s ({len(trimmed)/audio_sr:.1f}s)")
                except Exception as trim_err:
                    log(f"Trim warning (non-fatal): {trim_err}")

            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=ref_path,
                ref_text=ref_text if ref_text.strip() else None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=min(2048, max(64, int(len(text) * 3))),
            )
        else:
            return flask.jsonify({"error": f"Unknown model: {model_type}"}), 400

        # Save output
        ts = int(time.time() * 1000)
        filename = f"qwen3tts_{ts}.wav"
        filepath = os.path.join(get_output_dir(), filename)
        import soundfile as sf
        sf.write(filepath, wavs[0], sr)

        duration = len(wavs[0]) / sr

        return flask.jsonify({
            "success": True,
            "filename": filename,
            "duration": round(duration, 1),
            "sample_rate": sr,
            "model": model_type,
        })

    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500


# ── HTML Template ───────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3 TTS — CustomVoice + Voice Clone</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .container {
    max-width: 780px;
    width: 100%;
    margin: 24px;
    background: #16213e;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }
  h1 {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 2px;
    color: #f0f0ff;
  }
  .subtitle {
    font-size: 13px;
    color: #8888aa;
    margin-bottom: 20px;
  }

  /* Status bar */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: #0f3460;
    border-radius: 12px;
    margin-bottom: 18px;
    font-size: 13px;
  }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .status-dot.loading { background: #f0a500; animation: pulse 1.5s infinite; }
  .status-dot.ready { background: #4ade80; }
  .status-dot.error { background: #f87171; }
  .status-dot.inactive { background: #555; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .status-text { flex: 1; }
  .retry-btn {
    background: #1e3a5f; border: none; color: #aaccff;
    padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }
  .retry-btn:hover { background: #2a4a7a; }

  /* Model selector */
  .model-selector {
    display: flex;
    gap: 8px;
    margin-bottom: 18px;
  }
  .model-btn {
    flex: 1;
    padding: 10px 16px;
    border: 2px solid #1a4a7a;
    border-radius: 10px;
    background: #0f3460;
    color: #8899bb;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
    font-family: inherit;
    font-size: 14px;
  }
  .model-btn:hover { border-color: #4a90d9; color: #aaccff; }
  .model-btn.active {
    border-color: #4ade80;
    background: #0a2a1a;
    color: #4ade80;
    font-weight: 600;
  }
  .model-btn .desc {
    font-size: 11px;
    color: #667;
    font-weight: 400;
    display: block;
    margin-top: 2px;
  }
  .model-btn.active .desc { color: #6a9a7a; }

  /* Forms */
  label { font-size: 13px; font-weight: 600; color: #aab; display: block; margin-bottom: 4px; }
  textarea, select, input {
    width: 100%;
    background: #0f3460;
    border: 1px solid #1a4a7a;
    color: #e0e0e0;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border 0.2s;
  }
  textarea:focus, select:focus, input:focus { border-color: #4a90d9; }

  .controls, .clone-controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 14px 0;
  }
  .clone-controls { grid-template-columns: 1fr 1fr 1fr; }

  .instruct-row { margin-bottom: 16px; }
  .instruct-row input { padding: 10px 14px; }

  /* Upload area */
  .upload-zone {
    border: 2px dashed #1a4a7a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 14px;
    background: #0a1a2e;
  }
  .upload-zone:hover, .upload-zone.dragover { border-color: #4a90d9; background: #0f2240; }
  .upload-zone.has-file { border-color: #4ade80; border-style: solid; }
  .upload-zone .icon { font-size: 32px; margin-bottom: 8px; }
  .upload-zone .hint { font-size: 13px; color: #667; }
  .upload-zone .file-info { font-size: 13px; color: #4ade80; margin-top: 6px; }
  .upload-zone input[type="file"] { display: none; }
  .upload-status {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
  }

  /* x-vector toggle */
  .toggle-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding: 10px 14px;
    background: #0a1a2e;
    border-radius: 10px;
  }
  .toggle-row label { margin: 0; cursor: pointer; flex: 1; }
  .toggle-row .desc { font-size: 12px; color: #667; font-weight: 400; }
  .toggle-switch {
    position: relative;
    width: 44px;
    height: 24px;
    flex-shrink: 0;
  }
  .toggle-switch input { opacity: 0; width: 0; height: 0; }
  .toggle-slider {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: #1a4a7a;
    border-radius: 12px;
    cursor: pointer;
    transition: 0.3s;
  }
  .toggle-slider::before {
    content: "";
    position: absolute;
    width: 18px; height: 18px;
    left: 3px; bottom: 3px;
    background: #e0e0e0;
    border-radius: 50%;
    transition: 0.3s;
  }
  .toggle-switch input:checked + .toggle-slider { background: #4ade80; }
  .toggle-switch input:checked + .toggle-slider::before { transform: translateX(20px); }

  /* Button */
  .btn-gen {
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    background: linear-gradient(135deg, #1a6b3c, #228b4f);
    color: #fff;
    margin-top: 4px;
  }
  .btn-gen:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(34,139,79,0.4); }
  .btn-gen:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-gen.generating { background: linear-gradient(135deg, #555, #666); }

  /* Output */
  .output-section {
    margin-top: 20px;
    padding: 16px;
    background: #0f3460;
    border-radius: 12px;
    display: none;
  }
  .output-section.show { display: block; }
  .output-actions { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
  .output-actions audio { flex: 1; min-width: 0; height: 40px; }
  .output-actions a {
    background: #1e3a5f;
    padding: 8px 16px;
    border-radius: 8px;
    color: #aaccff;
    text-decoration: none;
    font-size: 13px;
    white-space: nowrap;
    cursor: pointer;
  }
  .output-actions a:hover { background: #2a4a7a; }
  .error-msg { margin-top: 8px; color: #f87171; font-size: 13px; }

  .section-hidden { display: none; }

  /* Loading spinner */
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid #8888aa;
    border-top-color: #4a90d9;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Waveform and trim */
  #waveformCanvas { display: block; }
  #trimSlider {
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    background: #1a4a7a;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }
  #trimSlider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    background: #4ade80;
    border-radius: 50%;
    cursor: pointer;
    border: 2px solid #0a1a2e;
  }
  #trimSlider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    background: #4ade80;
    border-radius: 50%;
    cursor: pointer;
    border: 2px solid #0a1a2e;
  }

  /* Saved voice profiles */
  .saved-voices { margin-top: 16px; }
  .saved-voices-title {
    font-size: 13px; font-weight: 600; color: #aab;
    margin-bottom: 8px; cursor: pointer;
    display: flex; align-items: center; gap: 6px;
  }
  .saved-voices-title:hover { color: #cce; }
  .profile-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-height: 240px;
    overflow-y: auto;
  }
  .profile-card {
    background: #0f3460;
    border: 1px solid #1a4a7a;
    border-radius: 10px;
    padding: 10px 12px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }
  .profile-card:hover { border-color: #4a90d9; background: #153a6a; }
  .profile-card .name { font-size: 13px; font-weight: 600; color: #e0e0ff; }
  .profile-card .meta { font-size: 11px; color: #667; margin-top: 2px; }
  .profile-card .delete-btn {
    position: absolute;
    top: 4px; right: 6px;
    background: none; border: none;
    color: #666; cursor: pointer;
    font-size: 14px; padding: 2px 4px;
    border-radius: 4px;
  }
  .profile-card .delete-btn:hover { color: #f87171; background: #2a1a1a; }
  .profile-card.active { border-color: #4ade80; background: #0a2a1a; }
  .save-voice-btn {
    background: #1e3a5f; border: none; color: #aaccff;
    padding: 6px 14px; border-radius: 8px; cursor: pointer;
    font-size: 12px;
  }
  .save-voice-btn:hover { background: #2a4a7a; }
</style>
</head>
<body>
<div class="container">
  <h1>🎙️ Qwen3 TTS</h1>
  <div class="subtitle">CustomVoice + Voice Clone · 10 lingue · M5 Max</div>

  <!-- Status bar -->
  <div class="status-bar" id="statusBar">
    <div class="status-dot loading" id="statusDot"></div>
    <span class="status-text" id="statusText">Initializing…</span>
    <button class="retry-btn" id="retryBtn" style="display:none" onclick="retryLoad()">Retry</button>
  </div>

  <!-- Model selector -->
  <div class="model-selector" id="modelSelector">
    <button class="model-btn active" data-model="cv" onclick="switchModel('cv')">
      🎤 CustomVoice
      <span class="desc">Preset voices with style control</span>
    </button>
    <button class="model-btn" data-model="base" onclick="switchModel('base')">
      🧬 Voice Clone
      <span class="desc">Clone any voice from 3s audio</span>
    </button>
  </div>

  <!-- CustomVoice section -->
  <div id="cvSection">
    <!-- Quality selector -->
    <div class="toggle-row" style="margin-bottom:14px">
      <div>
        <label for="qualityToggle">⚡ Fast Mode</label>
        <div class="desc">Use 0.6B model (3-4× faster, excellent quality). Off = 1.7B High Quality.</div>
      </div>
      <label class="toggle-switch">
        <input type="checkbox" id="qualityToggle" onchange="updateModelForCV()">
        <span class="toggle-slider"></span>
      </label>
    </div>
    <div class="controls">
      <div>
        <label for="langSelect">Language</label>
        <select id="langSelect"></select>
      </div>
      <div>
        <label for="spkSelect">Speaker</label>
        <select id="spkSelect" size="9" style="font-size:12px;height:auto;min-height:280px"></select>
        <div id="spkDetails" style="margin-top:8px;padding:10px;background:#0a1a2e;border-radius:8px;font-size:12px;line-height:1.5;display:none"></div>
      </div>
    </div>
    <div class="instruct-row">
      <label for="instructInput">Style / Emotion <span style="font-weight:400;color:#667">(optional)</span></label>
      <input id="instructInput" type="text" placeholder="e.g. 'Speak happily and excitedly' — or leave empty for natural">
    </div>
  </div>

  <!-- Voice Clone section -->
  <div id="cloneSection" class="section-hidden">
    <!-- Upload zone -->
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('audioFile').click()"
         ondragover="event.preventDefault();this.classList.add('dragover')"
         ondragleave="this.classList.remove('dragover')"
         ondrop="event.preventDefault();handleDrop(event)">
      <div class="icon">🎤</div>
      <div class="hint" id="uploadHint">Click or drag a WAV/MP3 sample here (3-60 seconds recommended)</div>
      <div class="file-info" id="uploadInfo" style="display:none"></div>
      <div class="upload-status" id="uploadStatus" style="display:none"></div>
      <input type="file" id="audioFile" accept="audio/*" onchange="handleUpload(this)">
    </div>

    <!-- Waveform display + trim (shown after upload) -->
    <div id="waveformSection" style="display:none">
      <div style="position:relative;margin-bottom:8px">
        <canvas id="waveformCanvas" width="700" height="100" style="width:100%;height:100px;border-radius:8px;background:#0a1a2e;cursor:pointer"></canvas>
        <!-- Trim handles -->
        <div id="trimOverlay" style="display:none;position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none">
          <div id="trimRegion" style="position:absolute;top:0;bottom:0;background:rgba(74,222,128,0.2);border-left:2px solid #4ade80;border-right:2px solid #4ade80;pointer-events:none"></div>
        </div>
      </div>
      <!-- Trim sliders -->
      <div id="trimControls" style="display:none;margin-bottom:12px">
        <div style="display:flex;gap:16px;align-items:center;font-size:12px">
          <span id="trimLabel">Select a 60-second segment:</span>
          <span>Start: <b id="trimStartLabel">0.0</b>s</span>
          <span>End: <b id="trimEndLabel">60.0</b>s</span>
          <span style="flex:1;text-align:right;color:#888">⏱ <span id="trimDuration">60.0</span>s selected</span>
        </div>
        <input type="range" id="trimSlider" min="0" max="0" step="0.1" value="0"
               style="width:100%;margin-top:6px" oninput="updateTrim()">
        <div style="font-size:11px;color:#888;margin-top:2px" id="trimWarning"></div>
      </div>
    </div>

    <!-- Toggle ICL mode -->
    <div class="toggle-row">
      <div>
        <label for="iclToggle">🎯 ICL Mode (better quality)</label>
        <div class="desc">Needs transcript of the reference audio. If disabled, uses x-vector only (no transcript needed).</div>
      </div>
      <label class="toggle-switch">
        <input type="checkbox" id="iclToggle" onchange="toggleICL()">
        <span class="toggle-slider"></span>
      </label>
    </div>

    <div class="clone-controls">
      <div class="section-hidden" id="refTextSection">
        <label for="refText">Reference transcript (required for ICL)</label>
        <input id="refText" type="text" placeholder="Type the exact words spoken in the audio...">
      </div>
      <div>
        <label for="cloneLangSelect">Language</label>
        <select id="cloneLangSelect"></select>
      </div>
      <div>
        <label for="cloneInstruct">Style / Emotion</label>
        <input id="cloneInstruct" type="text" placeholder="e.g. 'Speak naturally'">
      </div>
    </div>

    <!-- Saved voice profiles -->
    <div class="saved-voices" id="savedVoicesSection" style="display:none">
      <div class="saved-voices-title" onclick="toggleProfiles()">
        📂 Saved Voices <span id="profileCount">(0)</span>
        <span style="font-weight:400;font-size:11px;color:#667">click to show</span>
      </div>
      <div id="profileGrid" class="profile-grid" style="display:none"></div>
      <div id="noProfiles" style="font-size:12px;color:#445;padding:8px;display:none">
        No saved voices yet. After a voice clone generation, click "💾 Save Voice" to keep the profile.
      </div>
    </div>
  </div>

  <!-- Common: text input -->
  <label for="textInput">Text to synthesize</label>
  <textarea id="textInput" rows="4" placeholder="Enter text in any of the 10 supported languages…">Ciao! Questo è un test di sintesi vocale con Qwen3 TTS.</textarea>

  <!-- Output folder -->
  <div style="display:flex;align-items:center;gap:8px;margin:8px 0 12px;padding:8px 12px;background:#0a1a2e;border-radius:8px;font-size:12px">
    <span style="white-space:nowrap">📁 Output:</span>
    <span id="outputDirText" style="flex:1;color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">~/Desktop</span>
    <button id="outputDirBtn" style="background:#1e3a5f;border:none;color:#aaccff;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:11px" onclick="changeOutputDir()">Choose…</button>
    <span id="outputDirStatus" style="color:#888;font-size:11px"></span>
  </div>

  <!-- Generate button -->
  <button class="btn-gen" id="genBtn" disabled onclick="generate()">⚡ Generate Speech</button>

  <!-- Output -->
  <div class="output-section" id="outputSection">
    <div class="output-actions">
      <audio id="audioPlayer" controls></audio>
      <a id="saveLink" download>💾 Save</a>
      <button id="saveVoiceBtn" class="save-voice-btn" style="display:none" onclick="saveVoiceProfile()">💾 Save Voice</button>
    </div>
    <div class="error-msg" id="errorMsg"></div>
  </div>
</div>

<script>
const API = '';
let currentModel = 'custom_voice';
let loading = false;
let refAudioFile = '';
let refAudioDuration = 0;

// ── Initialization ──

async function init() {
  // Load speakers with rich info
  try {
    const r = await fetch(API + '/api/speakers');
    const spkData = await r.json();
    const sel = document.getElementById('spkSelect');
    sel.innerHTML = '';
    let firstVal = '';
    const groups = {'Chinese 🇨🇳': [], 'English 🇬🇧': [], 'Japanese 🇯🇵': [], 'Korean 🇰🇷': []};
    for (const [name, info] of Object.entries(spkData)) {
      groups[info.language + ' ' + (info.language === 'Chinese' ? '🇨🇳' : info.language === 'English' ? '🇬🇧' : info.language === 'Japanese' ? '🇯🇵' : '🇰🇷')].push({name, info});
    }
    for (const [group, items] of Object.entries(groups)) {
      if (items.length === 0) continue;
      const grp = document.createElement('optgroup');
      grp.label = group + ' (' + items.length + ')';
      items.forEach(({name, info}) => {
        const opt = document.createElement('option');
        opt.value = name;
        const dialect = info.dialect !== 'Standard' ? ' [' + info.dialect + ']' : '';
        opt.textContent = info.emoji + ' ' + name + dialect + '  —  ' + info.style;
        opt.dataset.info = JSON.stringify(info);
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
      if (!firstVal) firstVal = items[0].name;
    }
    // Show details for first speaker
    // With optgroups, the first option in the first optgroup
    const firstOpt = sel.querySelector('option');
    if (firstOpt) {
      sel.value = firstOpt.value;
    }
    updateSpeakerDetails();

    // Update details on selection
    sel.addEventListener('change', updateSpeakerDetails);
  } catch(e) {}

  // Load languages into both selects
  try {
    const r = await fetch(API + '/api/languages');
    const langs = await r.json();
    for (const id of ['langSelect', 'cloneLangSelect']) {
      const sel = document.getElementById(id);
      sel.innerHTML = '';
      langs.forEach(l => {
        const opt = document.createElement('option');
        opt.value = l; opt.textContent = l;
        if (l === 'Italian') opt.selected = true;
        sel.appendChild(opt);
      });
    }
  } catch(e) {}

  // Load output directory
  loadOutputDir();

  // Load saved voice profiles
  loadProfiles();

  // Default: load CustomVoice
  switchModel('cv');

  // Poll status
  pollStatus();
}

// ── Output directory ──

async function loadOutputDir() {
  try {
    const r = await fetch(API + '/api/output_dir');
    const data = await r.json();
    if (data.output_dir) {
      document.getElementById('outputDirText').textContent = data.output_dir;
    }
  } catch(e) {}
}

function changeOutputDir() {
  // Try native macOS folder picker via pywebview (native app mode)
  if (window.pywebview && window.pywebview.api) {
    window.pywebview.api.pick_folder().then(function(path) {
      if (path && path.trim() !== '') {
        saveOutputDir(path.trim());
      }
    }).catch(function(err) {
      // Fallback to prompt
      const currentPath = document.getElementById('outputDirText').textContent;
      const newPath = prompt('Enter the full path for output:', currentPath);
      if (newPath && newPath.trim() !== '') saveOutputDir(newPath.trim());
    });
  } else {
    // Browser mode: use prompt
    const currentPath = document.getElementById('outputDirText').textContent;
    const newPath = prompt('Enter the full path for output:', currentPath);
    if (newPath && newPath.trim() !== '') saveOutputDir(newPath.trim());
  }
}

async function saveOutputDir(path) {
  const status = document.getElementById('outputDirStatus');
  try {
    const r = await fetch(API + '/api/output_dir', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({output_dir: path}),
    });
    const data = await r.json();
    if (data.error) {
      status.textContent = '❌ ' + data.error;
    } else {
      document.getElementById('outputDirText').textContent = data.output_dir;
      status.textContent = '✅ Saved';
      setTimeout(() => status.textContent = '', 3000);
    }
  } catch(e) {
    status.textContent = '❌ Error saving';
  }
}

// ── Model switching ──

async function switchModel(model) {
  // 'cv' is a family — resolve to specific model id
  if (model === 'cv') {
    model = getCVModelType();
  }
  currentModel = model;

  // Update UI buttons
  document.querySelectorAll('.model-btn').forEach(b => {
    b.classList.toggle('active', (b.dataset.model === 'cv' && model.startsWith('cv_')) || b.dataset.model === model);
  });

  // Show/hide sections
  document.getElementById('cvSection').classList.toggle('section-hidden', !model.startsWith('cv_'));
  document.getElementById('cloneSection').classList.toggle('section-hidden', model !== 'base');

  // Load the model if not loaded
  const status = await getModelStatus();
  if (!status[model] || status[model].status === 'not_loaded') {
    fetch(API + '/api/load_model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: model}),
    });
  }
}

function getCVModelType() {
  return document.getElementById('qualityToggle').checked ? 'cv_0b6' : 'cv_1b7';
}

function updateModelForCV() {
  // When quality toggle changes, switch to the new model
  switchModel('cv');
}

async function getModelStatus() {
  try {
    const r = await fetch(API + '/api/status');
    return await r.json();
  } catch(e) {
    return {};
  }
}

// ── Status polling ──

async function pollStatus() {
  try {
    const data = await getModelStatus();
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    const btn = document.getElementById('genBtn');

    // Determine overall status
    const model = currentModel;
    const status = data[model] || {status: 'not_loaded'};
    const meta = data.model_meta || {};
    const m = meta[model] || {};
    const fullName = m.name || (model.startsWith('cv_') ? 'CustomVoice' : model === 'base' ? 'VoiceClone' : model);

    if (status.status === 'ready') {
      dot.className = 'status-dot ready';
      const mode = model === 'cv_0b6' ? ' ⚡ Fast' : model.startsWith('cv_') ? ' 🎤 HQ' : ' 🧬';
      text.textContent = '✅ ' + fullName + mode + (status.speakers ? ' · ' + status.speakers.length + ' speakers' : '');
      btn.disabled = false;
      document.getElementById('retryBtn').style.display = 'none';
    } else if (status.status === 'error') {
      dot.className = 'status-dot error';
      text.textContent = '❌ ' + fullName + ': ' + (status.error || 'error');
      btn.disabled = true;
      document.getElementById('retryBtn').style.display = 'inline-block';
    } else if (status.status === 'loading') {
      dot.className = 'status-dot loading';
      text.textContent = '⏳ Loading ' + fullName + '…';
      btn.disabled = true;
      document.getElementById('retryBtn').style.display = 'none';
    } else {
      dot.className = 'status-dot inactive';
      const size = model.startsWith('cv_') ? '' : '';
      text.textContent = '○ ' + fullName + ' not loaded — click to load';
      btn.disabled = true;
      document.getElementById('retryBtn').style.display = 'none';
    }
  } catch(e) {
    document.getElementById('statusText').textContent = '⏳ Connecting…';
  }
  setTimeout(pollStatus, 2000);
}

function retryLoad() {
  switchModel(currentModel);
}

// ── Speaker details ──

function updateSpeakerDetails() {
  const sel = document.getElementById('spkSelect');
  const details = document.getElementById('spkDetails');
  const opt = sel.options[sel.selectedIndex];
  if (!opt || !opt.dataset.info) { details.style.display = 'none'; return; }
  try {
    const info = JSON.parse(opt.dataset.info);
    const lang = info.language;
    const flag = lang === 'Chinese' ? '🇨🇳' : lang === 'English' ? '🇬🇧' : lang === 'Japanese' ? '🇯🇵' : lang === 'Korean' ? '🇰🇷' : '🌐';
    details.style.display = 'block';
    details.innerHTML = ''
      + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 16px">'
      + '  <span><b>🎭 Type:</b> ' + info.gender + ', ' + info.age + '</span>'
      + '  <span><b>' + flag + ' Language:</b> ' + lang + (info.dialect !== 'Standard' ? ' (' + info.dialect + ')' : '') + '</span>'
      + '  <span><b>🎯 Style:</b> ' + info.style + '</span>'
      + '  <span><b>⭐ Best for:</b> ' + info.best_for + '</span>'
      + '</div>';
  } catch(e) { details.style.display = 'none'; }
}

// ── Voice Clone: Upload ──

function handleUpload(input) {
  if (!input.files || !input.files[0]) return;
  uploadFile(input.files[0]);
}

function handleDrop(event) {
  this.classList.remove('dragover');
  const files = event.dataTransfer.files;
  if (files.length > 0) uploadFile(files[0]);
}

async function uploadFile(file) {
  const zone = document.getElementById('uploadZone');
  const hint = document.getElementById('uploadHint');
  const info = document.getElementById('uploadInfo');
  const status = document.getElementById('uploadStatus');
  const waveformSec = document.getElementById('waveformSection');
  const trimControls = document.getElementById('trimControls');
  const trimOverlay = document.getElementById('trimOverlay');

  hint.textContent = '⏳ Uploading…';
  info.style.display = 'none';
  status.style.display = 'none';
  waveformSec.style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const r = await fetch(API + '/api/upload_ref_audio', {method: 'POST', body: formData});
    const data = await r.json();
    if (data.error) {
      hint.textContent = '❌ ' + data.error;
      zone.classList.remove('has-file');
    } else {
      refAudioFile = data.filename;
      refAudioDuration = data.duration;
      zone.classList.add('has-file');
      hint.textContent = '✅ ' + file.name;
      info.style.display = 'block';
      info.textContent = '📝 ' + data.duration + 's @ ' + data.sample_rate + 'Hz';

      // Show waveform
      waveformSec.style.display = 'block';
      drawWaveform(data.waveform, data.duration);

      // Handle trimming if needed
      if (data.needs_trim && data.duration > 60) {
        trimControls.style.display = 'block';
        trimOverlay.style.display = 'block';
        const slider = document.getElementById('trimSlider');
        slider.max = Math.max(0, data.duration - 60);
        slider.value = 0;
        document.getElementById('trimWarning').textContent = (data.trim_warning || '');
        updateTrim();
      } else {
        trimControls.style.display = 'none';
        trimOverlay.style.display = 'none';
        // Full audio used
        window._trimStart = 0;
        window._trimEnd = 0;
      }
      status.style.display = 'block';
      status.textContent = data.needs_trim ? '✂️ Use the trim slider below to select your segment' : '✅ Full audio ready (' + data.duration + 's)';
    }
  } catch(e) {
    hint.textContent = '❌ Upload failed: ' + e.message;
    zone.classList.remove('has-file');
  }
}

// ── Waveform drawing ──

function drawWaveform(waveform, duration) {
  const canvas = document.getElementById('waveformCanvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(700, Math.round(rect.width || 700) * 2); // HiDPI
  canvas.height = 100 * 2;
  canvas.style.width = '100%';
  canvas.style.height = '100px';

  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const centerY = h / 2;
  const barWidth = w / waveform.length;

  ctx.clearRect(0, 0, w, h);

  // Draw waveform as filled shape
  ctx.beginPath();
  ctx.moveTo(0, centerY);
  for (let i = 0; i < waveform.length; i++) {
    const x = i * barWidth + barWidth / 2;
    const amp = waveform[i] * (centerY - 4);
    ctx.lineTo(x, centerY - amp);
  }
  ctx.lineTo(w, centerY);
  for (let i = waveform.length - 1; i >= 0; i--) {
    const x = i * barWidth + barWidth / 2;
    const amp = waveform[i] * (centerY - 4);
    ctx.lineTo(x, centerY + amp);
  }
  ctx.closePath();
  ctx.fillStyle = '#4a90d9';
  ctx.fill();

  // Time labels
  ctx.fillStyle = '#667';
  ctx.font = `${Math.round(10 * (w / 700))}px -apple-system, sans-serif`;
  ctx.fillText('0s', 4, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(duration + 's', w - 4, h - 4);
  ctx.textAlign = 'left';

  // Store waveform data for trim overlay
  window._waveformData = { waveform, duration };
}

// ── Trim handling ──

function updateTrim() {
  const slider = document.getElementById('trimSlider');
  const start = parseFloat(slider.value);
  const windowLen = 60; // seconds
  const end = start + windowLen;

  document.getElementById('trimStartLabel').textContent = start.toFixed(1);
  document.getElementById('trimEndLabel').textContent = end.toFixed(1);
  document.getElementById('trimDuration').textContent = windowLen.toFixed(0);

  // Update green overlay on canvas
  const canvas = document.getElementById('waveformCanvas');
  const wf = window._waveformData;
  if (!wf || !canvas) return;

  const totalDur = wf.duration;
  const startFrac = Math.max(0, start / totalDur);
  const endFrac = Math.min(1, end / totalDur);

  const trimRegion = document.getElementById('trimRegion');
  trimRegion.style.left = (startFrac * 100) + '%';
  trimRegion.style.width = ((endFrac - startFrac) * 100) + '%';

  window._trimStart = start;
  window._trimEnd = Math.min(end, totalDur);
}

// ── Voice Clone: ICL toggle ──

function toggleICL() {
  const icl = document.getElementById('iclToggle').checked;
  document.getElementById('refTextSection').classList.toggle('section-hidden', !icl);
}

// ── Voice Profiles ──

let profilesVisible = false;

async function loadProfiles() {
  try {
    const r = await fetch(API + '/api/profiles/list');
    const data = await r.json();
    const profiles = data.profiles || [];
    const section = document.getElementById('savedVoicesSection');
    const grid = document.getElementById('profileGrid');
    const count = document.getElementById('profileCount');
    const noMsg = document.getElementById('noProfiles');

    count.textContent = '(' + profiles.length + ')';
    grid.innerHTML = '';

    if (profiles.length === 0) {
      section.style.display = 'block';
      noMsg.style.display = 'block';
      grid.style.display = 'none';
      return;
    }

    section.style.display = 'block';
    noMsg.style.display = 'none';
    grid.style.display = profilesVisible ? 'grid' : 'none';

    profiles.forEach(p => {
      const card = document.createElement('div');
      card.className = 'profile-card';
      const date = new Date(p.created || 0);
      const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
      const mode = p.x_vector_only === false ? 'ICL' : 'x-vector';
      card.innerHTML = ''
        + '<div class="name">🧬 ' + p.name + '</div>'
        + '<div class="meta">' + dateStr + ' · ' + mode + '</div>'
        + '<button class="delete-btn" onclick="event.stopPropagation();deleteProfile(\'' + p.name + '\')">✕</button>';
      card.onclick = function() { loadProfile(p); };
      grid.appendChild(card);
    });
  } catch(e) {}
}

function toggleProfiles() {
  profilesVisible = !profilesVisible;
  document.getElementById('profileGrid').style.display = profilesVisible ? 'grid' : 'none';
  loadProfiles(); // refresh
}

async function saveVoiceProfile() {
  if (!refAudioFile) return;
  const name = prompt('Name this voice profile:', '');
  if (!name || name.trim() === '') return;

  const payload = {
    name: name.trim(),
    ref_audio_file: refAudioFile,
    ref_text: document.getElementById('refText').value.trim(),
    x_vector_only: !document.getElementById('iclToggle').checked,
    trim_start: window._trimStart || 0,
    trim_end: window._trimEnd || 0,
  };

  try {
    const r = await fetch(API + '/api/profiles/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (data.error) {
      document.getElementById('errorMsg').textContent = '❌ ' + data.error;
    } else {
      document.getElementById('errorMsg').textContent = '✅ Voice profile "' + name.trim() + '" saved!';
      profilesVisible = true;
      loadProfiles();
    }
  } catch(e) {
    document.getElementById('errorMsg').textContent = '❌ Error saving profile';
  }
}

async function loadProfile(profile) {
  // Set form fields from profile
  document.getElementById('refText').value = profile.ref_text || '';
  document.getElementById('iclToggle').checked = profile.x_vector_only === false;
  toggleICL();

  // Set the ref audio file — it's stored in the profiles dir
  refAudioFile = profile.ref_audio_file;
  refAudioDuration = 0;

  // Update UI to show loaded profile
  const zone = document.getElementById('uploadZone');
  zone.classList.add('has-file');
  document.getElementById('uploadHint').textContent = '✅ Profile: ' + profile.name;
  document.getElementById('uploadInfo').style.display = 'block';
  document.getElementById('uploadInfo').textContent = '📂 Loaded from saved voices';
  document.getElementById('uploadStatus').style.display = 'block';
  document.getElementById('uploadStatus').textContent = '✅ Ready to generate with saved voice';

  // Set trim if saved
  if (profile.trim_start !== undefined) window._trimStart = profile.trim_start;
  if (profile.trim_end !== undefined) window._trimEnd = profile.trim_end;

  // Highlight active card
  document.querySelectorAll('.profile-card').forEach(c => c.classList.remove('active'));
  event.target.closest('.profile-card')?.classList.add('active');
}

async function deleteProfile(name) {
  if (!confirm('Delete voice profile "' + name + '"?')) return;
  try {
    await fetch(API + '/api/profiles/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name}),
    });
    loadProfiles();
  } catch(e) {}
}

// ── Generation ──

async function generate() {
  if (loading) return;
  const text = document.getElementById('textInput').value.trim();
  if (!text) { showError('Please enter some text'); return; }

  // Validate voice clone
  if (currentModel === 'base' && !refAudioFile) {
    showError('Please upload a reference audio sample first');
    return;
  }
  if (currentModel === 'base' && document.getElementById('iclToggle').checked) {
    const refText = document.getElementById('refText').value.trim();
    if (!refText) {
      showError('ICL mode needs the reference audio transcript');
      return;
    }
  }

  const btn = document.getElementById('genBtn');
  const outputSec = document.getElementById('outputSection');
  const errorMsg = document.getElementById('errorMsg');

  loading = true;
  btn.disabled = true;
  btn.textContent = '⏳ Generating… (~30-90s)';
  btn.className = 'btn-gen generating';
  outputSec.classList.remove('show');
  errorMsg.textContent = '';

  // Build request
  const payload = {
    model: currentModel,
    text: text,
    language: currentModel.startsWith('cv_')
        ? document.getElementById('langSelect').value
        : document.getElementById('cloneLangSelect').value,
    instruct: currentModel.startsWith('cv_')
        ? document.getElementById('instructInput')?.value || ''
        : document.getElementById('cloneInstruct')?.value || '',
  };

  if (currentModel.startsWith('cv_')) {
    payload.speaker = document.getElementById('spkSelect').value;
  } else {
    payload.ref_audio_file = refAudioFile;
    payload.ref_text = document.getElementById('refText').value.trim();
    payload.x_vector_only = !document.getElementById('iclToggle').checked;
    payload.trim_start = window._trimStart || 0;
    payload.trim_end = window._trimEnd || 0;
  }

  try {
    const r = await fetch(API + '/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    let data;
    try { data = await r.json(); } catch(e) { data = {error: 'Invalid response'}; }

    if (data.error) {
      showError(data.error);
    } else {
      const player = document.getElementById('audioPlayer');
      const saveLink = document.getElementById('saveLink');
      const saveVoiceBtn = document.getElementById('saveVoiceBtn');
      const url = API + '/output/' + data.filename;
      player.src = url;
      player.load();
      saveLink.href = url;
      saveLink.textContent = '💾 Save (' + data.duration + 's)';
      // Show "Save Voice" button only for voice clone mode
      if (currentModel === 'base' || data.model === 'base') {
        saveVoiceBtn.style.display = 'inline-block';
      } else {
        saveVoiceBtn.style.display = 'none';
      }
      outputSec.classList.add('show');
      errorMsg.textContent = '';
    }
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    loading = false;
    btn.disabled = false;
    btn.textContent = '⚡ Generate Speech';
    btn.className = 'btn-gen';
  }
}

function showError(msg) {
  document.getElementById('errorMsg').textContent = '❌ ' + msg;
  document.getElementById('outputSection').classList.add('show');
}

init();
</script>
</body>
</html>"""


# ── Server entry point ──────────────────────────────────────────────────────

def open_browser():
    time.sleep(2)
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass


def main():
    print(f"🎙️  Qwen3 TTS — CustomVoice + Voice Clone")
    print(f"   Server: http://localhost:{PORT}")
    print(f"   Output: {get_output_dir()}")
    print(f"   Temp (clones): {TEMP_DIR}")
    print(f"   Click a model to load it on demand.")
    print()

    # Open browser
    threading.Thread(target=open_browser, daemon=True).start()

    # Quiet logs
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
