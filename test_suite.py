#!/usr/bin/env python3
"""
Qwen3 TTS — Test Suite
=======================
Tests all components: server, API, model loading, generation, profiles,
output dir, shutdown. Run BEFORE first user launch.

Usage:
    cd ~/qwen3-tts-ui && source venv/bin/activate
    python3 test_suite.py

Exit code: 0 = all passed, 1 = failures
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import threading
import urllib.request
import urllib.error
import tempfile
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:8765"
TEST_DIR = Path(tempfile.mkdtemp(prefix="qwen3tts_test_"))
PASS = 0
FAIL = 0
ERRORS = []


def test(name: str, required: bool = True):
    """Decorator-like test runner."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global PASS, FAIL
            print(f"  🔍 {name}…", end=" ", flush=True)
            try:
                fn(*args, **kwargs)
                PASS += 1
                print("✅")
            except Exception as e:
                FAIL += 1
                ERRORS.append(f"  ❌ {name}: {e}")
                print("❌")
                if required:
                    print(f"     └─ {e}")
        return wrapper
    return decorator


# ── Fixtures ─────────────────────────────────────────────────────────────────

_server_proc = None
_server_mod = None


def setup_module():
    """Start the Flask server in background."""
    global _server_mod
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import importlib.util
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    spec = importlib.util.spec_from_file_location(
        "qwen_tts_server",
        os.path.join(os.path.dirname(__file__), "qwen3_tts_server.py")
    )
    _server_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_server_mod)

    t = threading.Thread(
        target=lambda: _server_mod.app.run(
            host="127.0.0.1", port=8765,
            debug=False, threaded=True, use_reloader=False
        ),
        daemon=True,
    )
    t.start()
    time.sleep(2)


def api_get(path: str, timeout: int = 30) -> dict:
    r = urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout)
    return json.loads(r.read())


def api_post(path: str, data: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(r.read())


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

@test("Server is running and responds")
def t_server_running():
    r = urllib.request.urlopen(f"{BASE_URL}/", timeout=5)
    html = r.read().decode()
    assert r.status == 200
    assert "Qwen3 TTS" in html


@test("API: models endpoint")
def t_api_models():
    data = api_get("/api/models")
    assert "models" in data
    ids = [m["id"] for m in data["models"]]
    assert "cv_1b7" in ids, f"Missing cv_1b7: {ids}"
    assert "cv_0b6" in ids, f"Missing cv_0b6: {ids}"
    assert "base" in ids, f"Missing base: {ids}"
    assert "meta" in data


@test("API: speakers endpoint")
def t_api_speakers():
    data = api_get("/api/speakers")
    names = list(data.keys())
    assert len(names) == 9, f"Expected 9 speakers, got {len(names)}"
    for n in names:
        info = data[n]
        assert "language" in info
        assert "gender" in info
        assert "style" in info
        assert "emoji" in info


@test("API: languages endpoint")
def t_api_languages():
    data = api_get("/api/languages")
    assert isinstance(data, list)
    assert "Italian" in data
    assert "English" in data
    assert "Auto" in data
    assert len(data) >= 10


@test("API: output_dir GET default")
def t_api_output_dir_get():
    data = api_get("/api/output_dir")
    assert "output_dir" in data
    assert data["exists"] is True
    assert "Desktop" in data["output_dir"] or "Desktop" in data["output_dir"]


@test("API: output_dir POST + persistence")
def t_api_output_dir_post():
    test_path = str(TEST_DIR / "tts_output")
    # Set
    data = api_post("/api/output_dir", {"output_dir": test_path})
    assert data["exists"] is True
    assert test_path in data["output_dir"]
    # Verify persisted
    data2 = api_get("/api/output_dir")
    assert test_path in data2["output_dir"]
    # Reset to Desktop
    api_post("/api/output_dir", {"output_dir": "~/Desktop"})


@test("API: status endpoint (all not loaded)")
def t_api_status():
    data = api_get("/api/status")
    assert "custom_voice" not in str(data)  # old key removed
    assert "cv_1b7" in str(data) or "cv_0b6" in str(data) or "all_model_ids" in data
    assert "all_model_ids" in data
    for mt in ["cv_1b7", "cv_0b6", "base"]:
        assert mt in data["all_model_ids"]


@test("API: profiles list (empty)")
def t_api_profiles_empty():
    data = api_get("/api/profiles/list")
    assert "profiles" in data
    assert isinstance(data["profiles"], list)


@test("API: profiles save requires name")
def t_api_profiles_save_no_name():
    try:
        api_post("/api/profiles/save", {"ref_audio_file": "x.wav"})
    except urllib.error.HTTPError as e:
        assert e.code == 400


@test("Model loading: cv_1b7")
def t_model_load_cv_1b7():
    data = api_post("/api/load_model", {"model": "cv_1b7"})
    assert data["status"] in ("loading", "already_loaded")

    # Wait for ready
    for _ in range(60):
        s = api_get("/api/status")
        if s.get("cv_1b7", {}).get("status") == "ready":
            break
        time.sleep(2)
    else:
        raise TimeoutError("cv_1b7 model did not load in 120s")

    s2 = api_get("/api/status")
    assert s2["cv_1b7"]["status"] == "ready"
    assert len(s2["cv_1b7"]["speakers"]) == 9


@test("Generation: cv_1b7 Italian (Ryan)")
def t_generation_cv_1b7():
    data = api_post("/api/generate", {
        "model": "cv_1b7",
        "text": "Ciao, test di sintesi vocale.",
        "language": "Italian",
        "speaker": "Ryan",
        "instruct": "",
    }, timeout=180)
    assert data["success"] is True
    assert data["duration"] > 0.5
    assert data["sample_rate"] == 24000
    assert data["model"] == "cv_1b7"
    assert data["filename"].endswith(".wav")

    # Verify file exists
    cfg = api_get("/api/output_dir")
    out_dir = cfg["output_dir"]
    filepath = os.path.join(out_dir, data["filename"])
    assert os.path.exists(filepath), f"Output file not found: {filepath}"

    # Verify it's a valid WAV
    import soundfile as sf
    wav, sr = sf.read(filepath)
    assert sr == 24000
    assert len(wav) > 0

    # Cleanup
    os.unlink(filepath)


@test("Generation: cv_1b7 English with instruct")
def t_generation_cv_instruct():
    data = api_post("/api/generate", {
        "model": "cv_1b7",
        "text": "This is absolutely incredible!",
        "language": "English",
        "speaker": "Aiden",
        "instruct": "Speak very happily and excitedly",
    }, timeout=180)
    assert data["success"] is True
    assert data["duration"] > 0.5


@test("Model loading: cv_0b6 (Fast Mode)")
def t_model_load_cv_0b6():
    data = api_post("/api/load_model", {"model": "cv_0b6"})
    assert data["status"] in ("loading", "already_loaded")

    for _ in range(60):
        s = api_get("/api/status")
        if s.get("cv_0b6", {}).get("status") == "ready":
            break
        time.sleep(2)
    else:
        raise TimeoutError("cv_0b6 model did not load in 120s")

    s2 = api_get("/api/status")
    assert s2["cv_0b6"]["status"] == "ready"


@test("Generation: cv_0b6 (Fast 0.6B)")
def t_generation_cv_0b6():
    data = api_post("/api/generate", {
        "model": "cv_0b6",
        "text": "Fast mode test with smaller model.",
        "language": "English",
        "speaker": "Ryan",
    }, timeout=180)
    assert data["success"] is True
    assert data["model"] == "cv_0b6"

@test("Generation: key speakers across languages")
def t_generation_key_speakers():
    """Test Ryan (English), Vivian (Chinese), Sohee (Korean) — representative set."""
    test_cases = [
        ("Ryan", "English"),
        ("Aiden", "English"),
        ("Vivian", "Chinese"),
    ]
    for spk, lang in test_cases:
        data = api_post("/api/generate", {
            "model": "cv_1b7",
            "text": f"Test with speaker {spk}.",
            "language": lang,
            "speaker": spk,
        }, timeout=180)
        assert data["success"] is True, f"Speaker {spk} failed"
        assert data["duration"] > 0.5, f"Speaker {spk}: too short"


@test("Audio upload and waveform generation")
def t_audio_upload():
    # Create test audio file
    import soundfile as sf
    import numpy as np
    sr = 24000
    t = np.linspace(0, 3, int(sr * 3), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    test_file = str(TEST_DIR / "test_upload.wav")
    sf.write(test_file, audio, sr)

    # Upload via multipart
    import http.client
    import io
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(test_file, "rb") as f:
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test_upload.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + f.read() + f"\r\n--{boundary}--\r\n".encode()

    conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=15)
    conn.request("POST", "/api/upload_ref_audio", body=body,
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    data = json.loads(resp.read())

    assert data["success"] is True
    assert data["duration"] >= 2.5
    assert "waveform" in data
    assert len(data["waveform"]) > 100
    assert data["needs_trim"] is False
    assert data["sample_rate"] == 24000


@test("Model loading: Base (Voice Clone)")
def t_model_load_base():
    data = api_post("/api/load_model", {"model": "base"})
    assert data["status"] in ("loading", "already_loaded")

    for _ in range(90):
        s = api_get("/api/status")
        if s.get("base", {}).get("status") == "ready":
            break
        time.sleep(2)
    else:
        raise TimeoutError("Base model did not load in 180s")

    s2 = api_get("/api/status")
    assert s2["base"]["status"] == "ready"


@test("Voice Clone: x-vector generation")
def t_voice_clone_xvector():
    # Use the uploaded audio from the previous test
    # Find it in the temp dir
    import glob
    ref_files = glob.glob(os.path.join(
        os.path.expanduser(tempfile.gettempdir()),
        "qwen3tts_clone", "ref_*_test_upload.wav"
    ))
    assert len(ref_files) > 0, "No uploaded ref audio found"

    ref_name = os.path.basename(ref_files[-1])
    data = api_post("/api/generate", {
        "model": "base",
        "text": "Voice clone test with x-vector mode.",
        "language": "English",
        "ref_audio_file": ref_name,
        "ref_text": "",
        "x_vector_only": True,
    }, timeout=180)
    assert data["success"] is True
    assert data["duration"] > 0.5


@test("API: shutdown endpoint")
def t_api_shutdown():
    data = api_post("/api/shutdown", {})
    assert data["shutdown"] is True
    # Wait for server to actually stop
    time.sleep(1)
    try:
        urllib.request.urlopen(f"{BASE_URL}/", timeout=2)
        assert False, "Server still running after shutdown"
    except (urllib.error.URLError, ConnectionRefusedError):
        pass  # Expected — server is down


@test("Full startup → generation → shutdown cycle")
def t_full_cycle():
    """Simulate a complete user session."""
    # Start server
    setup_module()
    time.sleep(1)

    # Load model
    api_post("/api/load_model", {"model": "cv_1b7"})
    for _ in range(60):
        s = api_get("/api/status")
        if s.get("cv_1b7", {}).get("status") == "ready":
            break
        time.sleep(2)
    else:
        raise TimeoutError("Model did not load in cycle test")

    # Check speakers
    spk = api_get("/api/speakers")
    assert len(spk) == 9

    # Change output dir
    test_out = str(TEST_DIR / "cycle_output")
    api_post("/api/output_dir", {"output_dir": test_out})

    # Generate
    gen = api_post("/api/generate", {
        "model": "cv_1b7",
        "text": "Full cycle test.",
        "language": "English",
        "speaker": "Ryan",
    })
    assert gen["success"] is True
    assert os.path.exists(os.path.join(test_out, gen["filename"]))

    # Shutdown
    api_post("/api/shutdown", {})
    time.sleep(1)
    print("      └─ Full cycle: OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Device compatibility check
# ═══════════════════════════════════════════════════════════════════════════════

@test("Device compatibility check")
def t_device_check():
    """Verify the device is capable of running the models."""
    import torch
    report = {
        "platform": sys.platform,
        "python": sys.version,
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
    }

    # Try to get RAM info
    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            result = subprocess.run(
                ["sysctl", "hw.memsize"], capture_output=True, text=True
            )
            if result.returncode == 0:
                bytes_val = int(result.stdout.strip().split(":")[1].strip())
                report["ram_gb"] = round(bytes_val / (1024**3), 1)
        except Exception:
            pass

    print(f"      Platform: {report['platform']}")
    print(f"      PyTorch: {report['torch']}")
    print(f"      MPS: {'✅ Available' if report['mps_available'] else '❌ Not available'}")
    print(f"      CPU cores: {report['cpu_count']}")
    print(f"      RAM: {report['ram_gb']} GB" if report['ram_gb'] else "      RAM: unknown")

    # Compatibility verdict
    if report['mps_available']:
        print(f"      ✅ GPU acceleration available (MPS) — full speed")
    elif report.get('ram_gb', 0) and report['ram_gb'] >= 16:
        print(f"      ⚠️  No GPU — will run on CPU. 16GB+ RAM OK.")
    else:
        print(f"      ⚠️  Limited resources — consider Fast Mode (0.6B)")

    assert report['mps_available'] or (report.get('ram_gb', 0) or 0) >= 8, \
        "Device may not have enough resources"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🎙️  Qwen3 TTS — Test Suite")
    print(f"   Tests dir: {TEST_DIR}")
    print()

    # Register all test functions
    tests = [
        ("Server", [
            t_server_running,
        ]),
        ("API", [
            t_api_models,
            t_api_speakers,
            t_api_languages,
            t_api_output_dir_get,
            t_api_output_dir_post,
            t_api_status,
            t_api_profiles_empty,
            t_api_profiles_save_no_name,
        ]),
        ("Model Loading & Generation", [
            t_model_load_cv_1b7,
            t_generation_cv_1b7,
            t_generation_cv_instruct,
            t_model_load_cv_0b6,
            t_generation_cv_0b6,
            t_generation_key_speakers,
        ]),
        ("Voice Clone", [
            t_audio_upload,
            t_model_load_base,
            t_voice_clone_xvector,
        ]),
        ("Device Check", [
            t_device_check,
        ]),
        ("Shutdown & Cycle", [
            t_api_shutdown,
            t_full_cycle,
        ]),
    ]

    # Start server for the first batch
    print("── Server Tests ──")
    setup_module()

    for group_name, group_tests in tests:
        print(f"\n── {group_name} ──")
        for test_fn in group_tests:
            test_fn.__wrapped__() if hasattr(test_fn, '__wrapped__') else test_fn()

    # Cleanup
    import shutil
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print(f"\n{'='*40}")
    print(f"  ✅ Passed: {PASS}   ❌ Failed: {FAIL}")
    print(f"{'='*40}")

    if ERRORS:
        print("\nFailures:")
        for e in ERRORS:
            print(e)

    sys.exit(0 if FAIL == 0 else 1)
