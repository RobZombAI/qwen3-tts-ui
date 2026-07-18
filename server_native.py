#!/usr/bin/env python3
"""
Qwen3 TTS — Native macOS App
=============================
Professional TTS app with:
- CustomVoice (3 models, 9 speakers)
- Voice Clone (upload or record audio)
- Live progress logs
- Native pywebview window (no browser)
- Clean shutdown

Usage: ./venv/bin/python3 app_native.py
"""

import os, sys, json, time, threading, logging, uuid, tempfile
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, render_template_string, send_file, jsonify

app = Flask(__name__)

OUTPUT_DIR = os.path.expanduser("~/Desktop")
TEMP_DIR = os.path.join(tempfile.gettempdir(), "qwen3tts_clone")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

MODEL_IDS = {
    "cv_1b7": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "cv_0b6": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base":   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

MODEL_NAMES = {
    "cv_1b7": "CustomVoice 1.7B HQ",
    "cv_0b6": "CustomVoice 0.6B Fast",
    "base":   "Voice Clone 1.7B",
}

SPEAKERS = {
    "Vivian": "🇨🇳 Bright young female",
    "Serena": "🇨🇳 Warm young female",
    "Uncle_Fu": "🇨🇳 Deep mellow male",
    "Dylan": "🇨🇳 Beijing male",
    "Eric": "🇨🇳 Sichuan male",
    "Ryan": "🇬🇧 Dynamic male",
    "Aiden": "🇬🇧 Sunny American male",
    "Ono_Anna": "🇯🇵 Playful Japanese female",
    "Sohee": "🇰🇷 Warm Korean female",
}

LANGUAGES = ["Auto","Chinese","English","Japanese","Korean","German","French","Russian","Portuguese","Spanish","Italian"]

# ── Model manager ──
_model = None
_model_type = None
_model_ready = threading.Event()
_loading_msg = "💡 Click a model above to load it"
_loading_log = []

def log(msg):
    global _loading_msg, _loading_log
    _loading_msg = msg
    _loading_log.append(msg)
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


def load_model_thread(mt):
    global _model, _model_type
    name = MODEL_NAMES.get(mt, mt)
    log(f"🔃 {name}: initializing…")
    try:
        import torch
        from qwen_tts import Qwen3TTSModel

        log(f"🔃 Importing dependencies…")
        if torch.backends.mps.is_available():
            device = "mps"; dtype = torch.bfloat16; attn = "sdpa"
            log(f"⚙️  Device: MPS (Apple Silicon GPU)")
        else:
            device = "cpu"; dtype = torch.float32; attn = "eager"
            log(f"⚙️  Device: CPU (no GPU detected)")

        log(f"📦 Loading model weights…")
        _model = Qwen3TTSModel.from_pretrained(MODEL_IDS[mt], dtype=dtype, attn_implementation=attn)
        log(f"✅ Loaded into memory")

        if device != "cpu":
            log(f"📤 Moving to {device.upper()}…")
            _model.model = _model.model.to(device)
            log(f"✅ On {device.upper()}")
        _model.device = torch.device(device)

        log(f"🔥 Warming up…")
        try:
            if mt.startswith("cv_"):
                _model.generate_custom_voice(text="Hi.", language="English", speaker="Ryan", max_new_tokens=2)
            else:
                _model.generate_voice_clone(text="Hi.", language="English", max_new_tokens=2)
            log(f"✅ Warm-up complete")
        except Exception as e:
            log(f"⚠️ Warm-up: {e}")

        _model_type = mt
        _model_ready.set()
        log(f"✅ {name} ready! You can now generate.")
    except Exception as e:
        log(f"❌ Failed: {e}")


# ── HTML UI ──
HTML = """<!DOCTYPE html>
<html>
<head><title>Qwen3 TTS</title>
<meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:30px;max-width:720px;margin:auto}
h1{font-size:26px;margin-bottom:4px;color:#f0f0ff}
.sub{color:#8888aa;font-size:13px;margin-bottom:20px}
.card{background:#16213e;border-radius:16px;padding:24px;margin-bottom:16px}
label{display:block;font-size:13px;font-weight:600;color:#aab;margin-bottom:4px;margin-top:12px}
label:first-child{margin-top:0}
select,input,textarea{width:100%;background:#0f3460;border:1px solid #1a4a7a;color:#e0e0e0;border-radius:8px;padding:10px;font-size:14px;outline:none}
textarea{height:80px;resize:vertical}
audio{width:100%;margin-top:12px}
.log-area{background:#0a1a2e;border-radius:8px;padding:12px;margin-top:12px;font-family:monospace;font-size:11px;max-height:200px;overflow-y:auto;line-height:1.6}
.log-area .ok{color:#4ade80}.log-area .info{color:#88aacc}.log-area .warn{color:#f0a500}.log-area .err{color:#f87171}.log-area .step{color:#8888aa}
#status{padding:12px;background:#0f3460;border-radius:10px;margin:12px 0;font-size:13px;min-height:20px}
.section-hidden{display:none}
.btn-gen{width:100%;padding:14px;border:none;border-radius:10px;font-size:16px;font-weight:700;cursor:pointer;background:linear-gradient(135deg,#1a6b3c,#228b4f);color:#fff;margin-top:16px}
.btn-gen:disabled{opacity:.4;cursor:not-allowed}
.btn-quit{width:100%;padding:10px;border:none;border-radius:8px;background:#3a1a1a;color:#d87171;font-size:13px;cursor:pointer;margin-top:8px}
.btn-quit:hover{background:#5a2a2a}
.btn-rec{border:none;padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;margin:4px 4px 0 0}
.btn-rec.rec{background:#6b1a1a;color:#ff6b6b}.btn-rec.rec:hover{background:#8a2a2a}
.btn-rec.idle{background:#1a3a6b;color:#6baaff}.btn-rec.idle:hover{background:#2a4a8a}
.btn-rec:disabled{opacity:.4;cursor:not-allowed}
.clone-audio-info{background:#0a1a2e;border-radius:8px;padding:10px;margin-top:8px;font-size:12px;color:#88aacc;display:none}
input[type=file]{width:100%;background:#0f3460;border:1px solid #1a4a7a;color:#e0e0e0;border-radius:8px;padding:8px;font-size:12px;outline:none;margin-top:4px}
</style>
</head>
<body>
<h1>🎙️ Qwen3 TTS</h1>
<div class="sub">Select a model → Load → Generate. Close window to quit.</div>

<div class="card" style="text-align:center">
<button onclick="loadModel('cv_1b7')" class="mbtn" data-m="cv_1b7" style="background:#1a6b3c;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">🎤 CustomVoice 1.7B HQ</button>
<button onclick="loadModel('cv_0b6')" class="mbtn" data-m="cv_0b6" style="background:#b8860b;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">⚡ Fast Mode 0.6B</button>
<button onclick="loadModel('base')" class="mbtn" data-m="base" style="background:#4a4a8a;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">🧬 Voice Clone</button>
<div id="status">💡 Click a model to load</div>
<div id="logArea" class="log-area" style="display:none"></div>
</div>

<!-- CustomVoice section -->
<div id="cvSection" style="display:block">
<form class="card" action="/generate" method="post" id="cvForm">
<input type="hidden" name="mode" value="cv">
<label>Text</label>
<textarea name="text" required>Ciao! Questo è un test di sintesi vocale.</textarea>
<label>Language</label>
<select name="language">""" + "".join(f'<option{" selected" if l=="Italian" else ""}>{l}</option>' for l in LANGUAGES) + """</select>
<label>Speaker</label>
<select name="speaker">""" + "".join(f'<option value="{s}">{n}</option>' for s,n in SPEAKERS.items()) + """</select>
<label>Style (optional)</label>
<input name="instruct" placeholder="e.g. 'Speak happily'">
</form>
</div>

<!-- Voice Clone section -->
<div id="cloneSection" style="display:none">
<div class="card">
<label>Reference Audio</label>
<input type="file" id="audioFile" accept="audio/*" onchange="uploadAudio(this.files[0])">
<div style="margin-top:8px;display:flex;align-items:center;gap:8px">
<button id="recBtn" class="btn-rec idle" onclick="toggleRecord()">🎤 Record</button>
<span id="recTimer" style="font-size:12px;color:#8888aa">0s</span>
</div>
<div id="cloneAudioInfo" class="clone-audio-info">No audio loaded</div>

<label style="margin-top:16px">Reference Text (for ICL mode)</label>
<textarea id="refText" style="height:60px" placeholder="Transcribe the audio for better quality (optional)"></textarea>

<label>Mode</label>
<select id="cloneMode">
<option value="x_vector">x-vector (faster, less data needed)</option>
<option value="icl">ICL (better quality, needs transcript)</option>
</select>

<label>Text to Speak</label>
<textarea id="cloneText" style="height:80px">Ciao! Questa è la mia voce clonata.</textarea>
<label>Language</label>
<select id="cloneLang">""" + "".join(f'<option{" selected" if l=="Italian" else ""}>{l}</option>' for l in LANGUAGES) + """</select>

<button class="btn-gen" onclick="generateClone()" disabled>🧬 Generate Voice Clone</button>
</div>
</div>

<div id="output" class="card" style="display:none"><audio id="player" controls></audio></div>
<button class="btn-quit" onclick="quitApp()">⏻ Close App</button>

<script>
let currentModel = 'cv_1b7';
let refAudioFile = '';
let refAudioDuration = 0;
let mediaRecorder = null;
let recChunks = [];
let recTimer = null;
let recSeconds = 0;

function loadModel(m) {
  currentModel = m;
  document.getElementById('logArea').style.display = 'block';
  document.getElementById('logArea').innerHTML = '<div class="step">⏳ Loading...</div>';
  document.getElementById('status').textContent = '⏳ Loading...';
  document.querySelector('.btn-gen').disabled = true;

  // Toggle sections
  const isClone = m === 'base';
  document.getElementById('cvSection').style.display = isClone ? 'none' : 'block';
  document.getElementById('cloneSection').style.display = isClone ? 'block' : 'none';

  fetch('/load?m=' + m);
}

function quitApp() {
  if (window.pywebview && window.pywebview.api) {
    window.pywebview.api.shutdown();
  } else {
    fetch('/quit').then(() => window.close());
  }
}

// ── Audio Recording ──

async function toggleRecord() {
  const btn = document.getElementById('recBtn');
  const timer = document.getElementById('recTimer');
  if (btn.classList.contains('rec')) {
    // Stop recording
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    clearInterval(recTimer);
    btn.textContent = '🎤 Record';
    btn.className = 'btn-rec idle';
    btn.disabled = true;
    timer.textContent = '⏳ Processing...';
  } else {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      recChunks = [];
      recSeconds = 0;
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recChunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(recChunks, { type: 'audio/webm' });
        await uploadRecordedBlob(blob);
        btn.disabled = false;
      };
      mediaRecorder.start(100); // collect data every 100ms
      btn.textContent = '⏹ Stop';
      btn.className = 'btn-rec rec';
      timer.textContent = '0s';
      recTimer = setInterval(() => {
        recSeconds++;
        timer.textContent = recSeconds + 's';
      }, 1000);
    } catch(e) {
      alert('Microphone access denied. Please allow microphone permissions.');
    }
  }
}

async function uploadRecordedBlob(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'recording.webm');
  formData.append('type', 'record');
  try {
    const r = await fetch('/api/upload_audio', { method: 'POST', body: formData });
    const d = await r.json();
    if (d.success) {
      refAudioFile = d.filename;
      refAudioDuration = d.duration_sec || 0;
      document.getElementById('cloneAudioInfo').style.display = 'block';
      document.getElementById('cloneAudioInfo').textContent = '✅ Recorded audio: ' + d.duration_sec.toFixed(1) + 's';
      document.getElementById('recTimer').textContent = '✅ Saved';
    } else {
      document.getElementById('recTimer').textContent = '❌ Failed';
    }
  } catch(e) {
    document.getElementById('recTimer').textContent = '❌ Upload error';
  }
}

// ── Audio Upload ──

async function uploadAudio(file) {
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  formData.append('type', 'upload');
  document.getElementById('cloneAudioInfo').style.display = 'block';
  document.getElementById('cloneAudioInfo').textContent = '⏳ Uploading...';
  try {
    const r = await fetch('/api/upload_audio', { method: 'POST', body: formData });
    const d = await r.json();
    if (d.success) {
      refAudioFile = d.filename;
      refAudioDuration = d.duration_sec || 0;
      document.getElementById('cloneAudioInfo').textContent = '✅ ' + file.name + ' (' + d.duration_sec.toFixed(1) + 's)';
    } else {
      document.getElementById('cloneAudioInfo').textContent = '❌ Upload failed';
    }
  } catch(e) {
    document.getElementById('cloneAudioInfo').textContent = '❌ Upload error';
  }
}

// ── Voice Clone Generation ──

async function generateClone() {
  const text = document.getElementById('cloneText').value.trim();
  if (!text) { alert('Enter text to speak'); return; }
  if (!refAudioFile) { alert('Upload or record a reference audio first'); return; }
  document.querySelector('.btn-gen').disabled = true;
  try {
    const r = await fetch('/api/generate_clone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        language: document.getElementById('cloneLang').value,
        ref_audio_file: refAudioFile,
        ref_text: document.getElementById('refText').value.trim() || '',
        mode: document.getElementById('cloneMode').value,
      })
    });
    const d = await r.json();
    if (d.success) {
      document.getElementById('output').style.display = 'block';
      document.getElementById('player').src = '/output/' + d.filename;
      document.querySelector('.btn-gen').disabled = false;
    } else {
      alert('Generation failed: ' + (d.error || 'unknown error'));
      document.querySelector('.btn-gen').disabled = false;
    }
  } catch(e) {
    alert('Error: ' + e.message);
    document.querySelector('.btn-gen').disabled = false;
  }
}

// ── Status Polling ──

setInterval(async () => {
  try {
    let r = await fetch('/api_status');
    let d = await r.json();
    let s = document.getElementById('status');
    if (d.loading) {
      s.textContent = '⏳ ' + d.msg;
    } else if (d.model) {
      s.style.color = '#4ade80';
      s.textContent = '✅ ' + d.model + ' ready';
      document.querySelector('.btn-gen').disabled = false;
    }
    if (d.log && d.log.length > 0) {
      let la = document.getElementById('logArea');
      la.style.display = 'block';
      la.innerHTML = d.log.map(function(l) {
        let cls = 'info';
        if (l.includes('✅')) cls = 'ok';
        else if (l.includes('⚠️')) cls = 'warn';
        else if (l.includes('❌')) cls = 'err';
        else if (l.includes('📦')||l.includes('📤')||l.includes('🔥')||l.includes('⚙️')||l.includes('🔃')) cls = 'step';
        return '<div class="' + cls + '">' + l + '</div>';
      }).join('');
      la.scrollTop = la.scrollHeight;
    }
  } catch(e) {}
}, 1500);
</script>
</body>
</html>"""


# ── Routes ──

@app.route("/api/health")
def api_health():
    return jsonify({"status": "alive"})

@app.route("/api/shutdown")
def api_shutdown():
    log("⏻ Shutting down...")
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except: pass
    def _exit():
        time.sleep(0.2)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
    return jsonify({"status": "shutdown"})

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/load")
def load_model():
    mt = request.args.get("m", "")
    if mt not in MODEL_IDS:
        return "❌ Unknown model"
    if _model_ready.is_set() and _model_type == mt:
        return "✅ Already loaded"
    _model_ready.clear()
    global _loading_log
    _loading_log = []
    threading.Thread(target=load_model_thread, args=(mt,), daemon=True).start()
    return f"⏳ Loading {MODEL_NAMES.get(mt, mt)}..."

@app.route("/api_status")
def api_status():
    if _model_ready.is_set() and _model:
        return jsonify({"model": MODEL_NAMES.get(_model_type), "loading": False, "log": list(_loading_log)})
    return jsonify({"loading": True, "msg": _loading_msg, "log": list(_loading_log)})

@app.route("/api/upload_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file"})
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "No file selected"})

    ext = os.path.splitext(f.filename)[1] or ".webm"
    fname = f"ref_{uuid.uuid4().hex[:8]}{ext}"
    fpath = os.path.join(TEMP_DIR, fname)
    f.save(fpath)

    # Get duration
    try:
        import soundfile as sf
        data, sr = sf.read(fpath)
        dur = len(data) / sr
    except:
        dur = 0

    log(f"📥 Audio uploaded: {fname} ({dur:.1f}s)")
    return jsonify({"success": True, "filename": fname, "duration_sec": dur})

@app.route("/api/generate_clone", methods=["POST"])
def generate_clone():
    if not _model_ready.is_set() or not _model:
        return jsonify({"success": False, "error": "Model not loaded"})

    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"success": False, "error": "Text required"})

    lang = data.get("language", "Auto")
    ref_audio_file = data.get("ref_audio_file", "")
    ref_text = (data.get("ref_text") or "").strip()
    mode = data.get("mode", "x_vector")

    ref_path = os.path.join(TEMP_DIR, os.path.basename(ref_audio_file))
    if not os.path.exists(ref_path):
        return jsonify({"success": False, "error": "Reference audio not found — re-upload"})

    log(f"🧬 Voice clone: {len(text)} chars, mode={mode}, lang={lang}")

    import soundfile as sf
    import torch

    language = lang if lang != "Auto" else None

    # Read reference audio → move to device
    ref_wav, ref_sr = sf.read(ref_path)
    if ref_sr != 16000:
        import scipy.signal
        ref_wav = scipy.signal.resample(ref_wav, int(len(ref_wav) * 16000 / ref_sr))
        ref_sr = 16000
    ref_tensor = torch.from_numpy(ref_wav).float().to(_model.device)

    if mode == "icl" and ref_text:
        wavs, sr = _model.generate_voice_clone(
            text=text, language=language,
            ref_audio=ref_tensor, ref_sr=ref_sr,
            ref_text=ref_text,
            max_new_tokens=min(2048, max(64, len(text)*3)),
        )
    else:
        wavs, sr = _model.generate_voice_clone(
            text=text, language=language,
            ref_audio=ref_tensor, ref_sr=ref_sr,
            max_new_tokens=min(2048, max(64, len(text)*3)),
        )

    ts = int(time.time() * 1000)
    fname = f"qwen3tts_clone_{ts}.wav"
    sf.write(os.path.join(OUTPUT_DIR, fname), wavs[0] if isinstance(wavs, (list, tuple)) else wavs, sr)
    dur = len(wavs[0] if isinstance(wavs, (list, tuple)) else wavs) / sr

    log(f"✅ Clone generated: {dur:.1f}s → {fname}")
    return jsonify({"success": True, "filename": fname, "duration": dur})

@app.route("/generate", methods=["POST"])
def generate():
    if not _model_ready.is_set() or not _model:
        return render_template_string(HTML.replace('id="output" style="display:none"','id="output"').replace('<div id="output" class="card" style="display:none">','<div id="output" class="card"><p style="color:#f87171">⚠️ Load a model first!</p>'))
    text = request.form.get("text","").strip()
    if not text: return "❌ Text required", 400
    language = request.form.get("language","Auto")
    speaker = request.form.get("speaker","Ryan")
    instruct = request.form.get("instruct","").strip()
    import soundfile as sf
    lang = language if language != "Auto" else None
    log(f"🎯 Generating: {len(text)} chars, {speaker}, {language}")
    wavs, sr = _model.generate_custom_voice(text=text, language=lang, speaker=speaker, instruct=instruct if instruct else None, max_new_tokens=min(2048, max(64, len(text)*3)))
    ts = int(time.time()*1000)
    fname = f"qwen3tts_{ts}.wav"
    sf.write(os.path.join(OUTPUT_DIR, fname), wavs[0], sr)
    dur = len(wavs[0]) / sr
    log(f"✅ {dur:.1f}s → {fname}")
    result = HTML.replace('<div id="output" class="card" style="display:none">', '<div id="output" class="card">')
    result = result.replace('</body>', f'<p style="color:#4ade80;font-size:13px">✅ {dur:.1f}s — saved to Desktop</p><audio id="player" src="/output/{fname}" controls autoplay style="width:100%;margin-top:12px"></audio></body>')
    return render_template_string(result)

@app.route("/output/<path:fname>")
def serve_output(fname):
    fname = os.path.basename(fname)
    # Check in OUTPUT_DIR first, then TEMP_DIR
    for d in [OUTPUT_DIR, TEMP_DIR]:
        fp = os.path.join(d, fname)
        if os.path.exists(fp):
            return send_file(fp, mimetype="audio/wav")
    return "Not found", 404

@app.route("/quit")
def quit():
    return api_shutdown()

# ── Main ──
if __name__ == "__main__":
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(f"\n🎙️  Qwen3 TTS — Server on http://localhost:8765")
    app.run(host="127.0.0.1", port=8765, debug=False)
