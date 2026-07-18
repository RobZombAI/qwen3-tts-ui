#!/usr/bin/env python3
"""
Qwen3 TTS — Server with live progress in browser
=================================================
All process logs shown inside the web UI.
No need to check Terminal — everything visible in the app.

Usage:
    cd ~/qwen3-tts-ui && source venv/bin/activate && python3 server_minimal.py
"""

import os, sys, json, time, threading, logging
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, render_template_string, send_file, jsonify

app = Flask(__name__)

OUTPUT_DIR = os.path.expanduser("~/Desktop")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ── Model manager with live progress ──
_model = None
_model_type = None
_model_ready = threading.Event()
_loading_msg = "💡 Click a model above to load it"
_loading_log = []  # Full log of messages

def log(msg):
    """Add message to live log (visible in browser)."""
    global _loading_msg, _loading_log
    _loading_msg = msg
    _loading_log.append(msg)
    # Print to Terminal too so user has both views
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
            device = "mps"
            model_dtype = torch.bfloat16
            attn = "sdpa"
            log(f"⚙️  Device: MPS (Apple Silicon GPU)")
        else:
            device = "cpu"
            model_dtype = torch.float32
            attn = "eager"
            log(f"⚙️  Device: CPU (no GPU detected)")

        log(f"📦 Downloading/loading model weights…")
        _model = Qwen3TTSModel.from_pretrained(MODEL_IDS[mt], dtype=model_dtype, attn_implementation=attn)
        log(f"✅ Model loaded into memory")

        if device != "cpu":
            log(f"📤 Moving model to {device.upper()}…")
            _model.model = _model.model.to(device)
            log(f"✅ Model on {device.upper()}")

        _model.device = torch.device(device)

        log(f"🔥 Warming up (compiling kernels)…")
        try:
            _model.generate_custom_voice(text="Hi.", language="English", speaker="Ryan", max_new_tokens=2)
            log(f"✅ Warm-up complete")
        except Exception as e:
            log(f"⚠️ Warm-up skipped: {e}")

        _model_type = mt
        _model_ready.set()
        log(f"✅ {name} ready! You can now generate speech.")

    except Exception as e:
        log(f"❌ Failed to load model: {e}")


# ── Routes ──

MODEL_BUTTONS = """
<button onclick="loadModel('cv_1b7')" style="background:#1a6b3c;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4rpx">🎤 CustomVoice 1.7B HQ</button>
<button onclick="loadModel('cv_0b6')" style="background:#b8860b;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">⚡ CustomVoice 0.6B Fast</button>
<button onclick="loadModel('base')" style="background:#4a4a8a;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">🧬 Voice Clone 1.7B</button>
"""

HTML = """<!DOCTYPE html>
<html>
<head><title>Qwen3 TTS</title>
<meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:30px;max-width:700px;margin:auto}
h1{font-size:26px;margin-bottom:4px;color:#f0f0ff}
.sub{color:#8888aa;font-size:13px;margin-bottom:20px}
.card{background:#16213e;border-radius:16px;padding:24px;margin-bottom:16px}
label{display:block;font-size:13px;font-weight:600;color:#aab;margin-bottom:4px;margin-top:12px}
label:first-child{margin-top:0}
select,input,textarea{width:100%;background:#0f3460;border:1px solid #1a4a7a;color:#e0e0e0;border-radius:8px;padding:10px;font-size:14px;outline:none}
textarea{height:100px;resize:vertical}
.btn-gen{width:100%;padding:14px;border:none;border-radius:10px;font-size:16px;font-weight:700;cursor:pointer;background:linear-gradient(135deg,#1a6b3c,#228b4f);color:#fff;margin-top:16px}
.btn-gen:disabled{opacity:.4;cursor:not-allowed}
audio{width:100%;margin-top:12px}
.log-area{background:#0a1a2e;border-radius:8px;padding:12px;margin-top:12px;font-family:monospace;font-size:11px;max-height:200px;overflow-y:auto;line-height:1.6}
.log-area .ok{color:#4ade80}
.log-area .info{color:#88aacc}
.log-area .warn{color:#f0a500}
.log-area .err{color:#f87171}
.log-area .step{color:#8888aa}
</style>
</head>
<body>
<h1>🎙️ Qwen3 TTS</h1>
<div class="sub">Select a model → Load it → Generate speech. Close Terminal to quit.</div>

<div class="card">
""" + MODEL_BUTTONS + """
<div id="status" style="padding:12px;background:#0f3460;border-radius:10px;margin:12px 0;font-size:13px;min-height:20px">💡 Click a model above to load it</div>
<div id="logArea" class="log-area" style="display:none"></div>
</div>

<form class="card" action="/generate" method="post">
<label>Text</label>
<textarea name="text" required>Ciao! Questo è un test di sintesi vocale.</textarea>
<label>Language</label>
<select name="language">
""" + "".join(f'<option{" selected" if l=="Italian" else ""}>{l}</option>' for l in LANGUAGES) + """
</select>

<label>Speaker</label>
<select name="speaker">
""" + "".join(f'<option value="{s}">{n}</option>' for s,n in SPEAKERS.items()) + """
</select>
<label>Style (optional)</label>
<input name="instruct" placeholder="e.g. 'Speak happily'">

<button class="btn-gen" type="submit" disabled>⚡ Generate Speech</button>
</form>

<div id="output" class="card" style="display:none">
<audio id="player" controls></audio>
</div>

<button onclick="fetch('/quit').then(()=>window.close())" style="width:100%;padding:10px;border:none;border-radius:8px;background:#4a1a1a;color:#f87171;font-size:13px;cursor:pointer;margin-top:8px">⏻ Quit App</button>

<script>
function loadModel(m) {
  let status = document.getElementById('status');
  let logArea = document.getElementById('logArea');
  logArea.style.display = 'block';
  logArea.innerHTML = '<div class="step">⏳ Starting load of ' + m + '...</div>';
  status.className = '';
  status.textContent = '⏳ Loading ' + m + '...';
  document.querySelector('.btn-gen').disabled = true;

  fetch('/load?m=' + m).then(r => r.text()).then(t => {
    status.textContent = t;
  });
}

// Poll status every 1.5s — show progress and log
setInterval(async () => {
  try {
    let r = await fetch('/api_status');
    let d = await r.json();
    let s = document.getElementById('status');
    let logArea = document.getElementById('logArea');

    // Update status
    if (d.loading) {
      s.textContent = '⏳ ' + d.msg;
    } else if (d.model) {
      s.style.color = '#4ade80';
      s.textContent = '✅ ' + d.model + ' ready';
      document.querySelector('.btn-gen').disabled = false;
    }

    // Update log if we have new entries
    if (d.log && d.log.length > 0) {
      logArea.style.display = 'block';
      logArea.innerHTML = d.log.map(function(l) {
        let cls = 'info';
        if (l.includes('✅')) cls = 'ok';
        else if (l.includes('⚠️')) cls = 'warn';
        else if (l.includes('❌')) cls = 'err';
        else if (l.includes('📦') || l.includes('📤') || l.includes('🔥') || l.includes('⚙️') || l.includes('🔃')) cls = 'step';
        return '<div class="' + cls + '">' + l + '</div>';
      }).join('');
      logArea.scrollTop = logArea.scrollHeight;
    }
  } catch(e) {
    document.getElementById('status').textContent = '⏳ Connecting to server...';
  }
}, 1500);

// Show audio when form submits
const url = new URL(window.location);
const file = url.searchParams.get('file');
if (file) {
  document.getElementById('output').style.display = 'block';
  document.getElementById('player').src = '/output/' + file;
}
</script>
</body>
</html>"""


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
        return jsonify({
            "model": MODEL_NAMES.get(_model_type, "Model"),
            "loading": False,
            "log": list(_loading_log),
        })
    return jsonify({
        "loading": True,
        "msg": _loading_msg,
        "log": list(_loading_log),
    })


@app.route("/generate", methods=["POST"])
def generate():
    if not _model_ready.is_set():
        return render_template_string(HTML.replace(
            'style="display:none" id="output"', 'id="output" style="display:block"'
        ).replace(
            '<div id="output" class="card" style="display:none">',
            '<div id="output" class="card"><p style="color:#f87171">⚠️ Load a model first!</p>'
        ))

    text = request.form.get("text", "").strip()
    if not text:
        return "❌ Text required", 400

    language = request.form.get("language", "Auto")
    speaker = request.form.get("speaker", "Ryan")
    instruct = request.form.get("instruct", "").strip()

    import soundfile as sf
    import torch
    lang = language if language != "Auto" else None

    # Log generation start
    log(f"🎯 Generating: {len(text)} chars, speaker={speaker}, lang={language}")

    wavs, sr = _model.generate_custom_voice(
        text=text, language=lang,
        speaker=speaker,
        instruct=instruct if instruct else None,
        max_new_tokens=min(2048, max(64, len(text)*3)),
    )

    ts = int(time.time()*1000)
    fname = f"qwen3tts_{ts}.wav"
    sf.write(os.path.join(OUTPUT_DIR, fname), wavs[0], sr)

    dur = len(wavs[0]) / sr
    log(f"✅ Generated {dur:.1f}s → {fname}")

    # Return page with audio
    result = HTML.replace(
        '<div id="output" class="card" style="display:none">',
        '<div id="output" class="card">'
    ).replace(
        '</body>',
        f'<p style="color:#4ade80;font-size:13px">✅ {dur:.1f}s — saved to Desktop</p>'
        f'<audio id="player" src="/output/{fname}" controls autoplay style="width:100%;margin-top:12px"></audio>'
        '</body>'
    )
    return render_template_string(result)


@app.route("/output/<path:fname>")
def serve_output(fname):
    fpath = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    return send_file(fpath, mimetype="audio/wav")


@app.route("/quit")
def quit_app():
    """Clean shutdown: free memory, exit process. Called by 'Quit App' button."""
    log("⏻ Shutting down...")

    # Free MPS memory
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass

    # Stop Flask and exit after response is sent
    def _shutdown():
        import time
        time.sleep(0.2)
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()

    return jsonify({"status": "shutdown"})


# ── Main ──
if __name__ == "__main__":
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(f"\n🎙️  Qwen3 TTS")
    print(f"   Open http://localhost:8765 in your browser")
    print(f"   All process logs visible inside the app.")
    print(f"   Close Terminal to stop.\n")
    app.run(host="127.0.0.1", port=8765, debug=False)
