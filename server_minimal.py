#!/usr/bin/env python3
"""
Qwen3 TTS — Minimal Server
===========================
Dead simple. Open http://localhost:8765 in your browser.
Pick a model, click LOAD, then click GENERATE. That's it.

Usage:
    cd ~/qwen3-tts-ui && source venv/bin/activate && python3 server_minimal.py
"""

import os, sys, json, time, threading, logging
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Flask setup ──
from flask import Flask, request, render_template_string, send_file, jsonify

app = Flask(__name__)

OUTPUT_DIR = os.path.expanduser("~/Desktop")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_IDS = {
    "cv_1b7": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "cv_0b6": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
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


def load_model_thread(mt):
    global _model, _model_type
    import torch
    from qwen_tts import Qwen3TTSModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    attn = "sdpa" if device == "mps" else "eager"

    _model = Qwen3TTSModel.from_pretrained(MODEL_IDS[mt], dtype=dtype, attn_implementation=attn)
    if device != "cpu":
        _model.model = _model.model.to(device)
    _model.device = torch.device(device)
    _model_type = mt
    _model_ready.set()


# ── Routes ──

MODEL_BUTTONS = """
<button onclick="fetch('/load?m=cv_1b7').then(r=>r.text()).then(t=>document.getElementById('status').textContent=t)" style="background:#1a6b3c;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">🎤 CustomVoice 1.7B HQ</button>
<button onclick="fetch('/load?m=cv_0b6').then(r=>r.text()).then(t=>document.getElementById('status').textContent=t)" style="background:#b8860b;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">⚡ CustomVoice 0.6B Fast</button>
<button onclick="fetch('/load?m=base').then(r=>r.text()).then(t=>document.getElementById('status').textContent=t)" style="background:#4a4a8a;border:none;padding:12px 20px;border-radius:10px;color:white;font-size:15px;cursor:pointer;margin:4px">🧬 Voice Clone 1.7B</button>
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
#status{padding:12px;background:#0f3460;border-radius:10px;margin:12px 0;font-size:13px;min-height:20px}
audio{width:100%;margin-top:12px}
.loading{color:#f0a500}
.ready{color:#4ade80}
.error{color:#f87171}
</style>
</head>
<body>
<h1>🎙️ Qwen3 TTS</h1>
<div class="sub">Select a model → Load it → Generate speech. Close Terminal to quit.</div>

<div class="card">
""" + MODEL_BUTTONS + """
<div id="status">💡 Click a model above to load it</div>
</div>

<form class="card" action="/generate" method="post">
<label>Text</label>
<textarea name="text" required>Ciao! Questo è un test di sintesi vocale.</textarea>
<label>Language</label>
<select name="language">
""" + "".join(f'<option{" selected" if l=="Italian" else ""}>{l}</option>' for l in LANGUAGES) + """
</select>

<div id="cv_fields">
<label>Speaker</label>
<select name="speaker">
""" + "".join(f'<option value="{s}">{n}</option>' for s,n in SPEAKERS.items()) + """
</select>
<label>Style (optional)</label>
<input name="instruct" placeholder="e.g. 'Speak happily'">
</div>

<button class="btn-gen" type="submit">⚡ Generate Speech</button>
</form>

<div id="output" class="card" style="display:none">
<audio id="player" controls></audio>
</div>

<script>
// Auto-poll status
setInterval(async () => {
  let r = await fetch('/api_status');
  let d = await r.json();
  let s = document.getElementById('status');
  if (d.loading) {
    s.className = 'loading';
    s.textContent = '⏳ ' + d.msg;
  } else if (d.model) {
    s.className = 'ready';
    s.textContent = '✅ ' + d.model + ' ready';
    document.querySelector('.btn-gen').disabled = false;
  } else {
    s.className = '';
    s.textContent = '💡 Click a model above to load it';
    document.querySelector('.btn-gen').disabled = true;
  }
}, 2000);

// Show audio when page has ?file= parameter
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
    threading.Thread(target=load_model_thread, args=(mt,), daemon=True).start()
    return f"⏳ Loading {mt}..."


@app.route("/api_status")
def api_status():
    if _model_ready.is_set() and _model:
        return jsonify({"model": MODEL_IDS.get(_model_type, "Model"), "loading": False})
    return jsonify({"loading": True, "msg": "Loading..."})


@app.route("/generate", methods=["POST"])
def generate():
    if not _model_ready.is_set():
        return render_template_string(HTML.replace('id="output" style="display:none"', 'id="output"').replace('<!-- no audio -->', '<p style="color:#f87171">⚠️ Load a model first!</p>'))

    text = request.form.get("text", "").strip()
    if not text:
        return "❌ Text required", 400

    language = request.form.get("language", "Auto")
    speaker = request.form.get("speaker", "Ryan")
    instruct = request.form.get("instruct", "").strip()

    import soundfile as sf
    lang = language if language != "Auto" else None

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
    return render_template_string(HTML.replace('id="output" style="display:none"',
        f'id="output" style="display:block"').replace('<!-- no audio -->',
        f'<p style="color:#4ade80;font-size:13px">✅ {dur:.1f}s generated</p>').replace(
        '</body>',
        f'<audio id="player" src="/output/{fname}" controls autoplay style="width:100%;margin-top:12px"></audio>'
        f'<p style="font-size:12px;color:#888;margin-top:4px">Saved to Desktop</p>'
        '</body>'))


@app.route("/output/<path:fname>")
def serve_output(fname):
    fpath = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    return send_file(fpath, mimetype="audio/wav")


# ── Main ──
if __name__ == "__main__":
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(f"\n🎙️  Qwen3 TTS")
    print(f"   Open http://localhost:8765 in your browser")
    print(f"   Close Terminal to stop.\n")
    app.run(host="127.0.0.1", port=8765, debug=False)
