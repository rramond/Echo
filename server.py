"""Backend bridge between the Echo agent pipeline and emres-brain.html.

Run from anywhere:  .venv/bin/python server.py   (serves http://localhost:8765)

Endpoints (all JSON unless noted):
  GET  /                        -> emres-brain.html
  GET  /api/agents/A1           -> {running, task, log}
  POST /api/agents/A1/start     -> start the mail-watch loop
  POST /api/agents/A1/stop      -> stop it
  PUT  /api/agents/A1/task      -> {task} persist new task text
  POST /api/agents/A1/ask       -> {question} -> {answer}
  POST /api/agents/A1/voice     -> raw float32 PCM body + X-Sample-Rate header
                                   -> {question, answer, audio_b64 (wav)}
"""

import base64
import io
import json
import os
import re
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # agent.db and the piper .onnx models are addressed relatively

HTML_PATH = HERE.parent / "emres-brain.html"
TASK_PATH = HERE / "task.json"
PORT = 8765
FETCH_INTERVAL = 300  # seconds between inbox checks while running

DEFAULT_TASK = ("Watch the configured inboxes over IMAP, classify incoming mail, "
                "and answer voice queries about recent messages.")

# ---------------------------------------------------------------- agent state
_log = []          # newest first: {time(ms), event, status: ok|fail|run, dur(s)}
_log_lock = threading.Lock()
_runner = None     # background thread
_stop = threading.Event()
_voice_lock = threading.Lock()
_whisper = None
_piper = None


def add_log(event, status="ok", dur=0):
    with _log_lock:
        _log.insert(0, {"time": int(time.time() * 1000),
                        "event": event, "status": status, "dur": dur})
        del _log[200:]


def get_task():
    if TASK_PATH.exists():
        return json.loads(TASK_PATH.read_text())["task"]
    return DEFAULT_TASK


def set_task(text):
    TASK_PATH.write_text(json.dumps({"task": text}, indent=2))


# ---------------------------------------------------------------- mail loop
def run_cycle():
    # Imports are deferred: store.py pulls the embedding model via ollama on
    # import, and we want the UI to come up even when ollama isn't running yet.
    from fetch import fetch_new, decode_str, get_body
    from classify import triage
    from store import init_db, save, embed_pending

    init_db()
    t0 = time.time()
    new_items, errors = fetch_new()
    for err in errors:
        add_log(f"Fetch failed — {err}", "fail", 0)
    notified = 0
    for account, message_id, msg in new_items:
        subject = decode_str(msg.get("Subject"))
        r = triage(subject, get_body(msg)[:2000])
        notify = r.importance >= 2
        notified += notify
        save(message_id, decode_str(msg.get("From")), subject,
             msg.get("Date") or "", get_body(msg)[:2000], r, notify, account)
        add_log(f"{'🔔 ' if notify else ''}[{account}] {r.category.value} imp={r.importance} — {subject[:60]}",
                "ok", 0)
    embed_pending()
    dur = int(time.time() - t0)
    if new_items:
        add_log(f"Cycle done: {len(new_items)} new, {notified} notify-worthy", "ok", dur)
    else:
        add_log("Cycle done: no new mail", "ok", dur)


def runner():
    while not _stop.is_set():
        try:
            run_cycle()
        except Exception as e:
            add_log(f"Cycle failed: {e}", "fail", 0)
        _stop.wait(FETCH_INTERVAL)


def start_agent():
    global _runner
    if _runner and _runner.is_alive():
        return False
    _stop.clear()
    _runner = threading.Thread(target=runner, daemon=True)
    _runner.start()
    add_log("Agent started — watching inbox", "run", 0)
    return True


def stop_agent():
    _stop.set()
    add_log("Agent stopped", "ok", 0)


def is_running():
    return bool(_runner and _runner.is_alive() and not _stop.is_set())


# ---------------------------------------------------------------- voice
def load_voice_models():
    global _whisper, _piper
    if _whisper is None:
        from faster_whisper import WhisperModel
        add_log("Loading whisper model (first voice query)", "run", 0)
        _whisper = WhisperModel("medium", device="cpu", compute_type="int8")
    if _piper is None:
        from piper import PiperVoice
        _piper = PiperVoice.load("en_US-amy-medium.onnx")


def for_speech(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)          # [label](url) -> label
    text = re.sub(r"(?:https?://|www\.)\S+", "", text, flags=re.I)  # bare URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)                       # email addresses
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.M)           # "1." list markers
    # symbols TTS would read aloud; keep .,!?;:' for natural prosody
    text = re.sub(r"[*#_`>~|\[\](){}<>\"“”‘’«»/\\=+^&%$@—–-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def handle_voice(pcm_bytes, sample_rate):
    load_voice_models()
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    if sample_rate != 16000 and len(audio):
        n = int(len(audio) * 16000 / sample_rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    with _voice_lock:
        segments, info = _whisper.transcribe(audio, beam_size=5)
        question = " ".join(s.text for s in segments).strip()
    if not question:
        return {"error": "no speech detected"}

    t0 = time.time()
    from ask import ask
    answer = ask(question)
    buf = io.BytesIO()
    with _voice_lock:
        with wave.open(buf, "wb") as wav:
            _piper.synthesize_wav(for_speech(answer), wav)
    add_log(f"Voice query answered — “{question[:50]}”", "ok", int(time.time() - t0))
    return {"question": question, "answer": answer,
            "audio_b64": base64.b64encode(buf.getvalue()).decode()}


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/agents/A1":
            with _log_lock:
                log = list(_log)
            self._json({"running": is_running(), "task": get_task(), "log": log})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/agents/A1/start":
                start_agent()
                self._json({"running": True})
            elif self.path == "/api/agents/A1/stop":
                stop_agent()
                self._json({"running": False})
            elif self.path == "/api/agents/A1/ask":
                q = json.loads(self._body())["question"]
                from ask import ask
                self._json({"question": q, "answer": ask(q)})
            elif self.path == "/api/agents/A1/voice":
                rate = int(self.headers.get("X-Sample-Rate", 16000))
                self._json(handle_voice(self._body(), rate))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            add_log(f"Request failed: {e}", "fail", 0)
            self._json({"error": str(e)}, 500)

    def do_PUT(self):
        if self.path == "/api/agents/A1/task":
            set_task(json.loads(self._body())["task"])
            add_log("Task definition updated", "ok", 0)
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):  # quiet the default per-request stderr noise
        pass


if __name__ == "__main__":
    add_log("Server online", "ok", 0)
    print(f"EMRE'S BRAIN backend → http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
