# Echo

> A local, private email assistant that reads your inbox, tells you what matters, and answers questions about it, by text or by voice.

Echo watches one or more inboxes over IMAP (say, personal and work). A local LLM sorts each new message into a category and gives it an importance score, and everything is stored in SQLite with embeddings so you can ask questions like *"did I get any contest results this week?"* and get an answer drawn only from your own mail.

Everything runs on your own machine: [Ollama](https://ollama.com) for the LLM and embeddings, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text, and [Piper](https://github.com/rhasspy/piper) for text-to-speech.

---

## How it works

```
  inboxes (IMAP, UNSEEN)
        │
        ▼
   fetch.py ──► classify.py ──► store.py ──► agent.db
   clean body    qwen3:14b       SQLite +     (emails +
   (HTML-safe)   category +      nomic-embed   embeddings)
                 importance                        │
                                                   ▼
   🎤 whisper ──► ask.py ──► retrieve (filter + cosine) ──► qwen3:14b ──► piper 🔊
```

| Category      | What goes there                                              |
|---------------|--------------------------------------------------------------|
| `contest`     | Competitions: registrations, schedules, **results**          |
| `opportunity` | Summer schools, scholarships, programs you can apply to      |
| `event`       | Talks, ceremonies, webinars tied to a date                   |
| `ignore`      | Newsletters, spam, promos and discounts, 2FA codes, receipts |

Importance runs from **0** (beneath notice) to **3** (time-sensitive). Anything scored **≥ 2** is flagged 🔔.

## Project layout

| File            | Role |
|-----------------|------|
| `server.py`     | HTTP backend on `localhost:8765`. Serves the dashboard and runs the inbox-watch loop (every 5 min) |
| `fetch.py`      | IMAP fetch of unread mail (peek only, so nothing gets marked read), MIME decoding, HTML stripping |
| `classify.py`   | LLM triage into a `Triage` pydantic model (category, importance, reason) |
| `store.py`      | SQLite ledger, embeddings, and hybrid retrieval (keyword filter + vector rank) |
| `ask.py`        | Answers a question using only the retrieved emails |
| `voice-ask.py`  | Terminal voice loop: speak a question, hear the answer |
| `*test.py`, `player.py` | Small sanity checks for Ollama, Whisper, and Piper |

## Setup

**Requirements:** Python 3.12, [Ollama](https://ollama.com), a microphone and speakers if you want voice.

```bash
# 1. Models
ollama pull qwen3:14b
ollama pull nomic-embed-text

# 2. Python env
python -m venv .venv
.venv/bin/pip install ollama pydantic numpy python-dotenv beautifulsoup4 \
                      faster-whisper piper-tts sounddevice

# 3. Piper voice (en_US-amy-medium.onnx + .onnx.json in this folder)
#    https://huggingface.co/rhasspy/piper-voices
```

Create a `.env` file with one numbered block per inbox. For Gmail, the password is an [app password](https://myaccount.google.com/apppasswords), not your normal one. `HOST` is optional and defaults to `imap.gmail.com`.

```env
ACCOUNT_1_NAME=personal
ACCOUNT_1_ADDRESS=you@gmail.com
ACCOUNT_1_PASSWORD=xxxx xxxx xxxx xxxx

ACCOUNT_2_NAME=work
ACCOUNT_2_ADDRESS=you@company.com
ACCOUNT_2_PASSWORD=...
ACCOUNT_2_HOST=imap.company.com
```

Each email is tagged with its account name, so you can ask *"anything urgent in my work email?"* and Echo searches only that inbox. If one account fails to connect, the others still get checked. The old single-account `GMAIL_ADDRESS` / `GMAIL_APP_PASSWORD` pair still works and is treated as `personal`.

## Usage

```bash
# Full app: backend + dashboard (../emres-brain.html)
.venv/bin/python server.py            # → http://localhost:8765

# One-off triage of unread mail
.venv/bin/python fetch.py

# Ask from the terminal
.venv/bin/python ask.py "anything important about summer schools?"

# Ask out loud
.venv/bin/python voice-ask.py
```

### API

| Method | Endpoint                 | Body / Result |
|--------|--------------------------|---------------|
| `GET`  | `/api/agents/A1`         | `{running, task, log}` |
| `POST` | `/api/agents/A1/start`   | start the inbox-watch loop |
| `POST` | `/api/agents/A1/stop`    | stop it |
| `PUT`  | `/api/agents/A1/task`    | `{task}` |
| `POST` | `/api/agents/A1/ask`     | `{question}` → `{answer}` |
| `POST` | `/api/agents/A1/voice`   | raw float32 PCM + `X-Sample-Rate` → `{question, answer, audio_b64}` |

## Notes

- **Your mail stays on your machine.** Nothing goes to a cloud API; `agent.db` and `.env` are gitignored.
- The triage prompt in `classify.py` is written for one particular user (a student who does olympiads and is applying to universities abroad). Change the *ABOUT THE USER* section to fit your own priorities.
- Answers come in the language you ask in, and Romanian subjects and bodies decode correctly.
