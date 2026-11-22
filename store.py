import sqlite3
from datetime import datetime, timezone
import re

import numpy as np
import ollama
ollama.pull("nomic-embed-text")

EMBED_MODEL = "nomic-embed-text"


DB_PATH = "agent.db"

_CATEGORY_HINTS = {
    "opportunity": ["opportunit", "oportunit", "summer school", "scholarship",
                    "bursa", "internship", "intern", "program"],
    "contest":     ["contest", "competition", "olympiad", "olimpiad", "concurs",
                    "result", "rezultat"],
    "event":       ["event", "eveniment", "webinar", "talk", "ceremony", "conferin"],
}


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                message_id  TEXT PRIMARY KEY,
                sender      TEXT,
                subject     TEXT,
                date        TEXT,
                body        TEXT,
                category    TEXT,
                importance  INTEGER,
                reason      TEXT,
                notified    INTEGER,
                triaged_at  TEXT
            )
        """),
        c.execute("""
                   CREATE TABLE IF NOT EXISTS embeddings (
                       message_id TEXT PRIMARY KEY,
                       vector     BLOB
                   )
               """)


def already_seen(message_id):
    with sqlite3.connect(DB_PATH) as c:
        return c.execute(
            "SELECT 1 FROM emails WHERE message_id = ?", (message_id,)
        ).fetchone() is not None


def save(message_id, sender, subject, date, body, triage, notified):
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            """INSERT OR IGNORE INTO emails
               (message_id, sender, subject, date, body, category, importance,
                reason, notified, triaged_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, sender, subject, date, body,
             triage.category.value, triage.importance, triage.reason,
             int(notified), datetime.now(timezone.utc).isoformat()),
        )

def _detect_categories(question):
    q = question.lower()
    return [cat for cat, hints in _CATEGORY_HINTS.items() if any(h in q for h in hints)]

def _detect_importance(question):
    q = question.lower()
    return any(w in q for w in ["important", "urgent", "notify", "attention", "miss", "matter"])

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def retrieve(question, limit=8):
    cats = _detect_categories(question)
    want_important = _detect_importance(question)

    where, params = [], []
    if cats:
        where.append("category IN (%s)" % ",".join("?" * len(cats)))
        params += cats
    if want_important:
        where.append("importance >= 2")
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(f"SELECT * FROM emails{clause} ORDER BY triaged_at DESC",
                         params).fetchall()
        if not rows:                       # filter too strict → widen to everything
            rows = c.execute("SELECT * FROM emails ORDER BY triaged_at DESC").fetchall()
        vecs = {r["message_id"]: np.frombuffer(r["vector"], dtype=np.float32)
                for r in c.execute("SELECT message_id, vector FROM embeddings").fetchall()}

    qvec = _embed(question, "search_query")
    scored = [(_cosine(qvec, vecs[r["message_id"]]) if r["message_id"] in vecs else -1.0, r)
              for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]

def _embed(text, kind="search_document"):
    text = (text or "").strip() or "(empty)"
    # nomic uses task prefixes: documents and queries get embedded differently.
    resp = ollama.embed(model=EMBED_MODEL, input=f"{kind}: {text[:4000]}")
    return np.array(resp.embeddings[0], dtype=np.float32)


def embed_pending():
    # Embed any email that doesn't have a vector yet (new mail + old backlog).
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""
            SELECT e.message_id, e.subject, e.body FROM emails e
            LEFT JOIN embeddings em ON e.message_id = em.message_id
            WHERE em.message_id IS NULL
        """).fetchall()
        for r in rows:
            vec = _embed(f"{r['subject']}\n\n{r['body'] or ''}")
            c.execute("INSERT OR REPLACE INTO embeddings (message_id, vector) VALUES (?, ?)",
                      (r["message_id"], vec.tobytes()))
    return len(rows)