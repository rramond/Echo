import imaplib
import email
import re
import os
from email.header import decode_header, make_header
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import hashlib
from store import init_db, already_seen, save

from classify import triage   # reuse your Step 3 engine, unchanged

from store import embed_pending

load_dotenv()


def load_accounts():
    """Accounts come from .env as numbered groups:

        ACCOUNT_1_NAME=personal
        ACCOUNT_1_ADDRESS=me@gmail.com
        ACCOUNT_1_PASSWORD=app-password
        ACCOUNT_1_HOST=imap.gmail.com      (optional, this is the default)

    and so on for ACCOUNT_2_..., ACCOUNT_3_... The older single-account
    GMAIL_ADDRESS / GMAIL_APP_PASSWORD pair still works as a "personal" account.
    """
    accounts = []
    n = 1
    while os.environ.get(f"ACCOUNT_{n}_ADDRESS"):
        accounts.append({
            "name": os.environ.get(f"ACCOUNT_{n}_NAME", f"account{n}"),
            "address": os.environ[f"ACCOUNT_{n}_ADDRESS"],
            "password": os.environ[f"ACCOUNT_{n}_PASSWORD"],
            "host": os.environ.get(f"ACCOUNT_{n}_HOST", "imap.gmail.com"),
        })
        n += 1
    if not accounts and os.environ.get("GMAIL_ADDRESS"):
        accounts.append({"name": "personal",
                         "address": os.environ["GMAIL_ADDRESS"],
                         "password": os.environ["GMAIL_APP_PASSWORD"],
                         "host": "imap.gmail.com"})
    if not accounts:
        raise RuntimeError("No mail accounts configured in .env")
    return accounts


ACCOUNTS = load_accounts()

def fallback_id(msg):
    raw = (decode_str(msg.get("From")) + decode_str(msg.get("Subject"))
           + (msg.get("Date") or ""))
    return "fallback:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def decode_str(raw):
    # Subjects/senders may be MIME-encoded (=?UTF-8?...) — important for Romanian text.
    if raw is None:
        return ""
    return str(make_header(decode_header(raw)))

def looks_like_html(text):
    return re.search(r"<[a-z!/][^>]*>", text, re.I) is not None

def strip_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(
            r"display\s*:\s*none|max-height\s*:\s*0|visibility\s*:\s*hidden", re.I)):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()

def clean(text):
    # Senders stuff HTML — incl. hidden decoys — into the text/plain part too.
    return strip_html(text) if looks_like_html(text) else re.sub(r"\s+", " ", text).strip()

def get_body(msg):
    # Email bodies are MIME trees. Prefer text/plain; fall back to stripped HTML.
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and \
               "attachment" not in str(part.get("Content-Disposition") or ""):
                payload = part.get_payload(decode=True)
                if payload:
                    return clean(payload.decode(part.get_content_charset() or "utf-8",
                                          errors="replace"))
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return strip_html(payload.decode(
                        part.get_content_charset() or "utf-8", errors="replace"))
        return ""
    payload = msg.get_payload(decode=True)
    if not payload:
        return ""
    text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    return clean(text)


def fetch_account(account, limit=50):
    imap = imaplib.IMAP4_SSL(account["host"])
    imap.login(account["address"], account["password"])
    imap.select("INBOX")
    _, data = imap.search(None, "UNSEEN")        # server-side: only unread mail
    ids = data[0].split()
    new_items = []
    for eid in reversed(ids[-limit:]):           # most recent unread, newest first
        _, msg_data = imap.fetch(eid, "(BODY.PEEK[])")   # PEEK: leaves it unread
        msg = email.message_from_bytes(msg_data[0][1])
        message_id = msg.get("Message-ID") or fallback_id(msg)
        if already_seen(message_id):             # ledger says judged already → skip
            continue
        new_items.append((account["name"], message_id, msg))
    imap.logout()
    return new_items


def fetch_new(limit=50):
    """Fetch unread mail from every account. One account failing (wrong
    password, server down) doesn't stop the others; its error is returned."""
    new_items, errors = [], []
    for account in ACCOUNTS:
        try:
            new_items += fetch_account(account, limit)
        except Exception as e:
            errors.append(f"{account['name']}: {e}")
    return new_items, errors


if __name__ == "__main__":
    init_db()
    new_items, errors = fetch_new()
    for err in errors:
        print(f"⚠️  {err}")
    if not new_items:
        print("No new mail since last run. 📭")
    for account, message_id, msg in new_items:
        sender = decode_str(msg.get("From"))
        subject = decode_str(msg.get("Subject"))
        date = msg.get("Date") or ""
        body = get_body(msg)[:2000]
        r = triage(subject, body)
        notified = r.importance >= 2
        save(message_id, sender, subject, date, body, r, notified, account)
        flag = "🔔 NOTIFY" if notified else "  quiet"
        print(f"{flag} | {r.category.value:12} imp={r.importance} | {subject[:45]}")
        print(f"         [{account}] from: {sender[:50]}")
        print(f"         └ {r.reason}\n")
    embed_pending()