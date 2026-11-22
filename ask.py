import sys
import ollama
from store import retrieve
from store import retrieve, embed_pending, init_db
import re
ANSWER_SYSTEM = """You answer questions about the user's email using ONLY the emails \
provided below. Rules:
- Use only information found in the provided emails. Never use outside knowledge, and \
never guess or invent details, dates, names, or amounts.
- If the provided emails do not contain the answer, say plainly that you don't see \
anything about that in the recent email. Do not speculate.
- When you state a fact, attribute it to the email it came from (by subject or sender).
- Be concise and direct. Answer in the language the question is asked in."""

def format_emails(rows):
    if not rows:
        return "(no emails found)"
    blocks = []
    for r in rows:
        blocks.append(
            f"--- Email ---\n"
            f"Inbox: {r['account']}\n"
            f"From: {r['sender']}\nSubject: {r['subject']}\nDate: {r['date']}\n"
            f"Category: {r['category']} (importance {r['importance']})\n"
            f"Content: {(r['body'] or '')[:1500]}"
        )
    return "\n\n".join(blocks)


def ask(question):
    rows = retrieve(question)
    context = format_emails(rows)
    resp = ollama.chat(
        model="qwen3:14b",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",
             "content": f"Emails:\n\n{context}\n\n---\n\nQuestion: {question}"},
        ],
        think=False,
        options={"temperature": 0},
    )
    return resp.message.content


if __name__ == "__main__":
    init_db()
    embed_pending()

    q = " ".join(sys.argv[1:]) or input("Ask about your email: ")
    print("\n" + ask(q) + "\n")