from enum import Enum
from pydantic import BaseModel, Field
import ollama


class Category(str, Enum):
    ignore = "ignore"
    contest = "contest"
    opportunity = "opportunity"
    event = "event"


class Triage(BaseModel):
    category: Category
    importance: int = Field(ge=0, le=3)
    reason: str


SYSTEM = """You triage email for a Romanian high-school student who competes in \
various contests and is applying to universities abroad. Put each email in exactly \
one category and rate how much they should care.

Categories:
- ignore: newsletters, marketing blasts, spam, login/2FA codes, automated receipts — \
purely transactional or promotional, no personal relevance.
- contest: anything about a competition they're in — registration, schedules, and \
especially RESULTS or "what happens next" details.
- opportunity: summer schools, scholarships, programs, or invitations from universities \
or contest organizers — a real chance to DO something, not an ad pretending to be one.
- event: talks, ceremonies, meetups, webinars, leisure events tied to a specific date.

importance: 0 = beneath notice, 1 = minor FYI, 2 = worth attention, 3 = high / \
time-sensitive / must act.

ABOUT THE USER (use this to judge RELEVANCE and importance, not category):
- Stage: pre-final year of high school
- Academic focus: mathematics, computer science, physics, engineering
- What genuinely matters: competitive olympiads, contests, summer schools/programs, scholarships in my field
- Generally NOT interested in: unsolicited recruitment from universities I'm not pursuing, regardless of prestige

Judgment rules:
- A real opportunity (a summer school they could apply to) is 'opportunity', importance \
2-3. A university marketing email selling its programs is 'ignore', importance \
0-1. Tell them apart by whether there's a concrete thing to apply to or attend.
- Discounts, sales and other commercial offers are 'ignore', importance 0, unless they \
are tied to a contest or program in the user's field.
- BE INCLUSIVE. When unsure whether something matters, score it HIGHER, not lower. Only \
use 'ignore' when you're confident it's noise. Missing one real opportunity is far worse \
than surfacing one extra email.
- reason: one short sentence naming the single detail that matters — a deadline, a \
result, a date.
- If the email body has little or no real text (image-only marketing, for example), \
say so plainly in the reason — e.g. "image-only promo, no text" — and do not invent \
content that isn't there.
- Distinguish a genuine opportunity from recruitment marketing:
  * Opportunity (importance 2-3): a SPECIFIC program, scholarship, competition, or \
invitation the user could concretely apply to or attend — especially if it aligns with \
their academic focus or the programs they're pursuing above, or if they were selected, \
nominated, or invited on merit.
  * Recruitment marketing (importance 0-1): an institution mass-mailing "apply to us" to \
prospective students it has no prior relationship with, with merge-field personalization \
and no specific selective program — EVEN IF it contains a real deadline. A deadline alone \
does not create relevance.
  * If the sender or subject is outside the user's stated interests, lean lower even when \
it sounds prestigious. If it's squarely inside them, lean higher.

Respond only with the JSON object."""


def triage(subject: str, body: str) -> Triage:
    resp = ollama.chat(
        model="qwen3:14b",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Subject: {subject}\n\n{body}"},
        ],
        format=Triage.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    return Triage.model_validate_json(resp.message.content)

##
if __name__ == "__main__":
    tests = [
        ("Your verification code is 481920",
         "Use this code to sign in. It expires in 10 minutes."),
        ("Applications open: Summer School in Mathematics 2026",
         "We invite talented students to apply for our 2-week residential summer "
         "school in Cluj. Deadline to apply: June 30. Travel grants available."),
        ("Discover everything our university has to offer",
         "Download our undergraduate prospectus and explore 200+ programs. "
         "Book a campus tour today!"),
        ("Results — National Olympiad in Informatics, Round 2",
         "The results of Round 2 are now published. Log in to see your score and "
         "whether you qualified for the national round."),
        ("21% off your next order this weekend",
         "Treat yourself! Use code SAVE10 at checkout. Ends Sunday."),
    ]
    for subject, body in tests:
        r = triage(subject, body)
        notify = r.importance >= 2
        flag = "🔔 NOTIFY" if notify else "  quiet"
        print(f"{flag} | {r.category.value:12} imp={r.importance} | {subject[:40]}")
        print(f"         └ {r.reason}\n")
##