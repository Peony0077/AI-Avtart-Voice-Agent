# Houston Party Network knowledge (HPN mode)

What the avatar answers HPN's customers from, in profile `hpn` (`config/profiles/hpn.yaml`). The avatar is
the website's **Event helper** with a face and a voice: every answer is the Event helper's own, word for word.

| File | What it is | Where it comes from |
|---|---|---|
| `event-helper-knowledge.json` | 111 topics (Craig's example questions, keywords, answers, buttons, follow-up questions), the business details, the packages and add-ons | HPN's export of its **live database**: `docs/event-helper-knowledge/` in the HPN project (`E:\HPN`, GitHub `Peony0077/HPN`), copied unchanged |
| `catalogue.json` | the 21 packages and add-ons with today's prices | `https://h-townparty.com/api/plan/catalog/` (the planner's own catalogue) laid over the export's; differences are listed in the file |
| `schedule.json` | every public night: karaoke, Jukebox Bingo, DJ party (venue, town, weekday, start time, every week / every other week) | the schema.org event data of `https://h-townparty.com/night-out/` |
| `avatar_overrides.json` | what the avatar does **not** take word for word, and why | written for the avatar (see below) |

## Refreshing

When prices, nights or answers change on the website:

```
# 1. In the HPN project, export the knowledge again (see its docs/event-helper-knowledge/README.md),
#    or pull the HPN repository if someone else did.
# 2. Here:
.venv/bin/python scripts/hpn_refresh.py --export ~/projects/HPN/docs/event-helper-knowledge/event-helper-knowledge.json
#    (Windows: .venv\Scripts\python scripts\hpn_refresh.py --export E:\HPN\docs\event-helper-knowledge\event-helper-knowledge.json)
# 3. Restart the avatar (scripts/run.sh --profile hpn, or scripts/tunnel.sh --profile hpn --detach).
```

The refresh only reads the website (plain page requests). It checks everything before writing: every topic has an
answer and no unfilled `{placeholder}`, every button has a page, every every-other-week night has its date. If a
check fails, nothing is written and the avatar keeps the last good copy. It prints the live counts next to the
export's (`{venues}`, `{regular_events}`...) and every price that changed.

## How an answer is made (`backend/app/hpn/`)

- `text.py` is HPN's own file (`backend/apps/helper/text.py`), copied unchanged.
- `engine.py` is HPN's `backend/apps/helper/engine.py` rewritten without Django: the same order of rules, the same
  scores, the same wording. Prices come from `catalogue.json`, nights from `schedule.json`, topics from the export.
  `backend/tests/test_hpn_engine.py` runs HPN's own helper tests against it (with HPN's test schedule).
- `voice.py` makes the spoken version: the text as written, plus the cards' facts (prices, tonight's nights) read
  where the website shows them. The page shows the answer like the website does (bold, cards, buttons that open
  h-townparty.com, suggested next questions).
- `match.py`: when the helper is not sure (often a misheard spoken word), the local LLM is shown the list of
  questions the helper can answer and picks one by number (or none). The chosen question goes back through the
  helper and is used only if the helper is sure of it; the page then shows "Answering: ...". The LLM never writes
  an answer.

## What differs from the website, and why (`avatar_overrides.json`, `engine.py`)

- **Customers only:** the DJs' team-app topics (category `team`) are left out.
- **Who is answering:** she says her name and that she is an AI avatar ("not a person, and not AI" is the website
  helper's own truth, not hers). The three topics that describe the website's text helper ("What happens to what I
  type here?", "What happened to the AI chat?", "Do you use AI?") are rewritten to say only what is true of the
  avatar, keeping their facts about HPN.
- **Who is DJing now:** the website reads HPN's private DJ rota; the avatar cannot, so it points to the Request a
  song page and never says who is (or is not) on.
- **Privacy in HPN mode:** no memory of customers, no leads, no summaries, no saved audio; phone numbers, emails and
  card numbers are masked in the log (HPN's own rule).

## Enquiries: a customer's details into HPN's inbox (`backend/app/hpn/enquiry.py`)

When a customer leaves a phone number or an email, or asks to be called back, the avatar takes down an enquiry:
it asks for what is missing (HPN needs a **name and an email**; typing the email is suggested, since speech
recognition mishears emails), **reads it all back**, and sends it **only after the customer says yes**. It goes
through HPN's own public endpoint, the one the Plan My Event form and HPN's app use
(`POST https://h-townparty.com/api/app/v1/plan/submit/`): HPN saves the Lead (source **"AI avatar"**), emails
info@h-townparty.com and rings the team's bell, as for any enquiry. The message holds the customer's own questions
from the conversation. Card numbers are never taken or sent. HPN's code is not changed.

`hpn.enquiries` in `config/profiles/hpn.yaml`: `live` (sent to HPN), `test` (the whole conversation, nothing sent:
use it for trying things out, every live test becomes a real lead in Craig's inbox) or `off`. HPN accepts at most
10 enquiries an hour from one address, and every avatar enquiry comes from the avatar PC's address.
