"""A customer's contact details, sent to Houston Party Network as an enquiry (a Lead in HPN's database).

When a customer leaves a phone number or an email, or asks to be called back, the avatar takes down an
enquiry - only with their say-so, read back first - and sends it through HPN's own public endpoint, the one
the Plan My Event form and HPN's app post to:

    POST https://h-townparty.com/api/app/v1/plan/submit/     (HPN backend/apps/leads/views.py submit_view)

HPN then does what it does for every enquiry: the Lead is saved (source "AI avatar"), Craig is emailed at
info@h-townparty.com, the team's bell rings. Nothing in HPN's code is changed. HPN requires a name and a valid
email (it answers 400 without them), so both are asked for; a phone number and the kind of event are sent
when known, and the customer's own questions in this conversation go in the message, so Craig knows what
they asked.

Spoken emails are where speech recognition fails most ("jo at example dot com" -> "joe at example.com"), so
the avatar reads everything back and sends nothing until the customer says yes; typing the email is offered.
The same rule HPN's AI concierge kept: name and email, a read-back, then an explicit yes.

Modes (config hpn.enquiries): "live" sends to HPN; "test" goes through the whole conversation and sends
nothing (said so to the customer, logged); "off" keeps the website helper's answer (details are not taken).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from app.hpn import text as T
from app.hpn.engine import CARD_RE, EMAIL_RE, PHONE_RE, Reply, _page, mask_private
from app.hpn.knowledge import SITE

SUBMIT_URL = f"{SITE}/api/app/v1/plan/submit/"
SOURCE = "AI avatar"                     # Lead.source in HPN's inbox (max 40 characters)
EVENT_TYPES = ("wedding", "foam", "corporate", "school", "party")   # HPN EventType values the avatar can tell

YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "send", "send it", "yes send it",
       "yes please", "please send", "go ahead", "that is right", "that is correct", "sounds good", "perfect",
       "do it", "confirm", "yes it is", "si"}
NO = {"no", "nope", "wrong", "change", "change it", "no change it", "that is wrong", "not right", "incorrect",
      "fix it", "edit"}
CANCEL = {"cancel", "stop", "never mind", "nevermind", "forget it", "do not send", "dont send", "no thanks",
          "no thank you", "not now"}
# "Can someone call me back" and the like: the flow starts even without details.
WANTS_CONTACT = ["call me", "call me back", "someone call me", "can you call me", "could you call me",
                 "give me a call", "call back", "callback", "ring me", "text me", "have someone call",
                 "have craig call", "contact me", "reach me", "reach out to me", "email me", "get back to me",
                 "leave my number", "leave my email", "leave my details", "send my details", "take my details",
                 "take my number", "take my email", "llamame"]
NAME_RE = re.compile(r"\b(?:my name is|my name's|name is|this is|i am|i'm|im|call me)\s+"
                     r"([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})", re.I)
NOT_NAMES = {"planning", "looking", "interested", "getting", "having", "hosting", "here", "ok", "okay", "fine", "good",
             "a", "an", "the", "not", "just", "so", "very", "trying", "calling", "asking", "wondering", "back",
             "on", "at", "free", "available", "sure", "ready", "done", "booking", "married", "engaged"}
SPOKEN_AT = re.compile(r"\b([a-z0-9][a-z0-9._%+\-]*)\s+at\s+([a-z0-9\-]+(?:\s*(?:dot|\.)\s*[a-z0-9\-]+)+)\b", re.I)


def spoken_email(text: str) -> str:
    """ "jo at example dot com" (what speech recognition writes) -> "jo@example.com"; anything else unchanged."""
    def join(m: re.Match) -> str:
        domain = re.sub(r"\s*(?:dot|\.)\s*", ".", m[2], flags=re.I)
        return f"{m[1]}@{domain}"
    return SPOKEN_AT.sub(join, text)


def find_email(text: str) -> str | None:
    m = EMAIL_RE.search(spoken_email(text))
    return m.group().strip(".").lower() if m else None


def find_phone(text: str) -> str | None:
    """A US phone number as (713) 555-0199; a card number is never taken for one."""
    if CARD_RE.search(text):
        return None
    m = PHONE_RE.search(text)
    if not m:
        return None
    digits = "".join(ch for ch in m.group() if ch.isdigit())[-10:]
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def find_name(text: str, asked: bool = False) -> str | None:
    """ "my name is Jo Smith" -> "Jo Smith". When the avatar has just asked for the name, a short answer of
    one to three words ("Jo Smith") is the name."""
    for m in NAME_RE.finditer(text):
        name = []
        for w in m.group(1).split():           # up to the first word that is not part of a name
            if w.lower() in NOT_NAMES or w.lower() in ("and", "my", "email", "phone", "number", "from", "with"):
                break
            name.append(w)
        if name:
            return " ".join(w[:1].upper() + w[1:] for w in name)[:120]
    if asked:
        plain = re.sub(r"[^A-Za-z'\- ]", " ", EMAIL_RE.sub(" ", PHONE_RE.sub(" ", text))).split()
        plain = [w for w in plain if w.lower() not in ("it", "is", "its", "my", "name", "the", "please")]
        if 1 <= len(plain) <= 3 and not any(w.lower() in NOT_NAMES for w in plain):
            return " ".join(w[:1].upper() + w[1:] for w in plain)
    return None


def event_type_from(intents: list[str], said: list[str]) -> str:
    """The kind of event the conversation was about: the customer's words first, then the answers' subjects."""
    for text in reversed(said):
        found = T.concepts(T.normalise(text))
        for kind in EVENT_TYPES:
            if kind in found:
                return kind
    for intent in reversed(intents):
        for kind in EVENT_TYPES:
            if kind in intent:
                return kind
        if any(w in intent for w in ("parties", "birthday", "private")):
            return "party"
    return ""


@dataclass
class Draft:
    name: str = ""
    email: str = ""
    phone: str = ""
    event_type: str = ""
    step: str = "collect"             # collect -> confirm -> (sent | cancelled)
    asked: str = ""                   # what the last question was for: "email" | "name" | "change"


@dataclass
class EnquiryFlow:
    """One conversation's enquiry. `send` posts to HPN (or pretends to, in test mode)."""
    business: dict
    mode: str = "live"                # live | test | off
    draft: Draft | None = None
    sent: bool = False
    said: list[str] = field(default_factory=list)       # the customer's own messages (the enquiry's message)
    intents: list[str] = field(default_factory=list)    # the answers' intents (the kind of event)
    pending: dict | None = None                         # confirmed, to be posted by the session

    @property
    def active(self) -> bool:
        return self.draft is not None and self.draft.step in ("collect", "confirm")

    def remember(self, said: str, intent: str) -> None:
        if said.strip():
            self.said.append(mask_card(said.strip())[:400])
            del self.said[:-8]
        if intent:
            self.intents.append(intent)
            del self.intents[:-8]

    def wants_to_start(self, text: str) -> bool:
        if self.mode == "off" or self.active:
            return False
        norm = T.normalise(text)
        return bool(find_email(text) or find_phone(text)) or T.any_phrase(norm, WANTS_CONTACT)

    # --- the conversation ---------------------------------------------------------------------------
    def start(self, text: str) -> Reply:
        self.draft = Draft(event_type=event_type_from(self.intents, self.said + [text]))
        self._take(text)
        if self.sent:
            return self._reply("You've already sent your details to the team in this conversation, so they'll "
                               "be in touch. If anything has changed, or it's urgent, call "
                               f"**{self.business['phone_display']}**.", intent="enquiry.already_sent",
                               chips=["Prices and packages", "What's on tonight?"], done=True)
        return self._next(intro=True)

    def handle(self, text: str) -> Reply | None:
        """The customer's answer while an enquiry is being taken down; None when they asked something else
        (the enquiry is then dropped, and the answer says nothing was sent)."""
        d = self.draft
        norm = T.normalise(text)
        if norm in CANCEL or T.any_phrase(norm, ["do not send", "never mind", "cancel"]):
            self.draft.step = "cancelled"
            return self._reply("No problem, I haven't sent anything. Is there anything else I can help with?",
                               intent="enquiry.cancelled", chips=["Prices and packages", "What's on tonight?"],
                               done=True)
        if d.step == "confirm":
            if norm in YES or (norm.startswith("yes") and len(norm.split()) <= 4):
                return self._send()
            if self._took(text):              # a corrected detail ("the email is ..."): read back again
                return self._next()
            if d.asked == "change" and find_name(text, asked=True):
                d.name = find_name(text, asked=True)
                return self._next()
            if norm in NO or T.any_phrase(norm, ["change", "wrong", "not right", "fix"]):
                d.asked = "change"
                return self._reply("Of course. Tell me what to change: the right name, email or phone. You "
                                   "can type it in the box.", intent="enquiry.change", chips=["Cancel"])
            return None
        took = self._took(text)
        if not took and d.asked == "name":
            name = find_name(text, asked=True)
            if name:
                d.name = name
                took = True
        if not took:
            return None
        return self._next()

    def _took(self, text: str) -> bool:
        """Details in the text, taken into the draft; whether there were any."""
        d = self.draft
        before = (d.name, d.email, d.phone)
        self._take(text)
        return (d.name, d.email, d.phone) != before

    def _take(self, text: str) -> None:
        d = self.draft
        d.email = find_email(text) or d.email
        d.phone = find_phone(text) or d.phone
        d.name = find_name(text, asked=d.asked == "name") or d.name

    def _next(self, intro: bool = False) -> Reply:
        d = self.draft
        lead = "I can pass your details to the Houston Party Network team so they can get back to you. " if intro else ""
        if not d.email:
            d.asked = "email"
            return self._reply(f"{lead}What email should the team reply to? Typing it in the box is the surest way.",
                               intent="enquiry.ask_email", chips=["Cancel"])
        if not d.name:
            d.asked = "name"
            return self._reply(f"{lead}And what name should they use?", intent="enquiry.ask_name", chips=["Cancel"])
        d.step, d.asked = "confirm", ""
        lines = [f"Name: **{d.name}**", f"Email: **{d.email}**"]
        if d.phone:
            lines.append(f"Phone: **{d.phone}**")
        if d.event_type:
            lines.append(f"About: **{EVENT_LABEL.get(d.event_type, d.event_type)}**")
        test = " (Test mode: nothing will really be sent.)" if self.mode == "test" else ""
        return self._reply("Here's what I'll send to the Houston Party Network team:\n\n" + "\n".join(lines)
                           + f"\n\nIs that right? Shall I send it?{test}", intent="enquiry.confirm",
                           chips=["Yes, send it", "No, change it", "Cancel"])

    def _send(self) -> Reply:
        d = self.draft
        payload = {"name": d.name, "email": d.email, "phone": d.phone, "event_type": d.event_type,
                   "message": self._message(), "source": SOURCE}
        if self.mode == "test":
            d.step = "sent"
            self.sent = True
            logger.info(f"HPN enquiry (TEST MODE, not sent): event {d.event_type or '-'}, phone "
                        f"{'yes' if d.phone else 'no'}")
            return self._reply("Test mode: this is where your details would go to the team. Nothing was sent.",
                               intent="enquiry.sent_test", chips=["Prices and packages", "What's on tonight?"],
                               done=True)
        # the session posts it (it awaits HTTP: post() below) and answers with finish()
        self.pending = payload
        return self._reply("", intent="enquiry.sending")

    def finish(self, status: int | None, body: dict | None) -> Reply:
        """The answer once HPN has replied to the post (status None: no reply at all)."""
        d = self.draft
        b = self.business
        self.pending = None
        if status in (200, 201) and (body or {}).get("ok"):
            d.step = "sent"
            self.sent = True
            return self._reply("Done: your details are with the Houston Party Network team, and they reply to "
                               f"every enquiry. If it's urgent, call **{b['phone_display']}**.",
                               intent="enquiry.sent", chips=["Prices and packages", "What's on tonight?"],
                               done=True)
        if status == 400:
            d.step, d.asked, d.email = "collect", "email", ""
            return self._reply("The team's system didn't accept that email address. Could you type it in the box "
                               "again?", intent="enquiry.ask_email", chips=["Cancel"])
        d.step = "cancelled"
        why = "a lot of enquiries have just been sent from here" if status == 429 else "I couldn't reach it just now"
        return self._reply(f"Sorry, your details weren't sent: {why}. Please use the contact form, call "
                           f"**{b['phone_display']}** or email **{b['email']}**.", intent="enquiry.failed",
                           links=self._contact_links(), done=True)

    def _message(self) -> str:
        asked = "\n".join(f"- {s}" for s in self.said[-6:]) or "-"
        return (f"Sent from Houston Party Network's AI avatar, with the customer's confirmation.\n\n"
                f"What they asked in the conversation:\n{asked}")[:4000]

    # --- helpers ------------------------------------------------------------------------------------
    def _contact_links(self) -> list[dict]:
        b = self.business
        return [_page("contact", "Contact form"), {"label": f"Call {b['phone_display']}", "path": f"tel:{b['phone_e164']}"},
                {"label": f"Email {b['email']}", "path": f"mailto:{b['email']}"}]

    def _reply(self, text: str, intent: str, chips: list[str] | None = None, links: list[dict] | None = None,
               done: bool = False) -> Reply:
        if done and self.draft and self.draft.step not in ("sent", "cancelled"):
            self.draft.step = "cancelled"
        return Reply(kind="enquiry", intent=intent, text=text, chips=chips or [], links=links or [], confident=True)


EVENT_LABEL = {"wedding": "a wedding", "foam": "a foam party", "corporate": "a corporate event",
               "school": "a school or community event", "party": "a party"}


def mask_card(text: str) -> str:
    """Card numbers never leave the avatar (nor its log); phone numbers and emails do, in an enquiry."""
    return CARD_RE.sub("[number removed]", text)


async def post(payload: dict, session_key: str, timeout: float = 15.0) -> tuple[int | None, dict | None]:
    """POST the enquiry to HPN. (status, json) or (None, None) when HPN could not be reached."""
    import httpx

    body = {**payload, "session": f"avatar-{session_key}"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "HPN-AI-Avatar/1"}) as client:
            resp = await client.post(SUBMIT_URL, json=body)
        try:
            data = resp.json()
        except ValueError:
            data = None
        logger.info(f"HPN enquiry posted: HTTP {resp.status_code}, lead {(data or {}).get('lead_id')}")
        return resp.status_code, data
    except Exception as exc:  # noqa: BLE001 - the customer is told and given the other ways
        logger.warning(f"HPN enquiry not sent: {exc!r} ({mask_private(str(payload.get('email', '')))})")
        return None, None
