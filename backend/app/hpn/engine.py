"""
Houston Party Network's Event helper, for the avatar: a question in, a Reply out.

A port of the website's own helper (HPN backend/apps/helper/engine.py @ the commit in text.py's header):
the same order of rules, the same scores, the same wording, so the avatar answers a question exactly as
h-townparty.com's Event helper does. Deterministic, no AI writes an answer: every price, night, phone
number and policy comes from knowledge/hpn/ (app.hpn.knowledge). The differences, all deliberate:

  * Data comes from Knowledge (the export, the live catalogue and schedule) instead of HPN's database.
  * Who is answering: the avatar says her name and that she is an AI avatar ("not AI" was the website
    helper's own truth, not hers) - AVATAR_TEXT below; the website-helper-only topics are rewritten in
    knowledge/hpn/avatar_overrides.json.
  * "Who is DJing now" needs HPN's private shift rota, which is not public: the reply points to the
    Request a song page instead of saying who is (or is not) on.
  * Wording that assumed typing ("a number typed into this chat") is said for voice and text alike.

The order a question is tried in, and why (HPN's own notes):
  1. Things that must never be mis-answered: an empty box, a card number, a greeting on its own.
  2. Talk about the conversation itself: who are you, a human please, a complaint.
  3. LIVE answers: package prices from the catalogue, the venue schedule, the phone and email, the towns.
  4. Craig's knowledge base, scored by how closely the question matches a topic's example questions and
     keywords. Above CONFIDENT it is the answer.
  5. Money and terms nobody has written a topic for (discounts, deposits, cancellations): never made up.
  6. Small talk, insults, off-topic questions, key-mashing.
  7. The polite fallback: "I'm not sure about that one yet", the closest topics, and how to reach a person.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.hpn import text as T
from app.hpn.knowledge import HOUSTON, SITE, Knowledge, clock, number_word

HIDDEN_OFFERINGS = {"outdoor-movie-night"}
STARTER_CHIPS = ["Plan an event", "Prices and packages", "What's on tonight?",
                 "Request a song", "Weddings", "Foam parties", "Contact us"]
PAGES = {
    "plan": ("Plan My Event", "/plan-my-event/"),
    "request": ("Request a song", "/request/"),
    "nights": ("Nights out", "/night-out/"),
    "contact": ("Contact us", "/contact-us/"),
    "weddings": ("Weddings", "/weddings/"),
    "parties": ("Parties & Events", "/parties-and-events/"),
    "foam": ("Foam Parties", "/foam-parties/"),
    "karaoke": ("Karaoke", "/karaoke/"),
    "bingo": ("Jukebox Bingo", "/jukebox-bingo/"),
    "venues": ("Bars & Venues", "/for-bars-and-venues/"),
    "story": ("Our story", "/story/"),
    "about": ("About us", "/about-us/"),
    "privacy": ("Terms & Privacy", "/terms-and-privacy/"),
    "delete": ("Delete your account", "/accounts/delete/"),
    "birthday": ("Birthdays", "/planning-a-birthday/"),
    "private": ("Private Parties", "/planning-a-private-party/"),
    "corporate": ("Corporate Events", "/planning-a-corporate-event/"),
    "school": ("School & Community", "/planning-a-school-or-community-event/"),
    "portal": ("My events", "/portal/"),
    "signin": ("Sign in", "/accounts/login/"),
}
# HPN apps/helper/models.py: card numbers (13-19 digits with spaces or dashes), phones, emails.
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def mask_private(text: str) -> str:
    """A question as it may be logged: card numbers, phone numbers and emails masked (HPN's rule)."""
    text = CARD_RE.sub("[number removed]", text or "")
    text = EMAIL_RE.sub("[email removed]", text)
    return PHONE_RE.sub("[phone removed]", text)


@dataclass
class Card:
    title: str
    subtitle: str = ""
    detail: str = ""
    path: str | None = None

    def as_dict(self) -> dict:
        return {"title": self.title, "subtitle": self.subtitle, "detail": self.detail, "path": self.path,
                "url": SITE + self.path if self.path and self.path.startswith("/") else self.path}


@dataclass
class Reply:
    kind: str
    intent: str
    text: str
    cards: list = field(default_factory=list)
    links: list = field(default_factory=list)
    chips: list = field(default_factory=list)
    confident: bool = True

    def as_dict(self) -> dict:
        links = _dedupe_links(self.links)
        chips = list(dict.fromkeys(c for c in self.chips if c))[:7]
        if not links and not chips:           # every reply ends with somewhere to go
            chips = STARTER_CHIPS[:4]
            links = [_page("contact")]
        return {"kind": self.kind, "intent": self.intent, "text": self.text.strip(),
                "cards": [c.as_dict() for c in self.cards], "links": [_absolute(link) for link in links],
                "chips": chips, "confident": self.confident}


def _page(key: str, label: str | None = None, path: str | None = None) -> dict:
    name, default = PAGES[key]
    return {"label": label or name, "path": path or default}


def _absolute(link: dict) -> dict:
    """The website's buttons are site paths; the avatar's page is not the website, so they open it."""
    path = link["path"]
    url = path if path.startswith(("tel:", "mailto:", "http")) else SITE + path
    return {"label": link["label"], "path": path, "url": url}


def _dedupe_links(links: list[dict]) -> list[dict]:
    seen, out = set(), []
    for link in links:
        if link["path"] not in seen:
            seen.add(link["path"])
            out.append(link)
    return out[:5]


def _and(items) -> str:
    items = [str(i) for i in items]
    if len(items) < 2:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def houston_now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(HOUSTON)).astimezone(HOUSTON)


def good_time_of_day(now: datetime | None = None) -> str:
    hour = houston_now(now).hour
    if 5 <= hour < 12:
        return "Good morning!"
    if 12 <= hour < 17:
        return "Good afternoon!"
    return "Good evening!"


FAMILY_PAGE = {"wedding": "weddings", "foam": "foam", "party": "parties", "corporate": "corporate",
               "school": "school"}
FAMILY_PLAN = {
    "wedding": {"label": "Plan your wedding", "path": "/contact-us/?event=wedding"},
    "foam": {"label": "Plan my foam party", "path": "/plan-my-event/?event_hint=foam"},
    "party": {"label": "Plan my party", "path": "/plan-my-event/?event=party"},
    "corporate": {"label": "Plan my event", "path": "/plan-my-event/?event=corporate"},
    "school": {"label": "Plan my event", "path": "/plan-my-event/?event=school"},
}
ADDON_ALIASES = {
    "photo-booth": ["photo booth", "photobooth", "photo booths", "selfie booth", "photo station"],
    "photo-booth-print-station": ["print station", "prints", "photo prints", "printing"],
    "audio-guest-book": ["audio guest book", "audio guestbook", "voice guest book", "phone guest book",
                         "guest book phone", "audio book"],
    "video-guest-book": ["video guest book", "video guestbook"],
    "guest-book-booth": ["guest book booth", "guestbook booth"],
    "karaoke-addon": ["karaoke"],
    "wireless-uplighting": ["uplighting", "uplights", "up lighting", "up lights", "uplight"],
    "dancing-on-a-cloud": ["dancing on a cloud", "dancing on the cloud", "cloud", "dry ice", "low fog", "fog"],
    "monograms": ["monogram", "monograms", "logo projection", "name on the wall", "projected name", "gobo"],
    "led-party-sticks": ["led sticks", "led party sticks", "led party stick", "glow sticks", "light sticks",
                         "led batons", "led"],
    "foam-dj": ["foam dj", "dj for foam", "dj for my foam", "dj for the foam", "dj with foam", "dj to foam"],
    "jukebox-bingo-private": ["jukebox bingo", "music bingo", "bingo"],
}
GUEST_BOOKS = ["audio-guest-book", "video-guest-book", "guest-book-booth"]
PACKAGE_WORDS = {
    "headliner": ["headliner-wedding-experience", "headliner-party"],
    "signature": ["signature-wedding", "signature-party"],
    "essentials": ["reception-essentials", "party-essentials"],
}
EVENT_ORDER = ("wedding", "foam", "corporate", "school", "party")
WEEKDAYS = {
    "monday": 0, "mondays": 0, "mon": 0, "tuesday": 1, "tuesdays": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wednesdays": 2, "wed": 2, "weds": 2, "thursday": 3, "thursdays": 3,
    "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fridays": 4, "fri": 4, "saturday": 5,
    "saturdays": 5, "sunday": 6, "sundays": 6, "lunes": 0, "martes": 1, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6,
}
MONTHS = ("january february march april may june july august september october november "
          "december jan feb mar apr jun jul aug sep sept oct nov dec").split()
DATEISH_RE = re.compile(r"\b(\d{1,2} \d{1,2}( \d{2,4})?|\d{1,2}(st|nd|rd|th))\b")
DJ_NOW_PHRASES = ["who is djing", "who is the dj", "who is dj", "which dj", "who is playing",
                  "who is on", "dj on now", "djs on now", "who is spinning", "taking requests",
                  "dj tonight", "djs tonight", "who is your dj", "find my dj", "find the dj",
                  "dj is on", "is there a dj", "which djs"]
TIP_WORDS = ["tip", "tips", "tipping", "venmo", "cash app", "cashapp", "paypal", "zelle"]
ASKING_FOR = ["can you", "could you", "do you do", "do you offer", "do you provide",
              "need a dj", "need a", "looking for", "want a", "want to book", "hire", "book",
              "dj for", "play at", "for my", "for our", "necesito", "quiero", "para una",
              "para mi", "busco"]
PRIORITY = ["pay to", "pay extra", "played sooner", "play it sooner", "skip the queue",
            "jump the queue", "skip the line", "move my song up", "priority", "get bumped"]
GUEST_SIDE = ["do not play", "must play", "first dance", "last dance", "our song", "guests",
              "guest", "our wedding", "entrance", "walk down", "processional", "father daughter",
              "mother son", "cake cutting"]
MOVIE_WORDS = ["movie", "movies", "outdoor movie", "movie night", "movie nights",
               "outdoor cinema", "movie screen", "projector", "film night"]
WHERE_US = ["where are you", "where r you", "where r u", "where are u", "where is your office",
            "where is your shop", "where is houston party network", "where is hpn",
            "your location", "where are y all", "where do you operate from",
            "where do you work from", "what city are you in", "are you local"]
# AVATAR (contact questions, 3 Nov 2026). Asking for a person's own number: the business line answers it
# (the knowledge has no personal numbers, and HPN's docs say Craig's own line is never given).
PERSON_WORDS = ["craig", "owner", "manager", "boss", "the dj", "your dj"]
CONTACT_WORDS = ["number", "phone", "cell", "cellphone", "mobile", "email", "e mail", "contact", "reach", "call",
                 "text", "whatsapp", "personal"]
# A venue's own phone or address (published on /night-out/), not Houston Party Network's.
VENUE_CONTACT_WORDS = ["phone", "number", "call", "address", "located", "location", "where is", "where are",
                       "directions", "how do i get", "contact", "reach"]
# When the team answers: no source says (business hours, reply times) - never the nights out.
REPLY_TIMES = ["do you answer", "do you reply", "do you respond", "answer the phone", "answer calls", "pick up",
               "are you open", "business hours", "office hours", "opening hours", "open hours", "when are you open",
               "get back to me", "reply on", "respond on", "how fast do you", "how quickly do you", "how long to reply"]
DETAILS_NOTE = ("Please don't leave your phone number or email with me: I can't pass details on, so they won't reach "
                "the team. To be contacted, send them through the contact form.")
VENUE_SUFFIXES = {"bar", "pub", "grill", "grille", "brewery", "brewing", "restaurant",
                  "tavern", "saloon", "lounge", "cantina", "icehouse", "company", "co"}
SIGNOFF_FILLER = set("""and ok okay so much again you all for now have a great good nice lovely
day night evening weekend then everyone guys team very really oh well alright cool perfect your
help the info information that is it awesome super hpn""".split())
ACK = ["ok", "okay", "cool", "great", "awesome", "nice", "got it", "sounds good", "perfect",
       "alright", "k", "kk", "sure", "yes", "no", "yep", "nope", "lol", "haha", "good",
       "fine", "wow", "neat", "understood", "noted"]
BOOKED = ("When you book, the agreement is the written quote and confirmation we send you, "
          "so the team will confirm it there.")
ASK = "The team will be glad to answer it."
POLICY_WORDS = {
    "policy_discount": ("discounts or special offers", BOOKED),
    "policy_deposit": ("deposits", BOOKED),
    "policy_contract": ("contract terms", BOOKED),
    "policy_cancel": ("cancellations, refunds or date changes", BOOKED),
    "policy_insurance": ("insurance", ASK),
    "policy_power": ("power and space at your venue",
                     "Tell us about your venue when you enquire, and the team will check it with you."),
    "policy_cover": ("cover charges or entry at our nights out",
                     "For a night out, check with the venue before you go, or ask our team."),
    "policy_age": ("age limits",
                   "For a night out, check with the venue before you go; for a private event, ask our team."),
    "policy_permit": ("permits", ASK),
    "policy_conduct": ("that", ASK),
    "policy_genre": ("particular styles of music",
                     "Put the songs you want on your planning form's must-play list, or ask the team "
                     "before you book."),
    "policy_health": ("health and safety questions like that",
                      "Please ask the team before the day, so they can tell you what's right for your guests."),
    "policy_dj_choice": ("which DJ plays your event", "Ask the team when you enquire."),
    "policy_arrival": ("arrival times",
                       "The timings are agreed with you before the day, and setup and teardown don't reduce "
                       "your scheduled entertainment time."),
}


@dataclass
class Question:
    raw: str
    norm: str
    words: list
    concepts: set
    event_type: str | None = None
    weekday: int | None = None
    towns: list = field(default_factory=list)
    kinds: list = field(default_factory=list)
    venues: list = field(default_factory=list)


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def bag(norm: str) -> set:
    return {_stem(w) for w in T.content_words(norm)} | {"#" + c for c in T.concepts(norm)}


def _venue_key(norm: str) -> str:
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in norm.split())


def _lower_first(price_display: str) -> str:
    if price_display.startswith("Starting at"):
        return "s" + price_display[1:]
    return price_display


def _lower_first_word(item: str) -> str:
    if len(item) > 1 and item[0].isupper() and item[1].islower():
        return item[0].lower() + item[1:]
    return item


class Engine:
    """The Event helper over one Knowledge. `name` is the avatar's name (the owner's persona setting)."""

    def __init__(self, kb: Knowledge, name: str = "Emma") -> None:
        self.kb = kb
        self.name = name
        self._index = self._build_index()
        self._vocab = self._build_vocabulary()
        self._town_aliases = self._known_towns()
        self._venue_aliases = self._known_venues()

    # --- facts ---------------------------------------------------------------------------------------
    def business(self) -> dict:
        return self.kb.business

    def contact_links(self) -> list[dict]:
        b = self.business()
        return [{"label": f"Call {b['phone_display']}", "path": f"tel:{b['phone_e164']}"},
                {"label": f"Email {b['email']}", "path": f"mailto:{b['email']}"},
                _page("contact", "Contact form")]

    def spanish_line(self) -> str:
        b = self.business()
        return f"Si prefiere, llámenos al {b['phone_display']} o escríbanos a {b['email']}."

    def offerings(self, kind: str | None = None) -> list:
        rows = [o for o in self.kb.offerings if o.slug not in HIDDEN_OFFERINGS and (kind is None or o.kind == kind)]
        return sorted(rows, key=lambda o: (o.kind, o.sort_order, o.name))

    @staticmethod
    def suits(offering, event_type: str | None) -> bool:
        if not event_type:
            return True
        suits = offering.suits_event_types or []
        return not suits or event_type in suits

    def packages_for(self, event_type: str) -> list:
        return [o for o in self.offerings("package") if self.suits(o, event_type)]

    @staticmethod
    def _family_of(o) -> str | None:
        for family in ("wedding", "foam", "party"):
            if family in (o.suits_event_types or []):
                return family
        return None

    def package_card(self, o, event_type: str | None = None) -> Card:
        key = FAMILY_PAGE.get(event_type) or FAMILY_PAGE.get(self._family_of(o)) or "plan"
        return Card(title=o.name + (f" ({o.badge})" if o.badge else ""), subtitle=o.price_display,
                    detail=o.hourly_display or o.tagline or o.summary, path=PAGES[key][1])

    @staticmethod
    def addon_card(o, event_type: str | None = None) -> Card:
        key = FAMILY_PAGE.get(event_type, "parties")
        return Card(title=o.name, subtitle=o.price_display, detail=o.summary, path=PAGES[key][1])

    # --- reading the question --------------------------------------------------------------------------
    def _known_towns(self) -> dict:
        names = set(self.kb.town_names())
        names.update(v.city.strip() for v in self.kb.venues if v.city.strip())
        towns = {}
        for name in names:
            norm = T.normalise(name)
            aliases = {norm}
            if norm.startswith("the "):
                aliases.add(norm[4:])
            towns[name] = sorted(aliases)
        return towns

    def _known_venues(self) -> dict:
        venues = {}
        for name in (v.name for v in self.kb.venues):
            key = _venue_key(T.normalise(name))
            if not key:
                continue
            aliases = {key}
            words = key.split()
            if words and words[0] == "the" and len(words) > 2:
                words = words[1:]
                aliases.add(" ".join(words))
            while len(words) > 2 and words[-1] in VENUE_SUFFIXES:
                words = words[:-1]
                aliases.add(" ".join(words))
            venues[name] = sorted(aliases)
        return venues

    def read(self, text: str) -> Question:
        norm = T.normalise(text)
        found = T.concepts(norm)
        if "price" in found and re.search(r"\bhow much (space|room|area|power|notice|time)\b", norm) \
                and not re.search(r"\b(price|cost|costs|fee|charge|rate|quote)\b", norm):
            found.discard("price")
        if re.search(r"\b(my|team|work) (schedule|shifts?)\b", norm):
            found.discard("what_on")
        q = Question(raw=text, norm=norm, words=T.content_words(norm), concepts=found)
        for event_type in EVENT_ORDER:
            if event_type in found:
                q.event_type = event_type
                break
        for word in norm.split():
            if word in WEEKDAYS:
                q.weekday = WEEKDAYS[word]
                break
        padded = f" {norm} "
        for town, aliases in self._town_aliases.items():
            if any(f" {a} " in padded for a in aliases):
                q.towns.append(town)
        keyed = f" {_venue_key(norm)} "
        for venue, aliases in self._venue_aliases.items():
            if any(f" {a} " in keyed for a in aliases):
                q.venues.append(venue)
        if "karaoke" in found:
            q.kinds.append("karaoke")
        if "bingo" in found:
            q.kinds.append("jukebox_bingo")
        if T.any_phrase(norm, ["dj party", "dj night", "dj parties", "dance night"]):
            q.kinds.append("dj")
        return q

    # --- the answer -------------------------------------------------------------------------------------
    def start(self, now: datetime | None = None) -> Reply:
        """The opening words: the website helper's, said by the avatar."""
        return Reply(
            kind="greeting", intent="greeting",
            text=(f"{good_time_of_day(now)} I'm {self.name}, Houston Party Network's AI Agent. "
                  f"I answer from our own prices and schedule.\n\nAsk me about packages and prices, what's on "
                  f"tonight, song requests or how booking works. I'm not a person, so for anything else, call "
                  f"or email us."),
            chips=STARTER_CHIPS, links=[_page("plan"), _page("contact")])

    def ask(self, text: str, last: str | None = None, now: datetime | None = None) -> Reply:
        """Answer one question. `last` is the previous reply's intent, for "and tomorrow?"."""
        raw = (text or "")[:2000]
        if not raw.strip():
            return self._empty()
        if CARD_RE.search(raw):
            return self._card_number()
        details = self._customer_details(raw)
        if details:
            raw = EMAIL_RE.sub(" ", PHONE_RE.sub(" ", raw))     # answered without them, never repeated back
        norm = T.normalise(raw)
        if not norm:
            return self._empty()
        greeted, rest = T.strip_greeting(norm)
        spanish = T.is_spanish(norm)
        if greeted and not rest:
            reply = self._greeting(now, spanish)
        elif greeted and T.any_phrase(rest, T.HOW_ARE_YOU) and not self._substantive(rest):
            reply = self._how_are_you(now, greeted=True)
        else:
            reply = self.route(rest if greeted else norm, raw, last, now)
            if greeted and reply.kind != "greeting":
                hello = "¡Hola!" if T.has_phrase(norm, "hola") else good_time_of_day(now)
                if T.any_phrase(rest, T.HOW_ARE_YOU):
                    hello += " I'm well, thanks for asking."
                reply.text = f"{hello}\n\n{reply.text}"
        if details:
            reply = self._with_details_note(reply)
        if spanish:
            reply.text = f"{reply.text}\n\n{self.spanish_line()}"
        return reply

    def _customer_details(self, raw: str) -> bool:
        """Whether the message holds the customer's own phone number or email (not Houston Party Network's)."""
        own_phone = "".join(ch for ch in self.business()["phone_e164"] if ch.isdigit())[-10:]
        own_email = self.business()["email"].lower()
        phones = ["".join(ch for ch in m.group() if ch.isdigit())[-10:] for m in PHONE_RE.finditer(raw)]
        emails = [m.group().lower() for m in EMAIL_RE.finditer(raw)]
        return any(p != own_phone for p in phones) or any(e != own_email for e in emails)

    def _with_details_note(self, reply: Reply) -> Reply:
        """Details left with the avatar reach nobody (no leads in HPN mode): said, and the contact form offered.
        A question asked with them is still answered; with nothing else asked, the call-back answer."""
        if reply.kind in ("fallback", "greeting") or reply.intent in (
                "contact", "smalltalk.ack", "smalltalk.empty", "smalltalk.gibberish", "smalltalk.callback",
                "smalltalk.human", "smalltalk.thanks"):
            return self._callback()
        reply.text = f"{DETAILS_NOTE}\n\n{reply.text}"
        reply.links = [_page("contact", "Contact form")] + list(reply.links)
        return reply

    @staticmethod
    def _substantive(norm: str) -> bool:
        return bool(T.concepts(norm) & set(T.DOMAIN_CONCEPTS))

    def route(self, norm: str, raw: str, last: str | None, now: datetime | None) -> Reply:
        q = self.read(norm)
        c = q.concepts
        if norm in ACK:
            return self._ack()
        if T.any_phrase(norm, T.WHO) and not T.any_phrase(norm, ["you guys", "you all", "your company",
                                                                "the company"]):
            return self._who()
        if T.any_phrase(norm, T.CALLBACK):
            return self._callback()
        if T.any_phrase(norm, T.HUMAN):
            return self._human()
        if T.any_phrase(norm, PERSON_WORDS) and T.any_phrase(norm, CONTACT_WORDS) and not q.venues:
            return self._person_contact()     # AVATAR: "What's Craig's number?" got the founding story
        signoff = self._only_signoff(norm)
        if signoff == "bye":
            return self._bye(now)
        if signoff == "thanks":
            return self._thanks()
        if T.any_phrase(norm, T.COMPLAINT):
            return self._complaint()
        if T.any_phrase(norm, T.OFF_TOPIC) and not (c & set(T.DOMAIN_CONCEPTS)):
            return self._off_topic()
        # AVATAR fix (HPN's own helper has this bug, 3 Nov 2026): the follow-up is only a question made of the
        # event's own words ("what about foam?", "and weddings?"). With anything else in it, it is a question of
        # its own: the price answers' own suggestions "What does Signature Wedding include?" and "Is the foam
        # safe?" got the same price list again.
        if (last and last.startswith("prices.") and q.event_type and len(norm.split()) <= 5
                and not c - {q.event_type, "party", "dj"} and self._only_event_words(q)):
            return self._family_prices(q.event_type, q)
        if "price" in c and not c & set(T.POLICY_CONCEPTS) and (self.mentioned_addons(q)
                                                                 or self.mentioned_packages(q)):
            live = self.live_answer(q, last, now)
            if live is not None:
                return live
        topics = self.scored_topics(q)
        best_score, best = topics[0] if topics else (0, None)
        policy = [p for p in T.POLICY_CONCEPTS if p in c]
        if best is not None and self.exact_match(best, q) and (not policy or set(policy) & self.topic_concepts(best)):
            return self.topic_reply(best)
        live = self.live_answer(q, last, now)
        if live is not None:
            return live
        if best is not None and best_score >= self.kb.confident:
            if not policy or set(policy) & self.topic_concepts(best):
                return self.topic_reply(best)
        if policy:
            return self._policy(policy[0])
        if q.event_type and not T.any_phrase(norm, MOVIE_WORDS) and T.any_phrase(norm, ASKING_FOR):
            reply = self._family_prices(q.event_type, q)
            reply.text = f"Yes, we can help with that.\n\n{reply.text}"
            return reply
        if T.any_phrase(norm, T.HOW_ARE_YOU):
            return self._how_are_you(now)
        if T.any_phrase(norm, T.THANKS):
            return self._thanks()
        if T.any_phrase(norm, [p for p in T.BYE if p != "later"]):
            return self._bye(now)
        if T.any_phrase(norm, T.PROFANITY):
            return self._profanity()
        if T.looks_like_gibberish(norm, self._vocab):
            return self._gibberish()
        return self._fallback(topics)

    @staticmethod
    def _only_event_words(q: Question) -> bool:
        allowed = {word for concept in (q.event_type, "party", "dj") for phrase in T.CONCEPTS[concept]
                   for word in phrase.split()}
        return all(word in allowed for word in q.words)

    @staticmethod
    def _only_signoff(norm: str) -> str | None:
        padded = f" {norm} "
        kind = None
        for phrases, name in ((T.BYE, "bye"), (T.THANKS, "thanks")):
            for phrase in sorted(phrases, key=len, reverse=True):
                if f" {phrase} " in padded:
                    padded = padded.replace(f" {phrase} ", " ")
                    kind = "bye" if "bye" in (kind, name) else name
        if kind and set(padded.split()) <= SIGNOFF_FILLER:
            return kind
        return None

    def live_answer(self, q: Question, last: str | None, now: datetime | None) -> Reply | None:
        c, norm = q.concepts, q.norm
        nightish = bool(c & {"tonight", "tomorrow", "weekend", "this_week", "what_on"}) \
            or q.weekday is not None or bool(q.towns) or bool(re.search(r"\bnights?\b", norm))
        booking = bool(c & {"book", "price", "addons", "package", "extra_time"}) \
            or bool(c & {"wedding", "corporate", "school"}) \
            or T.any_phrase(norm, ["my party", "my event", "our party", "our event", "private",
                                   "my birthday", "for a party", "at my", "for my"])
        if c & set(T.POLICY_CONCEPTS):
            return None
        if T.any_phrase(norm, MOVIE_WORDS):
            return None
        if "price" in c and not c & {"book", "package", "addons", "extra_time", "wedding", "corporate",
                                     "school", "party", "foam"} and (
                q.venues or (q.kinds and (c & {"tonight", "tomorrow", "weekend", "this_week"}
                                          or q.weekday is not None or bool(re.search(r"\bnights?\b", norm))))):
            return self._policy("policy_cover")
        if "price" in c and "areas" in c:
            return None
        owner = bool(set(q.words) & {"bar", "venue", "pub", "restaurant", "brewery", "bars", "venues"}) \
            and T.any_phrase(norm, ["my", "our", "own", "we"])
        if owner and not q.event_type and "price" in c:
            return None
        if T.any_phrase(norm, WHERE_US) and not q.kinds and not booking:
            return self._contact()
        # AVATAR: a venue's phone or address is the venue's own, from its card on /night-out/ - HPN's helper
        # gave Houston Party Network's number, which reads as the bar's
        if q.venues and T.any_phrase(norm, VENUE_CONTACT_WORDS) and not booking:
            return self._venue_contact(q)
        # AVATAR: when the team answers is in no source: not tonight's nights ("do you answer on weekends?")
        if T.any_phrase(norm, REPLY_TIMES) and not q.kinds and not q.venues:
            return None
        if "availability" in c and "price" not in c and not (nightish and not booking and not self._dateish(norm)):
            return self._availability(q)
        package_slugs = self.mentioned_packages(q)
        addon_slugs = self.mentioned_addons(q)
        if "price" in c:
            if package_slugs:
                return self._package_details(package_slugs, q, priced=True)
            if addon_slugs:
                return self._addon_prices(addon_slugs, q)
            if "extra_time" in c:
                return self._extra_hours(q)
            if "addons" in c:
                return self._addon_list(q)
            event_type = q.event_type or self._last_family(last)
            if event_type:
                return self._family_prices(event_type, q)
            return self._price_overview()
        if package_slugs:
            return self._package_details(package_slugs, q, priced=False)
        if "extra_time" in c and (q.event_type or c & {"book", "package"}
                                  or T.any_phrase(norm, ["add", "stay", "extend", "cost", "extra hour",
                                                         "extra hours", "additional hour", "additional hours",
                                                         "extra time"])):
            return self._extra_hours(q)
        if "addons" in c and not c & {"tonight", "what_on"}:
            return self._addon_list(q)
        if T.any_phrase(norm, TIP_WORDS) or T.any_phrase(norm, PRIORITY):
            return None
        if T.any_phrase(norm, DJ_NOW_PHRASES) or ("request" in c and not booking
                                                  and not T.any_phrase(norm, GUEST_SIDE)):
            return self._djs_now(now)
        if "areas" in c and not c & {"tonight", "tomorrow", "weekend", "this_week", "what_on"}:
            return self._towns(q)
        if nightish and not booking and "contact" not in c:
            return self._nights(q, now)
        if q.venues and not booking and not c & {"contact", "price"}:
            return self._nights(q, now)
        if q.kinds and not booking and "request" not in c and ("where" in c or "what_on" in c):
            return self._nights(q, now)
        if last and last.startswith("nights.") and (c & {"tonight", "tomorrow", "weekend", "this_week"}
                                                     or q.weekday is not None or q.towns or q.kinds):
            return self._nights(q, now)
        if "areas" in c or (q.towns and booking):
            return self._towns(q)
        if "contact" in c and not set(q.words) & {"app", "iphone", "android", "download", "team", "dj", "djs",
                                                 "song", "songs", "owner", "venue", "delete", "account"}:
            return self._contact()
        return None

    @staticmethod
    def _dateish(norm: str) -> bool:
        return bool(DATEISH_RE.search(norm)) or any(T.has_phrase(norm, m) for m in MONTHS) \
            or T.any_phrase(norm, ["date", "next year", "next month", "next saturday", "next friday",
                                   "this saturday", "this friday"])

    @staticmethod
    def _last_family(last: str | None) -> str | None:
        if not last:
            return None
        for family in EVENT_ORDER:
            if family in last:
                return family
        if "birthday" in last or "parties" in last or "party" in last:
            return "party"
        return None

    def mentioned_packages(self, q: Question) -> list[str]:
        rows = self.offerings("package")
        by_slug = {o.slug: o for o in rows}
        found = [o.slug for o in rows if T.has_phrase(q.norm, T.normalise(o.name))]
        if T.has_phrase(q.norm, "headliner experience") and "headliner-wedding-experience" in by_slug:
            found.append("headliner-wedding-experience")
        if not found:
            for word, slugs in PACKAGE_WORDS.items():
                if T.has_phrase(q.norm, word):
                    family = q.event_type
                    picks = [s for s in slugs if s in by_slug and (family is None or self.suits(by_slug[s], family))]
                    found.extend(picks or [s for s in slugs if s in by_slug])
        return list(dict.fromkeys(found))

    def mentioned_addons(self, q: Question) -> list[str]:
        rows = self.offerings("addon")
        found = []
        for o in rows:
            names = [T.normalise(o.name)] + ADDON_ALIASES.get(o.slug, [])
            if o.slug == "karaoke-addon" and not (q.concepts & {"price"}):
                continue
            if o.slug == "jukebox-bingo-private" and not T.any_phrase(
                    q.norm, ["my party", "my event", "private", "for a party", "book", "at my", "for my",
                             "our party", "our event", "hire"]) and "price" not in q.concepts:
                continue
            if any(T.has_phrase(q.norm, n) for n in names):
                found.append(o.slug)
        if not found and T.any_phrase(q.norm, ["guest book", "guestbook", "guest books"]):
            found = [s for s in GUEST_BOOKS if any(o.slug == s for o in rows)]
        if "foam-dj" in found and not T.has_phrase(q.norm, "dj"):
            found.remove("foam-dj")
        return found

    # --- live replies -------------------------------------------------------------------------------------
    def _family_prices(self, event_type: str, q: Question) -> Reply:
        rows = self.packages_for(event_type)
        cards = [self.package_card(o, event_type) for o in rows]
        plan = FAMILY_PLAN[event_type]
        chips = ["Add-ons", "Extra hours", "How booking works"]
        links = [plan, _page(FAMILY_PAGE[event_type])]
        count = number_word(len(rows))
        if not rows:
            return Reply(kind="live", intent=f"prices.{event_type}", cards=[],
                         text="I can't find a published package for that right now, so the team will quote it "
                              "for you.", links=links + self.contact_links(), chips=chips, confident=False)
        if event_type == "wedding":
            addons = [o for o in self.offerings("addon") if "wedding" in (o.suits_event_types or [])]
            text = (f"Here are our {count} wedding packages. Every price is a starting price, confirmed when we "
                    "plan the day with you.\n\nCraig plans each wedding personally with the couple: we host your "
                    "wedding and help you organize the day, not just the music.")
            if addons:
                text += " Wedding add-ons: " + _and(f"{o.name} ({_lower_first(o.price_display)})"
                                                    for o in addons) + "."
                top = rows[-1]
                included = set(top.incompatible_with) | set(top.included_addons)
                if all(a.slug in included for a in addons):
                    text += f" {top.name} includes " + ("both." if len(addons) == 2 else "them all.")
            chips = ["What does Signature Wedding include?", "Dancing on a Cloud", "Is my date available?",
                     "How booking works"]
        elif event_type == "foam":
            text = ("Every foam package includes the foam cannon, the foam solution and an operator; Foam Party "
                    "Live adds a DJ. Every price is a starting price.")
            dj = next((o for o in self.offerings("addon") if o.slug == "foam-dj"), None)
            if dj:
                text += (f"\n\nYou can add a professional DJ to Foam Blast or Foam Frenzy, "
                         f"{_lower_first(dj.price_display)}.")
            text += (" Expanded sound for large events, extra foam equipment for a big crowd, or travel outside our "
                     "standard service area is a custom quote.")
            chips = ["Is the foam safe?", "Foam party add-ons", "What's included in Foam Frenzy?",
                     "Foam indoors or outdoors?"]
        else:
            who = {"party": "Birthdays, private parties, corporate and school events share",
                   "corporate": "Corporate events use",
                   "school": "School and community events use"}[event_type]
            text = (f"{who} our {count} party packages. Every price is a starting price, and every package "
                    f"includes a professional DJ and emcee.\n\nThey're designed for most private events of up to "
                    f"150 guests. Large guest counts, several event areas, complex venues, holidays, extended "
                    f"travel or extra production may need a custom quote. Setup and teardown are included and "
                    f"don't reduce your entertainment time.")
            chips = ["Party add-ons", "Extra hours", "Foam party prices", "How booking works"]
            if event_type == "party" and T.any_phrase(q.norm, ["birthday", "birthdays"]):
                links.append(_page("birthday"))
        return Reply(kind="live", intent=f"prices.{event_type}", text=text, cards=cards, links=links, chips=chips)

    def _price_overview(self) -> Reply:
        cards = []
        for family, title in (("wedding", "Wedding packages"), ("party", "Party packages"),
                              ("foam", "Foam party packages")):
            rows = self.packages_for(family)
            if not rows:
                continue
            cheapest = rows[0]
            cards.append(Card(title=title, subtitle=cheapest.price_display,
                              detail=f"{len(rows)} packages, starting with {cheapest.name}",
                              path=PAGES[FAMILY_PAGE[family]][1]))
        return Reply(
            kind="live", intent="prices.overview",
            text=("Here's where our packages start. Every price is a starting price. Birthdays, private parties, "
                  "corporate and school events share the party packages.\n\nPlan My Event gives you a plan with a "
                  "price in about a minute, and nothing is booked until you say so."),
            cards=cards, links=[_page("plan"), _page("parties"), _page("weddings"), _page("foam")],
            chips=["Wedding prices", "Party prices", "Foam party prices", "Add-ons", "Extra hours"])

    def _package_details(self, slugs: list[str], q: Question, priced: bool) -> Reply:
        rows = {o.slug: o for o in self.offerings("package")}
        chosen = [rows[s] for s in slugs if s in rows]
        cards = [self.package_card(o) for o in chosen]
        parts = []
        for o in chosen[:2]:
            line = f"**{o.name}**"
            if o.tagline:
                line += f": {o.tagline}"
            features = o.feature_list
            if features:
                lead = o.includes_lead
                line += "\n\n" + (f"{lead} " if lead else "It includes: ") + \
                    "; ".join(_lower_first_word(f) for f in features) + "."
            if o.choice_display:
                line += f" {o.choice_display}."
            parts.append(line)
        text = "\n\n".join(parts) + "\n\nPrices are starting prices."
        family = self._family_of(chosen[0]) if chosen else None
        links = [FAMILY_PLAN.get(family, _page("plan")), _page(FAMILY_PAGE.get(family, "parties"))]
        return Reply(kind="live", intent="prices.package", text=text, cards=cards, links=links,
                     chips=["Add-ons", "Extra hours", "Compare packages", "How booking works"])

    def _addon_prices(self, slugs: list[str], q: Question) -> Reply:
        rows = {o.slug: o for o in self.offerings("addon")}
        chosen = [rows[s] for s in slugs if s in rows]
        cards = [self.addon_card(o, q.event_type) for o in chosen]
        notes = []
        if "karaoke-addon" in slugs:
            notes.append("Karaoke is added to a party package, so a DJ comes with it.")
        if "jukebox-bingo-private" in slugs:
            notes.append("Jukebox Bingo is priced on request and can be booked on its own.")
        if "dancing-on-a-cloud" in slugs:
            notes.append("Dancing on a Cloud is subject to venue approval.")
        if any(o.price_usd is None for o in chosen):
            notes.append("Where it says Price on request, the price comes with your quote.")
        if q.event_type == "wedding" and any(o.suits_event_types and "wedding" not in o.suits_event_types
                                             for o in chosen):
            notes.append("It isn't a wedding extra: the wedding add-ons are Dancing on a Cloud and LED Party "
                         "Sticks, and the wedding packages show what each includes.")
        text = "Here's the current price. " + " ".join(notes) if len(chosen) == 1 else \
            "Here are the current prices. " + " ".join(notes)
        return Reply(kind="live", intent="prices.addon", text=text.strip(), cards=cards,
                     links=[_page("plan"), _page("parties")], chips=["All add-ons", "Party prices",
                                                                     "How booking works"])

    def _addon_list(self, q: Question) -> Reply:
        event_type = q.event_type
        rows = [o for o in self.offerings("addon") if self.suits(o, event_type)] if event_type \
            else self.offerings("addon")
        cards = [self.addon_card(o, event_type) for o in rows]
        if event_type == "wedding":
            text = ("For weddings there are two add-ons, and Headliner Wedding Experience includes both. Dancing "
                    "on a Cloud is subject to venue approval. Craig talks any other extras through with each "
                    "couple himself.")
            if len(rows) != 2:
                text = ("These are the add-ons for a wedding. Craig talks any other extras through with each "
                        "couple himself.")
        elif event_type == "foam":
            text = ("These can be added to a foam party, at the same prices whatever the occasion. Most need a DJ: "
                    "Foam Party Live has one, or add one to Foam Blast or Foam Frenzy.")
        else:
            text = ("Our extras cost the same whatever the occasion. Most are added to a package with a DJ. Where "
                    "it says Price on request, the price comes with your quote. Headliner Party includes your "
                    "choice of two.")
        return Reply(kind="live", intent="prices.addons", text=text, cards=cards,
                     links=[_page("plan"), _page(FAMILY_PAGE.get(event_type, "parties"))],
                     chips=["Party prices", "Wedding prices", "Foam party prices"])

    def _extra_hours(self, q: Question) -> Reply:
        rows = [o for o in self.offerings("package") if o.is_hourly]
        if q.event_type in ("party", "corporate", "school", "foam"):
            rows = [o for o in rows if self.suits(o, q.event_type)]
        cards = [Card(title=o.name, subtitle=o.price_display, detail=o.hourly_display,
                      path=PAGES[FAMILY_PAGE.get(self._family_of(o), "parties")][1]) for o in rows]
        text = "Extra time is priced by the hour on the party and foam packages, as below."
        if q.event_type in (None, "wedding"):
            text += " Weddings have no published extra-hour price: more time at a wedding is quoted by Craig."
        return Reply(kind="live", intent="prices.extra_hours", text=text, cards=cards,
                     links=[_page("plan"), _page("contact")],
                     chips=["Party prices", "Foam party prices", "Wedding prices"])

    def _djs_now(self, now: datetime | None) -> Reply:
        # The website reads HPN's private shift rota (who is booked at which bar right now); the avatar
        # cannot, so it never says who is - or is not - on. HPN's own words for where to look.
        return Reply(
            kind="live", intent="djs.now",
            text=("I can't see which DJs are on right now from here. When a night starts, its DJ shows up on the "
                  "Request a song page, and the list follows the venue schedule.\n\nAt a private event? Scan the "
                  "QR code at the DJ booth: it works whenever the DJ is taking requests."),
            links=[_page("request"), _page("nights")],
            chips=["What's on tonight?", "Can I tip the DJ?", "How do song requests work?"])

    def _nights(self, q: Question, now: datetime | None) -> Reply:
        local = houston_now(now)
        today = local.date()
        c = q.concepts
        if "tonight" in c:
            when, days = "tonight", [today]
        elif "tomorrow" in c:
            when, days = "tomorrow", [today + timedelta(days=1)]
        elif q.weekday is not None:
            ahead = (q.weekday - today.weekday()) % 7
            when, days = "day", [today + timedelta(days=ahead)]
        elif "weekend" in c:
            days = [today + timedelta(days=i) for i in range(7) if (today + timedelta(days=i)).weekday() in (4, 5, 6)]
            days = [d for d in days if d <= today + timedelta(days=(6 - today.weekday()) % 7)]
            when = "weekend"
        else:
            when, days = "week", [today + timedelta(days=i) for i in range(7)]
        slots = sorted(self.kb.slots, key=lambda s: (s.start_time, s.venue.name))
        towns = {t.lower() for t in q.towns}
        if towns:
            slots = [s for s in slots if (s.venue.city or "").strip().lower() in towns]
        if q.kinds:
            slots = [s for s in slots if s.service in q.kinds]
        if q.venues:
            slots = [s for s in slots if s.venue.name in q.venues]

        def happening(slot, day):
            if day == today:
                upcoming = slot.next_date(today=today, now=local)
                return upcoming is not None and upcoming <= today
            return slot.runs_on(day)

        found = [(day, slot) for day in days for slot in slots if happening(slot, day)]
        found.sort(key=lambda pair: (pair[0], pair[1].start_time, pair[1].venue.name.lower()))
        what = self._kind_words(q.kinds)
        where = f" at {_and(sorted(q.venues))}" if q.venues else (f" in {_and(sorted(q.towns))}" if q.towns else "")
        label = {"tonight": "tonight", "tomorrow": "tomorrow", "weekend": "this weekend",
                 "week": "in the next seven days"}.get(when) or f"on {days[0]:%A}"
        intent = f"nights.{when}"
        links = [_page("nights")]
        if when == "tonight":
            links.append(_page("request"))
        if not found:
            week = [today + timedelta(days=i) for i in range(1, 8)]
            later = [(d, s) for d in week for s in slots if s.runs_on(d)]
            later.sort(key=lambda pair: (pair[0], pair[1].start_time))
            text = f"I can't see any {what}{where} {label}."
            if later and when != "week":
                text += " These are the next ones coming up."
                cards = [self._night_card(d, s, today) for d, s in later[:5]]
            else:
                cards = []
                if q.towns or q.venues:
                    text += " We don't have a public night there right now."
            text += "\n\nOur schedule (Nights out) has every night we play."
            return Reply(kind="live", intent=intent, text=text, cards=cards, links=links,
                         chips=["What's on tonight?", "This weekend", "Karaoke nights", "Jukebox Bingo nights"])
        shown = found[:8]
        cards = [self._night_card(d, s, today) for d, s in shown]
        count = len(found)
        text = f"**{count} {what if count != 1 else self._singular(what)}{where} {label}.**"
        if count > len(shown):
            text += f" Here are the first {len(shown)}; our schedule (Nights out) has them all."
        text += "\n\nNights sometimes move: if one matters to you, check with the venue before you travel."
        chips = ["What's on tomorrow?", "This weekend", "Request a song"] if when == "tonight" \
            else ["What's on tonight?", "This weekend", "Karaoke nights"]
        return Reply(kind="live", intent=intent, text=text, cards=cards, links=links, chips=chips)

    @staticmethod
    def _night_card(day, slot, today) -> Card:
        if day == today:
            when = "tonight"
        elif day == today + timedelta(days=1):
            when = "tomorrow"
        else:
            when = f"{day:%a} {day.day} {day:%b}"
        return Card(title=slot.venue.name, subtitle=f"{slot.display_name} · {clock(slot.start_time)}",
                    detail=" · ".join(p for p in (slot.venue.city, when) if p), path=PAGES["nights"][1])

    @staticmethod
    def _kind_words(kinds: list[str]) -> str:
        names = {"karaoke": "karaoke nights", "jukebox_bingo": "Jukebox Bingo nights", "dj": "DJ parties"}
        if not kinds:
            return "nights"
        return _and(names[k] for k in kinds)

    @staticmethod
    def _singular(what: str) -> str:
        return {"nights": "night", "karaoke nights": "karaoke night", "Jukebox Bingo nights": "Jukebox Bingo night",
                "DJ parties": "DJ party"}.get(what, what)

    def _towns(self, q: Question) -> Reply:
        served = _and(self.kb.town_names())
        if q.towns:
            text = (f"Yes, {_and(sorted(q.towns))} {'is one of the towns' if len(q.towns) == 1 else 'are among the towns'} "
                    f"we serve. We work across {served}.")
        else:
            text = f"We serve {served}. Further afield is usually possible: ask."
        text += ("\n\nTravel outside our standard service area may need a custom quote. Tell us where your event is "
                 "and we'll confirm.")
        return Reply(kind="live", intent="towns", text=text, links=[_page("plan"), _page("contact")],
                     chips=["What's on tonight?", "Prices and packages", "How booking works"])

    def _venue_contact(self, q: Question) -> Reply:
        b = self.business()
        venues = [v for v in self.kb.venues if v.name in q.venues]
        lines = [f"**{v.name}**: {v.address}." + (f" Phone {v.phone_display}." if v.phone else "") for v in venues]
        text = ("\n\n".join(lines) + f"\n\n{'That is the venue' if len(venues) == 1 else 'Those are the venues'}"
                f"'s own {'number' if len(venues) == 1 else 'numbers'}. For Houston Party Network, call "
                f"{b['phone_display']} or email {b['email']}.")
        links = [{"label": f"Call {v.name}", "path": f"tel:{v.phone}"} for v in venues if v.phone][:2]
        return Reply(kind="live", intent="venue.contact", text=text, links=links + [_page("nights")],
                     chips=["What's on tonight?", "This weekend", "Contact us"])

    def _contact(self) -> Reply:
        b = self.business()
        return Reply(
            kind="live", intent="contact",
            text=(f"You can call Houston Party Network on **{b['phone_display']}** or email **{b['email']}**, or "
                  f"send the contact form. We reply to every enquiry; if it's urgent, call.\n\nWe're based in "
                  f"Tomball and work across greater Houston."),
            links=self.contact_links(), chips=["Plan an event", "Prices and packages", "Weddings"])

    # --- the knowledge base -------------------------------------------------------------------------------
    def _build_index(self) -> dict:
        entries, df = [], {}
        for topic in self.kb.topics:
            phrases = [T.normalise(topic.title)] + [T.normalise(q) for q in topic.question_lines]
            phrases = [p for p in phrases if p]
            keywords = [k for k in (T.normalise(k) for k in topic.keyword_list) if k]
            entry = {"topic": topic, "phrases": [(p, bag(p)) for p in phrases], "keywords": keywords,
                     "concepts": set().union(*(T.concepts(T.normalise(p)) for p in
                                               [topic.title] + topic.question_lines + topic.keyword_list))}
            entries.append(entry)
            for token in set().union(*(b for _, b in entry["phrases"])):
                df[token] = df.get(token, 0) + 1
        return {"entries": entries, "df": df}

    def _weight(self, token: str) -> float:
        total = max(len(self._index["entries"]), 1)
        return 1 + math.log(total / (1 + self._index["df"].get(token, 0)) + 1)

    def score_topic(self, entry: dict, q: Question, qbag: set) -> float:
        best = 0.0
        for phrase, pbag in entry["phrases"]:
            if q.norm == phrase:
                best = max(best, 100)
                continue
            if len(phrase.split()) >= 4 and T.has_phrase(q.norm, phrase):
                best = max(best, self.kb.exact)
            if not pbag or not qbag:
                continue
            shared = sum(self._weight(t) for t in pbag & qbag)
            if shared:
                coverage = shared / sum(self._weight(t) for t in pbag)
                precision = shared / sum(self._weight(t) for t in qbag)
                best = max(best, 60 * 2 * coverage * precision / (coverage + precision))
        keyword_score = 0
        words = {_stem(w) for w in q.norm.split()}
        for keyword in entry["keywords"]:
            if " " in keyword:
                if T.has_phrase(q.norm, keyword):
                    keyword_score += 10
            elif _stem(keyword) in words:
                keyword_score += 6
        return best + min(keyword_score, 24)

    @staticmethod
    def exact_match(topic, q: Question) -> bool:
        for line in [topic.title] + topic.question_lines:
            phrase = T.normalise(line)
            if phrase and (q.norm == phrase or (len(phrase.split()) >= 4 and T.has_phrase(q.norm, phrase))):
                return True
        return False

    def scored_topics(self, q: Question) -> list:
        qbag = bag(q.norm)
        scored = [(self.score_topic(e, q, qbag), e["topic"]) for e in self._index["entries"]]
        scored = [(s, t) for s, t in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1].sort_order, pair[1].title))
        return scored

    def topic_concepts(self, topic) -> set:
        for e in self._index["entries"]:
            if e["topic"] is topic:
                return e["concepts"]
        return set()

    def topic_reply(self, topic) -> Reply:
        links = list(topic.links or []) or [_page("plan"), _page("contact")]
        return Reply(kind="answer", intent=f"topic:{topic.slug}", text=self.kb.fill(topic.answer_template),
                     links=links, chips=list(topic.chips or []), confident=True)

    def _build_vocabulary(self) -> set:
        words = set()
        for phrases in list(T.CONCEPTS.values()) + [T.GREETINGS, T.HOW_ARE_YOU, T.THANKS, T.BYE, T.WHO, T.HUMAN,
                                                     T.COMPLAINT, T.OFF_TOPIC, T.SPANISH, ACK]:
            for phrase in phrases:
                words.update(phrase.split())
        for topic in self.kb.topics:
            for text in [topic.title, topic.answer_template, " ".join(topic.keyword_list),
                         " ".join(topic.question_lines)]:
                words.update(T.normalise(text).split())
        return words

    # --- small talk and the exceptional cases --------------------------------------------------------------
    def _greeting(self, now, spanish: bool = False) -> Reply:
        hello = "¡Hola!" if spanish else good_time_of_day(now)
        return Reply(kind="greeting", intent="greeting",
                     text=(f"{hello} I'm {self.name}, Houston Party Network's AI Agent. What can I help "
                           f"you with today: planning an event, or a night out?"),
                     chips=STARTER_CHIPS, links=[_page("plan")])

    def _how_are_you(self, now, greeted: bool = False) -> Reply:
        hello = f"{good_time_of_day(now)} " if greeted else ""
        return Reply(kind="smalltalk", intent="smalltalk.how_are_you",
                     text=(f"{hello}I'm doing well, thanks for asking! Are you planning an event, or looking for a "
                           f"night out?"), chips=STARTER_CHIPS[:5], links=[_page("plan")])

    def _thanks(self) -> Reply:
        return Reply(kind="smalltalk", intent="smalltalk.thanks",
                     text="You're welcome! Is there anything else I can help with?",
                     chips=STARTER_CHIPS[:5], links=[_page("plan")])

    def _ack(self) -> Reply:
        return Reply(kind="smalltalk", intent="smalltalk.ack", text="Is there anything else I can help with?",
                     chips=STARTER_CHIPS[:5], links=[_page("plan")])

    def _bye(self, now) -> Reply:
        b = self.business()
        part = {"Good morning!": "morning", "Good afternoon!": "afternoon"}.get(good_time_of_day(now), "evening")
        return Reply(kind="smalltalk", intent="smalltalk.bye",
                     text=(f"Thanks for stopping by, and have a lovely {part}! If you need us, call "
                           f"{b['phone_display']} or email {b['email']}."),
                     links=[_page("plan")] + self.contact_links()[:2], chips=["Plan an event"])

    def _who(self) -> Reply:
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.who",
                     text=(f"I'm {self.name}, Houston Party Network's AI Agent, not a "
                           "person. I answer from our own prices, packages and venue schedule, and from answers our "
                           "team has written.\n\nFor anything I can't answer, a real person is a call or an email "
                           f"away: {b['phone_display']} or {b['email']}, or use the contact form."),
                     links=self.contact_links(), chips=STARTER_CHIPS[:4])

    def _human(self) -> Reply:
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.human",
                     text=(f"Of course. You can reach the Houston Party Network team directly: call "
                           f"**{b['phone_display']}** (usually the fastest) or email **{b['email']}**, or send the "
                           f"contact form. We reply to every enquiry."),
                     links=self.contact_links(), chips=["Plan an event"])

    def _person_contact(self) -> Reply:
        """Someone's own number (Craig's, a manager's, a DJ's): only the business's details are given out."""
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.person_contact",
                     text=("I can only give out Houston Party Network's own contact details. To reach Craig or the "
                           f"team, call **{b['phone_display']}** (usually the fastest) or email **{b['email']}**, or "
                           "send the contact form. We reply to every enquiry."),
                     links=self.contact_links(), chips=["Plan an event", "Prices and packages"])

    def _callback(self) -> Reply:
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.callback",
                     text=("I can't pass messages on from here, so a number or email left with me won't reach the "
                           "team in time. To get a call back, send the contact form with your number and a good time to "
                           f"call, or call us on **{b['phone_display']}** or email **{b['email']}**."),
                     links=[_page("contact", "Contact form")] + self.contact_links()[:2], chips=["Plan an event"])

    def _complaint(self) -> Reply:
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.complaint",
                     text=("I'm sorry to hear that. I'm only a virtual helper, so I can't look into it myself, but "
                           f"our team will want to know. Please call **{b['phone_display']}** or email "
                           f"**{b['email']}**, or send the contact form, and we'll get back to you."),
                     links=self.contact_links(), chips=[])

    def _profanity(self) -> Reply:
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.profanity",
                     text=("I'm here to help with Houston Party Network events and nights out. If something has gone "
                           f"wrong, our team would like to hear about it: call {b['phone_display']} or email "
                           f"{b['email']}.\n\nOr ask me about prices, packages or what's on tonight."),
                     links=self.contact_links(), chips=STARTER_CHIPS[:4])

    def _off_topic(self) -> Reply:
        return Reply(kind="smalltalk", intent="smalltalk.offtopic",
                     text=("Sorry, I can only help with Houston Party Network events and nights out: packages and "
                           "prices, what's on tonight, song requests and booking."),
                     chips=STARTER_CHIPS, links=[_page("plan")])

    def _empty(self) -> Reply:
        return Reply(kind="smalltalk", intent="smalltalk.empty",
                     text=("Ask me anything about Houston Party Network: prices and packages, what's on tonight, "
                           "song requests or booking. Or tap one of these."),
                     chips=STARTER_CHIPS, links=[_page("plan")])

    def _gibberish(self) -> Reply:
        return Reply(kind="smalltalk", intent="smalltalk.gibberish",
                     text="Sorry, I didn't catch that. Could you put it another way? Or tap one of these.",
                     chips=STARTER_CHIPS, links=[_page("contact")], confident=False)

    def _card_number(self) -> Reply:
        b = self.business()
        return Reply(kind="smalltalk", intent="smalltalk.card",
                     text=("Please don't share card numbers or other payment details here. I haven't kept that "
                           "number, and nothing is paid through me: our team arranges payment with you directly."
                           f"\n\nIf you need to talk about a payment, call {b['phone_display']} or email "
                           f"{b['email']}."),
                     links=self.contact_links(), chips=["How does payment work?"])

    def _availability(self, q: Question) -> Reply:
        wedding = q.event_type == "wedding"
        plan = FAMILY_PLAN["wedding"] if wedding else _page("plan")
        return Reply(kind="smalltalk", intent="smalltalk.availability",
                     text=("I can't see our calendar, so I can't promise a date, but we'll confirm your date for "
                           "you. Send it through "
                           + ("the wedding enquiry form" if wedding else "Plan My Event (a few questions, then a price)")
                           + " or the contact form, and the team checks it.\n\nNothing on this site can hold a date: "
                           "availability is only ever confirmed by us, in writing, and nothing is booked until you "
                           "say so."),
                     links=[plan] + self.contact_links(), chips=["How booking works", "Prices and packages"])

    def _policy(self, concept: str) -> Reply:
        b = self.business()
        words, then = POLICY_WORDS[concept]
        return Reply(kind="smalltalk", intent=f"smalltalk.{concept}",
                     text=(f"That's one for the team: I can't confirm {words} here. {then}\n\nCall "
                           f"{b['phone_display']} or email {b['email']}, or use the contact form."),
                     links=self.contact_links(), chips=["How booking works", "Prices and packages"], confident=False)

    def _fallback(self, topics: list) -> Reply:
        b = self.business()
        chips = [t.title for _, t in topics[:3]] or STARTER_CHIPS[:3]
        return Reply(kind="fallback", intent="fallback",
                     text=("I'm not sure about that one yet. These might help, or ask the team directly: call "
                           f"{b['phone_display']}, email {b['email']} or use the contact form."),
                     links=self.contact_links(), chips=chips, confident=False)
