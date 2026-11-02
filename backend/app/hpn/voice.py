"""An Event-helper Reply as the avatar says it.

The page shows the reply as the website's helper does (bold, paragraphs, cards, buttons); the voice says
the same text, with the cards' facts read out where the website shows them under the first paragraph (a
price list, tonight's nights) - a spoken answer has no cards to look at. Nothing is added that is not in
the reply; buttons and their addresses are not read. Phone numbers, prices and times are said by the
speech rules (behavior/speech_text.normalize_for_speech) right before synthesis.
"""
from __future__ import annotations

import re

MAX_SPOKEN_CARDS = 6
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_BADGE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?:;," else text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:1].isupper() and text[1:2].islower() else text


_DAYS = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday",
         "Sat": "Saturday", "Sun": "Sunday"}
_MONTHS = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May", "Jun": "June",
           "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October", "Nov": "November",
           "Dec": "December"}
_SHORT_DATE = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")


def _say_date(text: str) -> str:
    """The card's "Fri 6 Nov" as it is said: "Friday, November 6"."""
    return _SHORT_DATE.sub(lambda m: f"{_DAYS[m[1]]}, {_MONTHS[m[3]]} {int(m[2])}", text)


def card_sentence(card: dict, named: bool = True) -> str:
    """One card as a sentence. ``named`` False: the text just said its name, so only the facts."""
    title = card.get("title", "").strip()
    subtitle = card.get("subtitle", "").strip()
    detail = card.get("detail", "").strip()
    m = _BADGE.match(title)
    if m:                                               # "Signature Wedding (Most Popular)"
        title = f"{m[1]}, {m[2].lower()}"
    if card.get("path") == "/night-out/" and " · " in subtitle:
        kind, _, at = subtitle.partition(" · ")         # "Karaoke · 9 PM", detail "Houston · tonight"
        where = ", ".join(_say_date(p) for p in detail.split(" · ") if p)
        return _sentence(f"{title}: {kind} at {at}" + (f", {where}" if where else ""))
    parts = [_lower_first(subtitle)] if subtitle else []
    if detail.lower().startswith("then "):              # a package sold by the hour: what each extra hour costs
        parts.append(detail)
    if not named and parts:
        facts = ", ".join(parts)
        return _sentence(facts[:1].upper() + facts[1:])
    return _sentence(f"{title}: {', '.join(parts)}" if parts else title)


# A venue's address as it is said: the street words in full and without the state and ZIP code (on the screen
# with the rest): Kokoro said "Rd" as "R D", "TX" as "T X" and a ZIP code as a number in the thousands.
_STREET_WORDS = {"Rd": "Road", "Ln": "Lane", "St": "Street", "Dr": "Drive", "Ave": "Avenue", "Blvd": "Boulevard",
                 "Hwy": "Highway", "Pkwy": "Parkway", "Fwy": "Freeway", "Ct": "Court", "N": "North", "S": "South",
                 "E": "East", "W": "West"}
_STREET = re.compile(r"\b(" + "|".join(_STREET_WORDS) + r")\b(?!['’])")
_STATE_ZIP = re.compile(r",?\s*TX\b,?(\s*\d{5}(-\d{4})?)?")


def _say_address(text: str) -> str:
    return _STREET.sub(lambda m: _STREET_WORDS[m[1]], _STATE_ZIP.sub("", text))


def spoken(reply: dict) -> str:
    """The text the avatar says for a reply (Reply.as_dict())."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", reply.get("text", "")) if p.strip()]
    paragraphs = [_BOLD.sub(r"\1", p).replace("·", ",") for p in paragraphs]
    paragraphs = ["\n".join(_sentence(line) for line in p.split("\n")) for p in paragraphs]
    if reply.get("intent") == "venue.contact":
        paragraphs = [_say_address(p) for p in paragraphs]
    cards = reply.get("cards") or []
    if cards:
        first = paragraphs[0] if paragraphs else ""
        # a package's details begin with its name ("Headliner Party: A complete ..."): its one card then
        # says only the price, not the name again
        name = _BADGE.sub(r"\1", cards[0].get("title", "")).strip()
        named = not (len(cards) == 1 and name and first.startswith(name))
        said = [card_sentence(c, named=named) for c in cards[:MAX_SPOKEN_CARDS]]
        if len(cards) > MAX_SPOKEN_CARDS:
            said.append("The rest are on the screen.")
        paragraphs[1:1] = [" ".join(said)]
    return "\n\n".join(paragraphs)
