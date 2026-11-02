"""Houston Party Network's facts, as the website's Event helper uses them, read from knowledge/hpn/.

  * event-helper-knowledge.json - HPN's export of its live knowledge base (scripts/hpn_refresh.py copies it
    from the HPN project): the topics with Craig's exact answers, the business details, the catalogue.
  * catalogue.json - the packages and add-ons with the live site's prices laid over the export's.
  * schedule.json - every public night (karaoke, Jukebox Bingo, DJ party) from /night-out/.
  * avatar_overrides.json - the few topics whose wording describes the website's text helper and is not
    true of the avatar, rewritten for it, and the categories left out (the DJs' team app).

The rules below are HPN's own (apps/venues/models.py VenueSlot, apps/venues/counts.py,
apps/services/models.py Offering), rewritten without Django so the answers match the website's.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

HOUSTON = ZoneInfo("America/Chicago")
SITE = "https://h-townparty.com"
# HPN apps/venues/models.py: a night with no end time is treated as running this long.
ASSUMED_RUN_HOURS = 3
SERVICE_NAMES = {"karaoke": "Karaoke", "jukebox_bingo": "Jukebox Bingo", "dj": "DJ Party"}
# HPN apps/venues/counts.py SERVED_TOWNS: where HPN travels for private events, whether or not a bar
# there has a night on; towns of later venues are added after these, alphabetically.
SERVED_TOWNS = ("Houston", "Tomball", "Spring", "The Woodlands", "Conroe", "Magnolia", "Waller", "Montgomery")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def number_word(n: int) -> str:
    """HPN services.models.number_word: "two" for 2; digits from eleven up."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def towns_sentence(names: list[str]) -> str:
    if len(names) < 2:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + " and " + names[-1]


# --- topics -----------------------------------------------------------------------------------------

@dataclass
class Topic:
    slug: str
    title: str
    category: str
    answer_template: str
    question_lines: list[str]
    keyword_list: list[str]
    links: list[dict]
    chips: list[str]
    sort_order: int = 0
    updated_at: str = ""
    overridden: bool = False      # avatar_overrides.json replaced the website helper's wording


# --- the catalogue -----------------------------------------------------------------------------------

@dataclass
class Offering:
    slug: str
    kind: str                     # "package" | "addon"
    name: str
    price_usd: Decimal | None
    price_display: str
    hourly_display: str = ""
    badge: str = ""
    tagline: str = ""
    summary: str = ""
    features: list[str] = field(default_factory=list)
    suits_event_types: list[str] = field(default_factory=list)
    builds_on: str | None = None
    includes_lead_text: str = ""
    choice_count: int = 0
    choice_options: list[str] = field(default_factory=list)
    incompatible_with: list[str] = field(default_factory=list)
    included_addons: list[str] = field(default_factory=list)
    hourly_rates: list[dict] = field(default_factory=list)
    sort_order: int = 0
    catalogue: Knowledge | None = field(default=None, repr=False, compare=False)

    @property
    def feature_list(self) -> list[str]:
        return [f.strip(" \t-•") for f in self.features if f.strip(" \t-•")]

    @property
    def is_hourly(self) -> bool:
        return bool(self.hourly_rates) or bool(self.hourly_display)

    @property
    def includes_lead(self) -> str:
        """ "Everything in Reception Essentials, plus:" (the live catalogue's own words when it has them)."""
        if self.includes_lead_text:
            return self.includes_lead_text
        base = self.catalogue.offering(self.builds_on) if (self.catalogue and self.builds_on) else None
        return f"Everything in {base.name}, plus:" if base else ""

    @property
    def choice_display(self) -> str:
        """ "Choose two: Photo Booth, ..." - Craig's words - or "" when there is no choice."""
        if not self.choice_count or not self.catalogue:
            return ""
        rows = [self.catalogue.offering(s) for s in self.choice_options]
        rows = sorted((o for o in rows if o is not None), key=lambda o: (o.sort_order, o.name))
        if not rows:
            return ""
        return f"Choose {number_word(self.choice_count)}: " + ", ".join(o.name for o in rows)


# --- the venue schedule ------------------------------------------------------------------------------

@dataclass
class Venue:
    name: str
    city: str = ""
    street: str = ""
    postcode: str = ""
    phone: str = ""                # E.164, as the venue's card on /night-out/ publishes it
    region: str = ""

    @property
    def address(self) -> str:
        """ "15945 Kuykendahl Rd, Houston, TX 77068", as the venue's card writes it."""
        state = " ".join(p for p in (self.region, self.postcode) if p)
        return ", ".join(p for p in (self.street, self.city, state) if p)

    @property
    def phone_display(self) -> str:
        """ "+12815806167" -> "(281) 580-6167" (a US number; anything else as published)."""
        digits = "".join(ch for ch in self.phone if ch.isdigit())
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}" if len(digits) == 10 else self.phone


@dataclass
class Slot:
    """HPN's VenueSlot: one recurring night at a venue."""
    venue: Venue
    service: str                  # karaoke | jukebox_bingo | dj
    weekday: int                  # 0 = Monday
    start_time: time
    end_time: time | None = None
    cadence: str = "weekly"       # weekly | fortnightly | monthly
    first_occurrence: date | None = None
    label: str = ""

    @property
    def display_name(self) -> str:
        return (self.label or "").strip() or SERVICE_NAMES.get(self.service, self.service)

    def occurrence_end(self, start: datetime) -> datetime:
        if self.end_time is None:
            return start + timedelta(hours=ASSUMED_RUN_HOURS)
        end = datetime.combine(start.date(), self.end_time, tzinfo=start.tzinfo)
        if end <= start:
            end += timedelta(days=1)
        return end

    def runs_on(self, day: date) -> bool:
        if day.weekday() != self.weekday:
            return False
        first = self.first_occurrence
        if first and day < first:
            return False
        if self.cadence == "fortnightly":
            return bool(first) and (day - first).days % 14 == 0
        if self.cadence == "monthly":
            return bool(first) and same_weekday_of_month(first, day)
        return True

    def next_date(self, today: date | None = None, now: datetime | None = None) -> date | None:
        now = now or datetime.now(HOUSTON)
        today = today or now.date()
        first = self.first_occurrence
        if today == now.date() and now.tzinfo is not None:
            yesterday = today - timedelta(days=1)
            if self.runs_on(yesterday):
                start = datetime.combine(yesterday, self.start_time, tzinfo=now.tzinfo)
                if self.occurrence_end(start) > now:
                    return yesterday
        candidate = today + timedelta(days=(self.weekday - today.weekday()) % 7)
        if candidate == now.date() and now.tzinfo is not None:
            start = datetime.combine(candidate, self.start_time, tzinfo=now.tzinfo)
            if self.occurrence_end(start) < now:
                candidate += timedelta(days=7)
        if self.cadence == "fortnightly":
            if not first:
                return None
            if (candidate - first).days % 14:
                candidate += timedelta(days=7)
        if self.cadence == "monthly":
            if not first:
                return None
            if candidate >= first:
                for _ in range(6):
                    if self.runs_on(candidate):
                        break
                    candidate += timedelta(days=7)
        if first and candidate < first:
            return first
        return candidate


def same_weekday_of_month(first: date, day: date) -> bool:
    """HPN apps/venues/recurrence.py: the same weekday of its month as `first` (the 2nd Saturday...)."""
    if day.weekday() != first.weekday():
        return False
    n = (first.day - 1) // 7 + 1
    if n == 5:
        return (day + timedelta(days=7)).month != day.month
    return (day.day - 1) // 7 + 1 == n


def clock(value: time) -> str:
    """HPN venue_tags._clock: 21:00 -> "9 PM", 20:30 -> "8:30 PM"."""
    hour = value.hour % 12 or 12
    minutes = f":{value.minute:02d}" if value.minute else ""
    return f"{hour}{minutes} {'AM' if value.hour < 12 else 'PM'}"


# --- everything together -----------------------------------------------------------------------------

class Knowledge:
    def __init__(self, *, business: dict, topics: list[Topic], offerings: list[Offering], venues: list[Venue],
                 slots: list[Slot], pages: dict, starter_chips: list[str], djs: int | None,
                 confident: int = 40, exact: int = 80, sources: dict | None = None,
                 excluded_categories: tuple[str, ...] = ()) -> None:
        self.business = business
        self.topics = sorted(topics, key=lambda t: (t.sort_order, t.title))
        self.offerings = offerings
        for o in offerings:
            o.catalogue = self
        self._by_slug = {o.slug: o for o in offerings}
        self.venues = venues
        self.slots = slots
        self.pages = pages
        self.starter_chips = starter_chips
        self.djs = djs
        self.confident = confident
        self.exact = exact
        self.sources = sources or {}
        self.excluded_categories = excluded_categories
        self.counts = self._counts()

    def offering(self, slug: str | None) -> Offering | None:
        return self._by_slug.get(slug) if slug else None

    def town_names(self) -> list[str]:
        """HPN counts.town_names: the standing list, plus any town a venue with a night is in."""
        towns = {v.city.strip() for v in self.venues if v.city.strip() and any(s.venue is v for s in self.slots)}
        known = {t.lower() for t in SERVED_TOWNS}
        return list(SERVED_TOWNS) + sorted(t for t in towns if t.lower() not in known)

    def _counts(self) -> dict:
        """HPN counts.schedule_counts: the numbers the answers' {placeholders} stand for."""
        def towns(rows):
            return len({s.venue.city.strip().lower() for s in rows if s.venue.city.strip()})

        karaoke = [s for s in self.slots if s.service == "karaoke"]
        bingo = [s for s in self.slots if s.service == "jukebox_bingo"]
        weekly_karaoke = [s for s in karaoke if s.cadence == "weekly"]
        counts = {
            "venues": len({id(s.venue) for s in self.slots}),
            "towns": towns(self.slots),
            "regular_events": len(self.slots),
            "karaoke_events": len(weekly_karaoke),
            "karaoke_towns": towns(weekly_karaoke),
            "bingo_events": len(bingo),
            "bingo_towns": towns(bingo),
            "town_list": towns_sentence(self.town_names()),
        }
        if self.djs is not None:
            counts["djs"] = self.djs
        return counts

    def fill(self, text: str) -> str:
        """{phone}, {email} and the schedule's tokens filled in; unknown braces left as they are."""
        text = text.replace("{phone}", self.business["phone_display"]).replace("{email}", self.business["email"])
        return re.sub(r"\{([a-z_]+)\}", lambda m: str(self.counts.get(m.group(1), m.group(0))), text)

    def topic(self, slug: str) -> Topic | None:
        return next((t for t in self.topics if t.slug == slug), None)

    # --- loading -----------------------------------------------------------------------------------
    @classmethod
    def load(cls, folder: Path) -> Knowledge:
        export = json.loads((folder / "event-helper-knowledge.json").read_text(encoding="utf-8"))
        overrides_path = folder / "avatar_overrides.json"
        overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
        excluded = tuple(overrides.get("exclude_categories", []))
        rewritten = overrides.get("topics", {})
        topics = []
        for t in export["topics"]:
            if not t.get("is_active", True) or t["category"] in excluded:
                continue
            new = rewritten.get(t["slug"], {})
            topics.append(Topic(
                slug=t["slug"], title=t["title"], category=t["category"],
                answer_template=new.get("answer", t.get("answer_template") or t["answer"]),
                question_lines=[q.strip() for q in t.get("example_questions", []) if q.strip()],
                keyword_list=[k.strip() for k in t.get("keywords", []) if k.strip()],
                links=[{"label": link["label"], "path": link["path"]} for link in new.get("links", t.get("links", []))],
                chips=new.get("chips", t.get("chips", [])),
                sort_order=t.get("sort_order", 0), updated_at=t.get("updated_at", ""), overridden=bool(new)))

        cat_path = folder / "catalogue.json"
        items = json.loads(cat_path.read_text(encoding="utf-8"))["items"] if cat_path.exists() else export["catalogue"]
        offerings = []
        for o in items:
            price = o.get("price_usd")
            offerings.append(Offering(
                slug=o["slug"], kind=o["kind"], name=o["name"],
                price_usd=Decimal(str(price)) if price not in (None, "") else None,
                price_display=o["price_display"], hourly_display=o.get("hourly_display", ""),
                badge=o.get("badge", ""), tagline=o.get("tagline", ""), summary=o.get("summary", ""),
                features=list(o.get("features", [])), suits_event_types=list(o.get("suits_event_types", [])),
                builds_on=o.get("builds_on"), includes_lead_text=o.get("includes_lead", ""),
                choice_count=o.get("choice_count", 0) or 0, choice_options=list(o.get("choice_options", [])),
                incompatible_with=list(o.get("incompatible_with", [])),
                included_addons=list(o.get("included_addons", [])),
                hourly_rates=list(o.get("hourly_rates", [])), sort_order=o.get("sort_order", 0)))

        venues, slots = [], []
        sched_path = folder / "schedule.json"
        if sched_path.exists():
            sched = json.loads(sched_path.read_text(encoding="utf-8"))
            by_name = {}
            for v in sched["venues"]:
                venue = Venue(name=v["name"], city=v.get("city", ""), street=v.get("street", ""),
                              postcode=v.get("postcode", ""), phone=v.get("phone", ""), region=v.get("region", ""))
                by_name[venue.name] = venue
                venues.append(venue)
            for s in sched["slots"]:
                slots.append(Slot(
                    venue=by_name[s["venue"]], service=s["service"], weekday=int(s["weekday"]),
                    start_time=time.fromisoformat(s["start_time"]),
                    end_time=time.fromisoformat(s["end_time"]) if s.get("end_time") else None,
                    cadence=s.get("cadence", "weekly"),
                    first_occurrence=date.fromisoformat(s["anchor"]) if s.get("anchor") else None,
                    label=s.get("label", "")))

        settings = export.get("helper_settings", {})
        pages = {k: (v["label"], v["path"]) for k, v in settings.get("pages", {}).items()}
        djs = export.get("placeholders", {}).get("{djs}")
        sources = {"knowledge_exported_at": export.get("exported_at"),
                   "schedule_read_at": json.loads(sched_path.read_text(encoding="utf-8")).get("read_at")
                   if sched_path.exists() else None,
                   "catalogue_read_at": json.loads(cat_path.read_text(encoding="utf-8")).get("read_at")
                   if cat_path.exists() else None}
        return cls(business=dict(export["business"]), topics=topics, offerings=offerings, venues=venues,
                   slots=slots, pages=pages, starter_chips=list(settings.get("starter_suggestions", [])),
                   djs=int(djs) if djs not in (None, "") else None,
                   confident=settings.get("confident_score", 40), exact=settings.get("exact_score", 80),
                   sources=sources, excluded_categories=excluded)
