"""What the HPN Event-helper tests start from - the same as HPN's own tests (backend/apps/helper/tests/base.py):
the real knowledge base and catalogue (knowledge/hpn), and HPN's small test schedule instead of the live one,
so "tonight" and "this weekend" are the same nights on every run.

THURSDAY_8PM is a fixed Houston evening: Thursday 29 October 2026, 8 p.m.
"""
from __future__ import annotations

import copy
import unittest
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path

from app.hpn.engine import Engine
from app.hpn.knowledge import HOUSTON, Knowledge, Slot, Venue

KNOWLEDGE = Path(__file__).resolve().parents[2] / "knowledge" / "hpn"
THURSDAY_8PM = datetime(2026, 10, 29, 20, 0, tzinfo=HOUSTON)
MORNING = datetime(2026, 10, 29, 9, 0, tzinfo=HOUSTON)
AFTERNOON = datetime(2026, 10, 29, 14, 0, tzinfo=HOUSTON)
PHONE, EMAIL = "(832) 295-0852", "Info@H-TownParty.com"


@lru_cache
def _real() -> Knowledge:
    return Knowledge.load(KNOWLEDGE)


def real_knowledge() -> Knowledge:
    """The live knowledge (a copy: tests may change it)."""
    return copy.deepcopy(_real())


def test_schedule() -> tuple[list[Venue], list[Slot]]:
    """HPN's: Thursday karaoke in Tomball at 9, Friday bingo in Spring at 7, Saturday karaoke in Houston at 9,
    Wednesday bingo in Montgomery at 7, Sunday DJ party in Houston at 8 (and Saturday karaoke in Tomball)."""
    venues, slots = [], []

    def venue(name, city, *nights):
        place = Venue(name=name, city=city, street="1 Main St")
        venues.append(place)
        for service, weekday, hour in nights:
            slots.append(Slot(venue=place, service=service, weekday=weekday, start_time=time(hour, 0)))
        return place

    venue("Molly's Pub Tomball", "Tomball", ("karaoke", 3, 21), ("karaoke", 5, 21))
    venue("Franklin's Tower", "Spring", ("jukebox_bingo", 4, 19))
    venue("Tony's Tavern", "Houston", ("karaoke", 5, 21), ("dj", 6, 20))
    venue("Mongo's", "Montgomery", ("jukebox_bingo", 2, 19))
    return venues, slots


def fixture_knowledge(**changes) -> Knowledge:
    real = real_knowledge()
    venues, slots = test_schedule()
    args = dict(business=real.business, topics=real.topics, offerings=real.offerings, venues=venues, slots=slots,
                pages=real.pages, starter_chips=real.starter_chips, djs=real.djs, confident=real.confident,
                exact=real.exact, sources=real.sources, excluded_categories=real.excluded_categories)
    args.update(changes)
    return Knowledge(**args)


class HelperTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = fixture_knowledge()
        self.engine = Engine(self.kb, "Emma")

    def rebuild(self, **changes) -> None:
        self.kb = fixture_knowledge(**changes)
        self.engine = Engine(self.kb, "Emma")

    def ask(self, text, last=None, now=THURSDAY_8PM) -> dict:
        return self.engine.ask(text, last=last, now=now).as_dict()

    def intent(self, text, last=None) -> str:
        return self.ask(text, last=last)["intent"]
