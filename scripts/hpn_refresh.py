"""Refresh the Houston Party Network knowledge the avatar answers from (knowledge/hpn/).

Three sources, all read-only (nothing is sent to HPN but plain page requests):

  1. The Event helper's knowledge base, exported from HPN's live database by the HPN project
     (docs/event-helper-knowledge/event-helper-knowledge.json in the HPN repository, E:\\HPN on the
     owner's PC): 111 topics with Craig's exact answers, the packages and add-ons, the contact details.
  2. The venue schedule from https://h-townparty.com/night-out/: every karaoke, Jukebox Bingo and DJ
     night (the page's schema.org Event data: venue, town, weekday, start time, every week / every
     other week and its date). Used for "what's on tonight?".
  3. The live packages and prices from https://h-townparty.com/api/plan/catalog/ (the planner's own
     catalogue, public). When it differs from the export, the live price wins and the difference is
     printed.

The checks at the end must all pass, or nothing is written: the avatar keeps answering from the last
good copy.

    .venv/bin/python scripts/hpn_refresh.py --export ~/projects/HPN/docs/event-helper-knowledge/event-helper-knowledge.json
    .venv/bin/python scripts/hpn_refresh.py --export E:/HPN/docs/event-helper-knowledge/event-helper-knowledge.json
    .venv/bin/python scripts/hpn_refresh.py --offline     # re-check the files already in knowledge/hpn/
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge" / "hpn"
SITE = "https://h-townparty.com"
HOUSTON = ZoneInfo("America/Chicago")
UA = "Mozilla/5.0 (HPN avatar knowledge refresh; read-only)"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# The night's kind as HPN's VenueSlot.service: its card and event name say "Open mic karaoke" for a
# karaoke night with a label (it still counts as karaoke).
SERVICE_BY_NAME = {"karaoke": "karaoke", "open mic karaoke": "karaoke", "jukebox bingo": "jukebox_bingo",
                   "dj party": "dj"}
CADENCE = {"P1W": "weekly", "P2W": "fortnightly", "P1M": "monthly"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def schedule_from_page(html: str) -> dict:
    """The /night-out/ page's schema.org Event list -> venues and their nights."""
    blocks = re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S)
    events = []
    for block in blocks:
        data = json.loads(block)
        if isinstance(data, list) and data and data[0].get("@type") == "Event":
            events = data
    if not events:
        raise SystemExit("no Event data on /night-out/ (the page changed?)")
    venues: dict[str, dict] = {}
    slots = []
    for ev in events:
        kind, _, place = ev["name"].partition(" at ")
        service = SERVICE_BY_NAME.get(kind.strip().lower())
        if service is None:
            raise SystemExit(f"unknown kind of night {kind!r} in {ev['name']!r}")
        loc = ev["location"]
        addr = loc.get("address", {})
        name = loc["name"]
        venues.setdefault(name, {
            "name": name,
            "city": (addr.get("addressLocality") or "").strip(),
            "street": addr.get("streetAddress", ""),
            "region": addr.get("addressRegion", ""),
            "postcode": addr.get("postalCode", ""),
            "phone": loc.get("telephone", ""),
        })
        sched = ev["eventSchedule"]
        day = sched["byDay"].rsplit("/", 1)[-1].lower()
        cadence = CADENCE.get(sched.get("repeatFrequency", "P1W"))
        if cadence is None:
            raise SystemExit(f"unknown repeat {sched.get('repeatFrequency')!r} for {ev['name']!r}")
        # The schedule's startDate is the night's next date: for every other week / once a month it
        # fixes which week (HPN's first_occurrence); a weekly night needs none.
        slots.append({
            "id": ev.get("@id", ""),
            "venue": name,
            "service": service,
            "label": kind.strip() if kind.strip().lower() not in ("karaoke", "jukebox bingo", "dj party") else "",
            "weekday": DAYS.index(day),
            "start_time": sched["startTime"],
            "end_time": (ev.get("endDate") or "")[11:16] or None,
            "cadence": cadence,
            "anchor": sched.get("startDate") if cadence != "weekly" else None,
            "next_date": ev.get("startDate", "")[:10],
        })
    return {"venues": sorted(venues.values(), key=lambda v: v["name"].lower()), "slots": slots}


def compare_catalogue(export: list[dict], live: dict) -> tuple[list[dict], list[str]]:
    """The export's catalogue with the live prices and wording laid over it; the differences."""
    live_rows = {o["slug"]: o for o in live.get("packages", []) + live.get("add_ons", [])}
    notes = []
    merged = []
    for item in export:
        row = dict(item)
        lv = live_rows.get(item["slug"])
        if lv is None:
            notes.append(f"{item['slug']}: in the export, not on the live site any more -> left out")
            continue
        for field, live_field in (("price_display", "price_display"), ("hourly_display", "hourly_display"),
                                  ("name", "name"), ("badge", "badge"), ("summary", "summary"),
                                  ("tagline", "tagline"), ("features", "features"),
                                  ("suits_event_types", "suits_event_types"), ("choice_count", "choice_count"),
                                  ("choice_options", "choice_options")):
            if live_field in lv and lv[live_field] != row.get(field):
                notes.append(f"{item['slug']}.{field}: export {row.get(field)!r} -> live {lv[live_field]!r}")
                row[field] = lv[live_field]
        live_price = lv.get("price")
        if (live_price or None) != (row.get("price_usd") or None):
            notes.append(f"{item['slug']}.price: export {row.get('price_usd')!r} -> live {live_price!r}")
            row["price_usd"] = live_price
        row["includes_lead"] = lv.get("includes_lead", "")
        row["included_addons"] = lv.get("included_addons", [])
        merged.append(row)
    for slug in set(live_rows) - {i["slug"] for i in export}:
        notes.append(f"{slug}: new on the live site, not in the export -> not used until the export has it")
    return merged, notes


def check(kb: dict, schedule: dict, catalogue: list[dict]) -> list[str]:
    """Problems that must stop the refresh."""
    problems = []
    topics = kb.get("topics", [])
    if len(topics) < 50:
        problems.append(f"only {len(topics)} topics in the export")
    for t in topics:
        if not t.get("answer", "").strip():
            problems.append(f"topic {t.get('slug')} has no answer")
        if re.search(r"\{[a-z_]+\}", t.get("answer", "")):
            problems.append(f"topic {t.get('slug')} has an unfilled placeholder")
        for link in t.get("links", []):
            if not link.get("path"):
                problems.append(f"topic {t.get('slug')} has a button without a page")
    if not schedule["slots"]:
        problems.append("the schedule has no nights")
    for s in schedule["slots"]:
        if s["cadence"] != "weekly" and not s["anchor"]:
            problems.append(f"{s['venue']}: an every-other-week night without its date")
    if len(catalogue) < 10:
        problems.append(f"only {len(catalogue)} packages and add-ons")
    b = kb.get("business", {})
    if not (b.get("phone_display") and b.get("email")):
        problems.append("no phone or email in the export")
    return problems


def summary(kb: dict, schedule: dict) -> list[str]:
    """The live counts against the export's placeholders (the answers' {venues}, {regular_events}...)."""
    ph = kb.get("placeholders", {})
    slots = schedule["slots"]
    live = {
        "{venues}": len({s["venue"] for s in slots}),
        "{regular_events}": len(slots),
        "{karaoke_events}": sum(s["service"] == "karaoke" for s in slots),
        "{bingo_events}": sum(s["service"] == "jukebox_bingo" for s in slots),
    }
    lines = []
    for key, value in live.items():
        was = ph.get(key)
        mark = "" if str(value) == str(was) else "   <- changed since the export (the live number is used)"
        lines.append(f"  {key:18} export {was!s:>4}   live {value:>4}{mark}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, help="HPN's event-helper-knowledge.json")
    ap.add_argument("--offline", action="store_true", help="only re-check the files in knowledge/hpn/")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.offline:
        kb = json.loads((OUT / "event-helper-knowledge.json").read_text(encoding="utf-8"))
        schedule = json.loads((OUT / "schedule.json").read_text(encoding="utf-8"))
        catalogue = json.loads((OUT / "catalogue.json").read_text(encoding="utf-8"))["items"]
        problems = check(kb, schedule, catalogue)
        print("\n".join(summary(kb, schedule)))
        print("problems:", problems or "none")
        sys.exit(1 if problems else 0)

    if args.export is None:
        default = Path.home() / "projects" / "HPN" / "docs" / "event-helper-knowledge" / "event-helper-knowledge.json"
        args.export = default
    kb = json.loads(args.export.expanduser().read_text(encoding="utf-8"))
    now = datetime.now(HOUSTON).isoformat(timespec="seconds")
    print(f"export: {args.export} (exported {kb.get('exported_at')})")

    schedule = schedule_from_page(get(f"{SITE}/night-out/").decode("utf-8"))
    schedule.update({"read_at": now, "source": f"{SITE}/night-out/ (schema.org Event data)"})
    live = json.loads(get(f"{SITE}/api/plan/catalog/"))
    catalogue, notes = compare_catalogue(kb["catalogue"], live)

    problems = check(kb, schedule, catalogue)
    print(f"schedule: {len(schedule['slots'])} nights at {len(schedule['venues'])} venues")
    print("\n".join(summary(kb, schedule)))
    print(f"catalogue: {len(catalogue)} items; differences from the export: {len(notes)}")
    for n in notes:
        print("  ", n)
    if problems:
        print("NOT WRITTEN, problems:", *problems, sep="\n  ")
        sys.exit(1)

    shutil.copyfile(args.export.expanduser(), OUT / "event-helper-knowledge.json")
    (OUT / "schedule.json").write_text(json.dumps(schedule, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "catalogue.json").write_text(json.dumps(
        {"read_at": now, "source": f"{SITE}/api/plan/catalog/ over the export's catalogue",
         "differences_from_export": notes, "items": catalogue}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
