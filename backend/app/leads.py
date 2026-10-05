"""Leads (owner, 2026-10-05: "the avatar agent should lead generation"): after a customer's conversation
the LLM reads it and writes down what the customer said about themselves and what they need - only
what the customer said, never what the avatar suggested. A conversation with a name, an email or a
phone number, or a real need, becomes a lead in data/leads.jsonl on this PC (never committed: /data
is ignored). The owner sees them in the page (Leads) and downloads them as CSV.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
from pathlib import Path

FIELDS = ("name", "email", "phone", "company", "need", "timeline", "budget", "interest", "next_step", "notes")
INTEREST = ("low", "medium", "high")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
_MAX_FIELD = 300


def lead_request(transcript: str, company: str = "") -> list[dict]:
    """LLM request: the conversation -> one JSON object with the lead's fields."""
    business = f" for {company}" if company else ""
    return [
        {"role": "system", "content": (
            f"You read a conversation between a customer and a voice assistant{business} and write down the lead. "
            "Use ONLY what the customer said about themselves (the assistant's suggestions are not facts about "
            "the customer). Reply with one JSON object and nothing else, with these keys: "
            '"name" (first and last name if given), "email", "phone", "company" (the customer\'s company), '
            '"need" (what they want, one short sentence), "timeline", "budget", '
            '"interest" (low, medium or high: how interested they are in going ahead), '
            '"next_step" (what was agreed, e.g. "call back on Tuesday"), "notes" (anything else useful, short). '
            "Use an empty string for anything the customer did not say. Write emails as name@example.com "
            '(speech recognition writes "john at gmail dot com") and phone numbers as digits.')},
        {"role": "user", "content": f"Conversation:\n{transcript}\n\nJSON:"},
    ]


def parse_lead(text: str) -> dict | None:
    """The LLM's JSON as a clean lead dict (known fields only, strings, emails checked), or None."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    lead = {}
    for key in FIELDS:
        value = data.get(key, "")
        value = "" if value is None else re.sub(r"\s+", " ", str(value)).strip()[:_MAX_FIELD]
        if value.lower() in ("none", "null", "n/a", "unknown", "not provided", "not mentioned", "-"):
            value = ""
        lead[key] = value
    heard, lead["email"] = lead["email"], lead["email"].lower().replace(" ", "")
    if lead["email"] and not _EMAIL.match(lead["email"]):
        lead["notes"] = (lead["notes"] + f" (email as heard: {heard})").strip()
        lead["email"] = ""
    digits = re.sub(r"[^\d+]", "", lead["phone"])
    lead["phone"] = digits if len(re.sub(r"\D", "", digits)) >= 7 else ""
    lead["interest"] = lead["interest"].lower() if lead["interest"].lower() in INTEREST else ""
    return lead


_CONTACT = re.compile(r"@|\bat\b.*\bdot\b|\d[\d\s().-]{6,}\d|\bmy name\b|\bi'?m [A-Z][a-z]+|\bcall me\b|"
                      r"\bemail\b|\bphone\b|\bnumber\b", re.I)


def worth_reading(user_messages: list[str]) -> bool:
    """Whether a conversation can hold a lead: an LLM pass after every one-question chat would only
    load the GPU (the next conversation's first reply waits for it)."""
    real = [m for m in user_messages if not m.startswith("(")]
    return len(real) >= 2 or any(_CONTACT.search(m) for m in real)


def is_lead(lead: dict | None) -> bool:
    """Worth the owner's time: a way to reach the customer or their name, or a need they showed real
    interest in."""
    if not lead:
        return False
    return bool(lead["email"] or lead["phone"] or lead["name"]
                or (lead["need"] and lead["interest"] in ("medium", "high")))


class LeadStore:
    """data/leads.jsonl: one line per conversation that became a lead."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, lead: dict, session_id: str, visitor_id: str | None = None) -> dict:
        record = {"time": time.strftime("%Y-%m-%d %H:%M"), "session": session_id, "visitor": visitor_id or "", **lead}
        rows = [r for r in self.load() if r.get("session") != session_id]   # the newest version of a session
        rows.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return record

    def load(self) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows

    @staticmethod
    def to_csv(rows: list[dict]) -> str:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=("time", *FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()
