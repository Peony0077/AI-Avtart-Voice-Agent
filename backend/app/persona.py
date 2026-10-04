"""The avatar's persona and her memory of the user across sessions.

Like Anam's personas and HeyGen's LiveAvatar settings: the owner sets her name, personality, greeting
and voice in the page (stored in data/persona.json, applied to the next reply). With memory on (the
default; HeyGen's LiveAvatar keeps "a single accumulated summary" of earlier sessions too) a short
summary of what the user said about themselves is kept in data/memory.json, on this PC only, and
given to the LLM in the next session; the page's "Forget me" erases it.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from loguru import logger

MAX_NAME = 40
MAX_PERSONALITY = 600
MAX_GREETING = 200
MAX_MEMORY_WORDS = 90


@dataclass
class Persona:
    name: str
    personality: str = ""
    greeting: str = ""
    voice: str = ""
    memory: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def greeting_text(self) -> str:
        return self.greeting.strip() or f"Hi, I'm {self.name}! How can I help you today?"


class PersonaStore:
    def __init__(self, path: Path, defaults: Persona) -> None:
        self.path = path
        self.defaults = defaults

    def load(self) -> Persona:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self.defaults
        except (OSError, ValueError) as exc:
            logger.warning(f"persona file unreadable ({exc!r}): using the defaults")
            return self.defaults
        return self.merge(self.defaults, data, voices=None)

    def save(self, persona: Persona) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(persona.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def merge(current: Persona, data: dict, voices: list[str] | None) -> Persona:
        """``current`` updated with the fields of ``data`` that are valid (lengths capped, control
        characters removed; a voice only if it is one of ``voices``, when given)."""
        def clean(value, limit: int) -> str:
            text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", str(value)).strip()
            return text[:limit]

        changes: dict = {}
        if "name" in data and clean(data["name"], MAX_NAME):
            changes["name"] = clean(data["name"], MAX_NAME)
        if "personality" in data:
            changes["personality"] = clean(data["personality"], MAX_PERSONALITY)
        if "greeting" in data:
            changes["greeting"] = clean(data["greeting"], MAX_GREETING)
        if "voice" in data and (voices is None or data["voice"] in voices):
            changes["voice"] = str(data["voice"])
        if "memory" in data:
            changes["memory"] = bool(data["memory"])
        return replace(current, **changes)


class MemoryStore:
    """A short summary of what the user told the avatar in earlier sessions (data/memory.json)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.generation = 0      # +1 on every "Forget me"; an update begun before it is not saved

    def load(self) -> str | None:
        try:
            text = json.loads(self.path.read_text(encoding="utf-8")).get("summary", "")
        except (OSError, ValueError):
            return None
        return text.strip() or None

    def save(self, summary: str, generation: int | None = None) -> bool:
        """False (nothing written) if the memory was erased since ``generation``: the update of a session
        that ended just before "Forget me" would otherwise bring the erased memory back."""
        if generation is not None and generation != self.generation:
            return False
        words = summary.split()
        if len(words) > MAX_MEMORY_WORDS:
            summary = " ".join(words[:MAX_MEMORY_WORDS])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"summary": summary.strip(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
        return True

    def clear(self) -> None:
        self.generation += 1
        self.path.unlink(missing_ok=True)


def memory_request(earlier: str | None, user_said: str) -> list[dict]:
    """LLM request: the earlier memory merged with what the user said in the latest conversation.

    Only the user's own words go in: given the whole conversation the 4B model also "remembered" what
    the avatar had said ("likes hearty trail-side meals" from a cooking persona's replies) and the date,
    and it guessed the user's gender from the name ("He")."""
    return [
        {"role": "system", "content": (
            "You keep a short memory about a user for a voice assistant. Merge the earlier memory with what the "
            "user said in the latest conversation. Keep only what the user actually said about themselves: their "
            "name, facts (work, family, places, plans, pets), preferences and the topics they care about. Do not "
            "add anything they did not say, no dates or times, no small talk. Refer to the user by name, or as "
            "'the user'; never guess their gender (no he or she). "
            f"At most {MAX_MEMORY_WORDS - 10} words, plain sentences, no lists. If there is nothing new worth "
            "keeping, repeat the earlier memory.")},
        {"role": "user", "content": f"Earlier memory: {earlier or 'none'}\n\nWhat the user said in the latest "
                                    f"conversation:\n{user_said}\n\nUpdated memory:"},
    ]
