"""System prompt and message assembly.

Layout keeps the long, stable prefix first so llama-server's prompt cache can reuse it:
    [system (persona + rules + rolling summary)] [recent turns ...] [latest user msg + retrieved context]
"""
from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM_TEMPLATE = """You are {name}, a warm, natural-sounding AI avatar having a real-time spoken conversation with the user.

How to answer:
- This is live speech. Reply in 1 to 3 short sentences unless the user explicitly asks for detail.
- Start with a short opening sentence, so speech can begin quickly.
- Plain spoken English only: no lists, no markdown, no emojis, no URLs, no stage directions.
- If you are not sure about a fact, say so instead of guessing.

Every reply MUST start with a short tag [emotion|gesture|intensity] and then the words you will say, for example:
[friendly|small_nod|0.4]Sure! What would you like to talk about?
[empathetic|none|0.5]I'm sorry to hear that. Do you want to talk about it?

emotion: neutral, friendly, happy, empathetic, concerned, serious, thinking, excited.
gesture: none, small_nod, nod, head_tilt, shake, smile.
intensity: 0.0 to 1.0, keep it subtle (usually 0.2 to 0.6).
Never mention or explain the tag.{extra}"""

KNOWLEDGE_TEMPLATE = """{question}

KNOWLEDGE retrieved from the user's documents (numbered sources):
{knowledge}

Answer using this KNOWLEDGE when it is relevant to the question. If the question is about the documents and the KNOWLEDGE does not contain the answer, say that you don't have that information in the documents. Never invent facts. Don't read source numbers aloud."""


def system_prompt(name: str, summary: str | None = None, knowledge_enabled: bool = False) -> str:
    extra = ""
    if summary:
        extra += f"\n\nSummary of the earlier conversation:\n{summary}"
    return SYSTEM_TEMPLATE.format(name=name, extra=extra)


@dataclass
class ConversationMemory:
    """Recent turns verbatim + an optional rolling summary of older turns."""

    persona: str
    max_turns: int = 8
    knowledge_enabled: bool = False
    turns: list[dict] = field(default_factory=list)  # {"role": ..., "content": ...}
    summary: str | None = None
    archived: list[dict] = field(default_factory=list)  # turns dropped from the window, not yet summarised

    def add(self, user_text: str, assistant_text: str) -> None:
        self.turns.append({"role": "user", "content": user_text})
        self.turns.append({"role": "assistant", "content": assistant_text})
        # Trim in blocks (drop the older half) so the cached prompt prefix changes rarely.
        if len(self.turns) > 2 * self.max_turns:
            cut = len(self.turns) - self.max_turns  # keep max_turns/2 exchanges
            cut -= cut % 2
            self.archived.extend(self.turns[:cut])
            del self.turns[:cut]

    def build(self, user_text: str, knowledge: str | None = None) -> list[dict]:
        content = user_text
        if knowledge:
            content = f"{user_text}\n\nKNOWLEDGE (retrieved from the user's documents):\n{knowledge}"
        return [
            {"role": "system", "content": system_prompt(self.persona, self.summary, self.knowledge_enabled)},
            *self.turns,
            {"role": "user", "content": content},
        ]

    def needs_summary(self) -> bool:
        return len(self.archived) >= 4

    def summary_request(self) -> list[dict]:
        transcript = "\n".join(f"{t['role']}: {_strip_meta(t['content'])}" for t in self.archived)
        prior = f"Existing summary: {self.summary}\n\n" if self.summary else ""
        return [
            {"role": "system", "content": "You write compact conversation summaries. Keep names, facts, preferences and open questions. Max 80 words. No metadata lines."},
            {"role": "user", "content": f"{prior}Conversation to add:\n{transcript}\n\nUpdated summary:"},
        ]

    def apply_summary(self, text: str) -> None:
        self.summary = text.strip()
        self.archived.clear()


def _strip_meta(text: str) -> str:
    if text.startswith("["):
        end = text.find("]")
        return text[end + 1:] if 0 < end < 80 else text
    return text
