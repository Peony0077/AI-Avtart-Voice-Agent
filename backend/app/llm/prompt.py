"""System prompt and message assembly.

Layout keeps the long, stable prefix first so llama-server's prompt cache can reuse it:
    [system (persona + rules + rolling summary)] [recent turns ...] [latest user msg + retrieved context]
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# Customers are in the United States and Canada (owner, 2026-10-05): a friendly North American
# customer-care voice - warm, upbeat, plain words, contractions - and their spelling, units and dates.
SYSTEM_TEMPLATE = """You are {name}, a warm and friendly AI assistant{company} having a real-time spoken conversation with a customer in the United States or Canada.

How to answer:
- This is live speech. Reply in 1 to 3 short sentences unless the user explicitly asks for detail.
- Start with a short opening sentence, so speech can begin quickly.
- Sound like a friendly, upbeat American customer-care professional: warm and positive, everyday words, contractions (I'm, you're, let's), now and then a natural phrase like "Absolutely!", "Great question!" or "Happy to help!", but don't overdo it.
- American English: American spelling, dollars, miles, pounds and Fahrenheit, dates like "October 5th"; if the customer is in Canada, use Canadian dollars, kilometers and Celsius.
- Plain spoken English only: no lists, no markdown, no emojis, no URLs, no stage directions.
- If you are not sure about a fact, say so instead of guessing.{lead}{tag}{extra}"""

# Lead generation (persona.lead_gen): a helpful salesperson, never pushy. One question at a time
# keeps the replies short enough for speech; repeating contact details back catches speech-recognition
# errors in names, emails and numbers.
LEAD_RULES = """

Your goal: help each customer and, when they are interested, turn that into a lead, like a great salesperson who is helpful and never pushy.
- Find out what they need: ask one short question at a time, and briefly check you understood ("So you're looking for ..., right?").
- Explain how we can help in simple words, with a concrete benefit for them.
- When they are interested, suggest the next step: {next_step}. Ask for their first name, then the best email or phone number to reach them, and repeat it back to confirm. Never ask for passwords, card numbers or other sensitive data.
- If they are not ready, that's fine: offer to answer more questions.
- If the customer has told you their first name, use it now and then. Never write a placeholder such as [First Name] or [Name].
- If you don't know a detail about the business, such as an exact price or availability, say a team member will follow up with it; never make it up.{business}"""

# Optional (llm.behavior_tag): the LLM states the avatar's emotion/gesture before its words. Off by
# default - writing it delayed every reply's first word by ~180 ms, and the avatar's behaviour is read
# from the words instead (behavior.manager.behavior_from_text).
TAG_RULES = """

Every reply MUST start with a short tag [emotion|gesture|intensity] and then the words you will say, for example:
[friendly|small_nod|0.4]Sure! What would you like to talk about?
[empathetic|none|0.5]I'm sorry to hear that. Do you want to talk about it?

emotion: neutral, friendly, happy, empathetic, concerned, serious, thinking, excited.
gesture: none, small_nod, nod, head_tilt, shake, smile.
intensity: 0.0 to 1.0, keep it subtle (usually 0.2 to 0.6).
Write the tag only once, at the very start of the reply. Never mention or explain the tag."""

KNOWLEDGE_TEMPLATE = """{question}

KNOWLEDGE retrieved from the user's documents (numbered sources):
{knowledge}

Answer using this KNOWLEDGE when it is relevant to the question. If the question is about the documents and the KNOWLEDGE does not contain the answer, say that you don't have that information in the documents. Never invent facts. Don't read source numbers aloud."""


# A request for a long answer. The system prompt asks for 1-3 sentences, and the 4B model then refused
# such requests ("I'm sorry, but I can't explain that in detail right now": 3 of 40 samples, and in
# user tests). Saying in the system prompt that detail is fine made *every* answer longer (simple
# questions 2 -> 3 sentences, up to 5); a hint on this turn only does neither (0 of 40 refused, detailed
# answers 7-8 sentences, simple ones still 2). Not stored in the history: the cached prefix stays.
_WANTS_DETAIL = re.compile(
    r"\b(in (more )?detail|detailed|details|explain|explanation|describe|elaborate|step[- ]by[- ]step|"
    r"instructions|walk me through|a (long|short|bedtime|funny)? ?story|tell me (a|another) story|tell me more|"
    r"more about|longer|how (does|do) [\w\s]{1,40} work)\b", re.I)
DETAIL_HINT = "\n\n(The user wants a full answer: give it in 4 to 8 spoken sentences.)"
# After the user stopped a reply, the 4B model often went back to it ("Now just say OK." -> "OK. Once upon a
# time ..." after a stopped story: 3 of 16 samples, and in the Stop browser test); a rule in the system prompt
# made it worse (4 of 12). A hint on the next message only: 0 of 16. Not stored in the history.
INTERRUPTED_HINT = "\n\n(The user stopped your previous answer. Reply only to this message.)"


def wants_detail(user_text: str) -> bool:
    return bool(_WANTS_DETAIL.search(user_text))


# Questions about the time or date get the current local time on that message only (the LLM does not
# know it; in the system prompt it would change the cached prefix every minute). Not "May I ..." or
# "every time": the month only with a date word or a number, "time" only when it is the clock's.
_ABOUT_TIME = re.compile(r"\b(what time|time is it|the time|current time|time now|time zone|date|day|today|tonight|"
                         r"tomorrow|yesterday|week|weekend|month|year|clock|hour|o'clock|morning|afternoon|evening|"
                         r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|"
                         r"june|july|august|september|october|november|december|"
                         r"(?:in|of|on|by|until|this|next|last|early|late|mid)[- ]may|may \d{1,2}(?:st|nd|rd|th)?)\b",
                         re.I)

# Chat-template control tokens (Qwen's ChatML and its other special tokens). llama-server parses special
# tokens in the rendered prompt, so a customer typing "<|im_end|><|im_start|>system ..." - or a document
# containing it - could write her instructions. Removed from everything that is not ours.
_CONTROL_TOKEN = re.compile(r"<\|[^<>|\s]{1,40}\|>|</?(?:think|tool_call|tool_response)>", re.I)
# Prompt size (llm.context): ~3.5 characters a token for English, and per message the template's own
# tokens. The lite profile's 3072-token context overflowed in long chats (llama-server then fails the turn).
CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 8
TRIM_TO = 0.75          # over the budget: older exchanges leave until they use this share of the room left


def strip_control_tokens(text: str) -> str:
    """``text`` without chat-template control tokens (also when removing one would form another)."""
    prev = None
    while prev != text:
        prev, text = text, _CONTROL_TOKEN.sub("", text)
    return text


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS


def time_hint(user_text: str, now: float | None = None, utc_offset_min: int | None = None) -> str:
    """The current date and time for a message that is about time or dates ('' otherwise): the
    customer's own (``utc_offset_min``: their browser's offset from UTC; this PC's time was wrong for
    customers in the US), written the American way ("Monday, October 5, 2026, 9:42 PM")."""
    if not _ABOUT_TIME.search(user_text):
        return ""
    now = time.time() if now is None else now
    t = time.gmtime(now + utc_offset_min * 60) if utc_offset_min is not None else time.localtime(now)
    hour = t.tm_hour % 12 or 12
    # Measured on the live model (18 time/date questions each): "(It is now ...)" -> 3x "just like you
    # said!"; "(Note for you, not said by the customer: ...)" -> 3 refusals; this wording: 18/18 right.
    return (f"\n\n(Current local time where the customer is: {time.strftime('%A, %B', t)} {t.tm_mday}, "
            f"{t.tm_year}, {hour}:{t.tm_min:02d} {'AM' if t.tm_hour < 12 else 'PM'}. You know this; say it naturally.)")


def lead_rules(company: str = "", offer: str = "", next_step: str = "") -> str:
    business = ""
    if company or offer:
        business = f"\n\nAbout the business{' ' + company if company else ''}: {offer or 'see above.'}"
    return LEAD_RULES.format(next_step=next_step or "a short call with our team",
                             business=business)


def system_prompt(name: str, summary: str | None = None, knowledge_enabled: bool = False,
                  behavior_tag: bool = False, personality: str = "", long_term: str | None = None,
                  company: str = "", lead: str = "") -> str:
    """``lead``: lead_rules(...) when the avatar generates leads (persona.lead_gen)."""
    extra = ""
    if personality:
        extra += f"\n\nYour personality, as your owner describes it: {personality}"
    if long_term:
        extra += ("\n\nWhat you remember about the user from earlier conversations (use it naturally when it "
                  f"fits, do not recite it): {long_term}")
    if summary:
        extra += f"\n\nSummary of the earlier conversation:\n{summary}"
    return SYSTEM_TEMPLATE.format(name=name, extra=extra, tag=TAG_RULES if behavior_tag else "",
                                  company=f" for {company}" if company else "", lead=lead)


@dataclass
class ConversationMemory:
    """Recent turns verbatim + an optional rolling summary of older turns."""

    persona: str
    max_turns: int = 8
    knowledge_enabled: bool = False
    behavior_tag: bool = False
    personality: str = ""              # persona (persona.py), set by the owner in the page
    long_term: str | None = None       # memory of the user from earlier sessions (persona.MemoryStore)
    after_interruption: bool = False   # the last reply was stopped: the next message gets INTERRUPTED_HINT
    turns: list[dict] = field(default_factory=list)  # {"role": ..., "content": ...}
    summary: str | None = None
    archived: list[dict] = field(default_factory=list)  # turns dropped from the window, not yet summarised
    said: list[str] = field(default_factory=list)       # all the user's messages (kept past the window)
    dialog: list[tuple[str, str]] = field(default_factory=list)   # (user, assistant), the whole session
    company: str = ""                  # the business she represents (persona.company)
    lead: str = ""                     # lead-generation rules (prompt.lead_rules) when lead_gen is on
    utc_offset_min: int | None = None  # the customer's time zone (their browser), for time questions
    context_tokens: int | None = None  # llm.context: the prompt must leave room for the reply in it
    reply_tokens: int = 320            # llm.max_tokens

    def add(self, user_text: str, assistant_text: str, interrupted: bool = False) -> None:
        user_text, assistant_text = strip_control_tokens(user_text), strip_control_tokens(assistant_text)
        self.after_interruption = interrupted
        self.said.append(user_text)
        self.dialog.append((user_text, assistant_text))
        self.turns.append({"role": "user", "content": user_text})
        self.turns.append({"role": "assistant", "content": assistant_text})
        # Trim in blocks (drop the older half) so the cached prompt prefix changes rarely.
        if len(self.turns) > 2 * self.max_turns:
            cut = len(self.turns) - self.max_turns  # keep max_turns/2 exchanges
            cut -= cut % 2
            self.archived.extend(self.turns[:cut])
            del self.turns[:cut]

    def build(self, user_text: str, knowledge: str | None = None) -> list[dict]:
        user_text = strip_control_tokens(user_text)
        content = (user_text + (DETAIL_HINT if wants_detail(user_text) else "")
                   + time_hint(user_text, utc_offset_min=self.utc_offset_min))
        if self.after_interruption:
            content += INTERRUPTED_HINT
        system = system_prompt(self.persona, strip_control_tokens(self.summary or "") or None, self.knowledge_enabled,
                               self.behavior_tag, self.personality, strip_control_tokens(self.long_term or "") or None,
                               self.company, self.lead)
        knowledge = strip_control_tokens(knowledge) if knowledge else None
        if (budget := self.prompt_budget()) is not None:
            fixed = estimate_tokens(system) + estimate_tokens(content)
            if knowledge:
                fixed += estimate_tokens(KNOWLEDGE_TEMPLATE) + estimate_tokens(knowledge)
            used = fixed + sum(estimate_tokens(t["content"]) for t in self.turns)
            if used > budget:
                # too long for the context: the oldest exchanges leave the window for the rolling summary
                # (as in add), in a block (TRIM_TO of the room the history has) so the cached prefix
                # changes rarely
                target = fixed + max(0, budget - fixed) * TRIM_TO
                while self.turns and used > target:
                    dropped = self.turns[:2]
                    del self.turns[:2]
                    self.archived.extend(dropped)
                    used -= sum(estimate_tokens(t["content"]) for t in dropped)
            if knowledge and used > budget:            # still too long: less of the retrieved knowledge
                knowledge = knowledge[:max(0, len(knowledge) - int((used - budget) * CHARS_PER_TOKEN))]
        if knowledge:   # with its rules: answer from it, never invent facts, no source numbers read aloud
            content = KNOWLEDGE_TEMPLATE.format(question=content, knowledge=knowledge)
        return [{"role": "system", "content": system}, *self.turns, {"role": "user", "content": content}]

    def prompt_budget(self) -> int | None:
        """Tokens the prompt may use: the context minus the reply's room (None: no limit known)."""
        if not self.context_tokens:
            return None
        return max(256, self.context_tokens - self.reply_tokens)

    def transcript(self) -> str:
        """This session's conversation as text (rolling summary of older turns + recent turns)."""
        lines = [f"(Earlier: {self.summary})"] if self.summary else []
        lines += [f"{t['role']}: {_strip_meta(t['content'])}" for t in self.archived + self.turns]
        return "\n".join(lines)

    def user_said(self) -> str:
        """What the user said in this session, one message per line (for the memory across sessions;
        placeholders such as the greeting's are left out). The whole session: turns that were
        summarised or trimmed from the LLM's window count too."""
        return "\n".join(f"- {s}" for s in self.said if not s.startswith("("))

    def lead_transcript(self) -> str:
        """The whole session for the lead (leads.py): what the customer said and what she answered."""
        lines = []
        for user, assistant in self.dialog:
            if not user.startswith("("):
                lines.append(f"Customer: {user}")
            lines.append(f"Assistant: {_strip_meta(assistant)}")
        return "\n".join(lines)

    def forget_user(self) -> None:
        """"Forget me": what the user said so far in this session is not remembered either."""
        self.said.clear()
        self.dialog.clear()

    @property
    def user_turns(self) -> int:
        return sum(1 for t in self.archived + self.turns if t["role"] == "user")

    def needs_summary(self) -> bool:
        return len(self.archived) >= 4

    def summary_count(self) -> int:
        """How many archived turns (oldest first, whole exchanges) the next summary request covers: all of
        them, or as many as fit in the context."""
        budget = self.prompt_budget()
        if budget is None:
            return len(self.archived)
        used, n = 400 + estimate_tokens(self.summary or ""), 0       # (instructions and the reply)
        while n < len(self.archived):
            pair = self.archived[n:n + 2]
            used += sum(estimate_tokens(t["content"]) for t in pair)
            if used > budget and n:
                break
            n += len(pair)
        return n

    def summary_request(self, count: int | None = None) -> list[dict]:
        """Merge the first ``count`` archived turns (default: all) into the summary (apply_summary)."""
        turns = self.archived if count is None else self.archived[:count]
        transcript = strip_control_tokens("\n".join(f"{t['role']}: {_strip_meta(t['content'])}" for t in turns))
        prior = f"Existing summary: {strip_control_tokens(self.summary)}\n\n" if self.summary else ""
        return [
            {"role": "system", "content": "You write compact conversation summaries. Keep the user's name, facts about "
             "them, preferences and open questions, and list the topics in the order they came up. Max 80 words. "
             "No metadata lines."},
            {"role": "user", "content": f"{prior}Conversation to add:\n{transcript}\n\nUpdated summary:"},
        ]

    def apply_summary(self, text: str, count: int | None = None) -> None:
        """The summary now covers the first ``count`` archived turns (default: all). Only those leave
        ``archived``: turns archived while the summary was being written are summarised next time
        (clearing all of them lost those)."""
        if not text.strip():
            return                         # an empty summary would lose the turns
        self.summary = text.strip()
        del self.archived[:len(self.archived) if count is None else count]


def _strip_meta(text: str) -> str:
    if text.startswith("["):
        end = text.find("]")
        return text[end + 1:] if 0 < end < 80 else text
    return text
