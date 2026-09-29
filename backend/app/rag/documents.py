"""Document parsing, cleaning and heading-aware chunking.

Supported: .pdf (pypdf, BSD-3), .docx (python-docx, MIT), .md/.markdown/.txt, .html/.htm (stdlib).
PyMuPDF is deliberately not used (AGPL-3.0).
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm"}


@dataclass
class Section:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    section: str
    page: int | None
    chunk_index: int


# ---------------------------------------------------------------- parsing
def parse(path: Path) -> tuple[str, list[Section]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf(path)
    if ext == ".docx":
        return _docx(path)
    if ext in (".html", ".htm"):
        return _html(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return _markdown(text, default_title=path.stem)


def _markdown(text: str, default_title: str) -> tuple[str, list[Section]]:
    title, sections, heading, buf = default_title, [], "", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                sections.append(Section("\n".join(buf), heading))
                buf = []
            heading = m.group(2).strip()
            if m.group(1) == "#" and title == default_title:
                title = heading
        else:
            buf.append(line)
    if buf:
        sections.append(Section("\n".join(buf), heading))
    return title, sections


def _pdf(path: Path) -> tuple[str, list[Section]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or path.stem
    sections = [Section(page.extract_text() or "", "", i + 1) for i, page in enumerate(reader.pages)]
    return title, sections


def _docx(path: Path) -> tuple[str, list[Section]]:
    import docx

    d = docx.Document(str(path))
    title = d.core_properties.title or path.stem
    sections, heading, buf = [], "", []
    for p in d.paragraphs:
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading") or style == "title":
            if buf:
                sections.append(Section("\n".join(buf), heading))
                buf = []
            heading = p.text.strip()
            if style == "title":
                title = heading
        elif p.text.strip():
            buf.append(p.text)
    for table in d.tables:  # tables → "cell | cell" lines
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        buf.append("\n".join(rows))
    if buf:
        sections.append(Section("\n".join(buf), heading))
    return title, sections


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts, self.skip, self.title = [], 0, ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("h1", "h2", "h3"):
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        if tag in ("p", "li", "br", "div", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer"):
            self.skip = max(0, self.skip - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def _html(path: Path) -> tuple[str, list[Section]]:
    p = _HTMLText()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))
    return _markdown(html.unescape("".join(p.parts)), default_title=p.title.strip() or path.stem)


# ---------------------------------------------------------------- cleaning + chunking
def clean(text: str) -> str:
    text = text.replace("­", "")                      # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # hyphenated line breaks (PDF)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _words(text: str) -> list[str]:
    return text.split()


def chunk_document(path: Path, root: Path, max_words: int = 220, overlap_words: int = 30) -> list[Chunk]:
    """Chunks stay inside one section; long sections are split on paragraph/sentence
    boundaries with a small overlap. ~220 words ≈ 300 tokens."""
    title, sections = parse(path)
    rel = str(path.relative_to(root)).replace("\\", "/")
    chunks: list[Chunk] = []
    for sec in sections:
        text = clean(sec.text)
        if not text:
            continue
        units = [u for p in re.split(r"\n\s*\n", text) for u in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", p) if u.strip()]
        buf: list[str] = []
        for unit in units:
            if buf and len(_words(" ".join(buf + [unit]))) > max_words:
                chunks.append(_make(rel, title, sec, " ".join(buf), len(chunks)))
                tail = _words(" ".join(buf))[-overlap_words:]
                buf = [" ".join(tail)] if overlap_words else []
            buf.append(unit.strip())
        if buf:
            chunks.append(_make(rel, title, sec, " ".join(buf), len(chunks)))
    return chunks


def _make(source: str, title: str, sec: Section, text: str, idx: int) -> Chunk:
    body = f"{sec.heading}: {text}" if sec.heading and not text.startswith(sec.heading) else text
    cid = hashlib.sha1(f"{source}|{idx}|{body}".encode()).hexdigest()[:16]
    return Chunk(cid, body, source, title, sec.heading, sec.page, idx)
