from pathlib import Path

import pytest

from app.rag.documents import chunk_document, clean, parse

KNOW = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "knowledge"


def test_markdown_sections_and_title(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n## Hours\nOpen 9 to 5.\n\n## Menu\nBread.\n", encoding="utf-8")
    title, sections = parse(f)
    assert title == "Title"
    assert [s.heading for s in sections if s.text.strip()] == ["Hours", "Menu"]


def test_chunks_keep_section_and_respect_size(tmp_path):
    f = tmp_path / "long.md"
    body = " ".join(f"Sentence number {i} talks about bread." for i in range(200))
    f.write_text(f"# Doc\n\n## Section A\n{body}\n", encoding="utf-8")
    chunks = chunk_document(f, tmp_path, max_words=100, overlap_words=10)
    assert len(chunks) > 5
    assert all(c.section == "Section A" and c.source == "long.md" for c in chunks)
    assert all(len(c.text.split()) <= 100 + 12 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_clean_fixes_pdf_hyphenation():
    assert clean("hyphen-\nated  words\n\n\n\nend") == "hyphenated words\n\nend"


def test_html_parsing(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><head><title>T</title><script>x=1</script></head><body><h2>Part</h2><p>Hello <b>world</b></p></body></html>")
    title, sections = parse(f)
    assert title == "T"
    assert sections[-1].heading == "Part" and "Hello world" in sections[-1].text and "x=1" not in sections[-1].text


@pytest.mark.skipif(not KNOW.exists(), reason="run scripts/make_knowledge_fixtures.py")
@pytest.mark.parametrize("name,needle", [("moonlight_bakery.md", "9:00 to 13:00"),
                                         ("orion_remote_work_policy.docx", "450 euros"),
                                         ("aurora_x2_manual.pdf", "blinks purple"),
                                         ("riverton_trail.txt", "11.4")])
def test_every_format_is_parsed(name, needle):
    chunks = chunk_document(KNOW / name, KNOW)
    assert any(needle in c.text for c in chunks), [c.text[:80] for c in chunks]
