"""Create a small FICTIONAL knowledge base in every supported format (md, txt, docx, html, pdf).

Facts are invented so the LLM cannot answer from prior knowledge — tests then prove that
answers come from retrieval. The PDF is printed by Playwright's Chromium (tests/e2e/make_pdf.mjs).

Usage: .venv\\Scripts\\python scripts/make_knowledge_fixtures.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "knowledge"

BAKERY_MD = """# Moonlight Bakery

## About
Moonlight Bakery is a small family bakery at 14 Willow Lane in Riverton. It was founded in 2011 by Amara Okafor,
who still bakes the morning bread herself.

## Opening hours
Monday to Friday the bakery is open from 7:00 to 18:00. On Saturday it opens from 8:00 to 16:00.
On Sunday the Moonlight Bakery is open from 9:00 to 13:00. It is closed on public holidays.

## Menu highlights
The honey oat loaf costs 6.50 dollars. Blueberry lemon scones cost 3.20 dollars and cardamom buns cost 2.90 dollars.
Gluten-free bread is baked only on Thursdays.

## Contact
Orders for birthday cakes must be placed at least three days in advance by phone at 555-0142.
"""

TRAIL_TXT = """Riverton Ridge Trail

The Riverton Ridge Trail is an 11.4 kilometre loop with 380 metres of elevation gain. Most hikers need about
four hours. The trail starts at the Old Mill car park. Dogs are allowed on a leash. The viewpoint called
Eagle Rock is at kilometre 6.2 and has a picnic table. In winter the upper section is closed from December to February.
"""

LAMP_HTML = """<!doctype html><html><head><title>Aurora X2 Smart Lamp - User Manual</title>
<style>body{font-family:Arial;margin:40px} h1{font-size:26px} h2{font-size:18px;margin-top:28px}</style></head><body>
<h1>Aurora X2 Smart Lamp - User Manual</h1>
<h2>Pairing</h2><p>To pair the Aurora X2 with the Aurora app, hold the power button for seven seconds until the light
blinks purple three times. Then select the lamp in the app within two minutes.</p>
<h2>Battery</h2><p>The built-in battery lasts about twelve hours at medium brightness and charges fully in ninety minutes
with the included USB-C cable.</p>
<h2>Factory reset</h2><p>To reset the lamp, press the power button and the dimmer button together for fifteen seconds
until the light turns solid white.</p>
<h2>Warranty</h2><p>The Aurora X2 has a two-year limited warranty. Water damage is not covered.</p>
</body></html>"""

POLICY = {
    "title": "Orion Labs Remote Work Policy",
    "sections": [
        ("Eligibility", "Employees may work remotely after completing a three-month probation period."),
        ("Core hours", "Remote employees must be available during core hours from 10:00 to 15:00 Riverton time."),
        ("Equipment allowance", "Orion Labs pays a one-time equipment allowance of 450 euros every two years for a desk, "
                                "chair or monitor. Receipts must be submitted within 30 days."),
        ("Travel", "Remote employees visit the Riverton office at least once per quarter; travel costs are reimbursed."),
    ],
}

QUESTIONS = [
    {"q": "What are the opening hours of the Moonlight Bakery on Sundays?", "source": "moonlight_bakery.md", "expect": ["9", "13"]},
    {"q": "How much is the equipment allowance at Orion Labs?", "source": "orion_remote_work_policy.docx", "expect": ["450"]},
    {"q": "How do I pair the Aurora X2 lamp?", "source": "aurora_x2_manual.pdf", "expect": ["seven", "purple"]},
    {"q": "How long is the Riverton Ridge Trail?", "source": "riverton_trail.txt", "expect": ["11.4"]},
    {"q": "Who founded the Moonlight Bakery?", "source": "moonlight_bakery.md", "expect": ["Amara"]},
    {"q": "What are the core hours for remote employees at Orion Labs?", "source": "orion_remote_work_policy.docx", "expect": ["10", "15"]},
    {"q": "What is the Wi-Fi password of the Moonlight Bakery?", "source": None, "expect": [], "unanswerable": True},
    {"q": "What is the capital of France?", "source": None, "expect": ["Paris"], "general": True},
]


def main() -> None:
    import json

    import docx

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "moonlight_bakery.md").write_text(BAKERY_MD, encoding="utf-8")
    (OUT / "riverton_trail.txt").write_text(TRAIL_TXT, encoding="utf-8")
    (OUT / "aurora_x2_manual.html").write_text(LAMP_HTML, encoding="utf-8")
    d = docx.Document()
    d.core_properties.title = POLICY["title"]
    d.add_heading(POLICY["title"], level=0)
    for heading, text in POLICY["sections"]:
        d.add_heading(heading, level=1)
        d.add_paragraph(text)
    d.save(str(OUT / "orion_remote_work_policy.docx"))
    subprocess.run(["node", "make_pdf.mjs", str(OUT / "aurora_x2_manual.html"), str(OUT / "aurora_x2_manual.pdf")],
                   cwd=ROOT / "tests" / "e2e", check=True, shell=False)
    (OUT / "aurora_x2_manual.html").unlink()   # keep the PDF only (html parsing is tested separately)
    (OUT.parent / "rag_questions.json").write_text(json.dumps(QUESTIONS, indent=2), encoding="utf-8")
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
