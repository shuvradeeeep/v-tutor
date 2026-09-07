"""
Generate the PDF test fixtures with pymupdf so the PDF path has real files to
chew on without depending on any copyrighted textbook.

    v-tutor/Scripts/python scripts/make_fixtures.py

Writes tests/fixtures/sample_chapter.pdf (3 pages, headings by font size,
running header + page-number footer) and tests/fixtures/scanned.pdf (no text
layer at all).
"""
from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

SECTIONS = [
    ("The Heart", [
        "The heart is a muscular organ about the size of a fist. It pumps blood through the whole body.",
        "An adult heart beats about 72 times per minute at rest. A child's heart beats a little faster.",
    ]),
    ("Chambers", [
        "The heart has four chambers. The two upper chambers are called atria.",
        "The two lower chambers are called ventricles. The left ventricle is the strongest chamber.",
    ]),
    ("Blood Flow", [
        "Blood leaves the heart through the aorta, the largest artery in the body.",
        "Veins carry blood back to the heart. Valves stop blood from flowing backwards.",
    ]),
    ("History", [
        "William Harvey described the circulation of blood in 1628.",
        "Before him, many people believed the liver made blood.",
    ]),
]


def make_chapter(path: Path) -> None:
    doc = fitz.open()
    per_page = 2
    for page_no in range(0, len(SECTIONS), per_page):
        page = doc.new_page()
        page.insert_text((72, 40), "Science for Class 6 - Chapter 3", fontsize=8, fontname="helv")
        y = 90.0
        for title, paras in SECTIONS[page_no:page_no + per_page]:
            page.insert_text((72, y), title, fontsize=18, fontname="hebo")
            y += 28
            for para in paras:
                rect = fitz.Rect(72, y, 540, y + 60)
                page.insert_textbox(rect, para, fontsize=11, fontname="helv")
                y += 60
            y += 12
        page.insert_text((300, 800), str(page_no // per_page + 1), fontsize=9, fontname="helv")
    doc.save(path)


def make_scanned(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(72, 72, 540, 700), color=(0, 0, 0), width=1)
    doc.save(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_chapter(OUT / "sample_chapter.pdf")
    make_scanned(OUT / "scanned.pdf")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
