"""CPU-only PDF extraction (native text layer, no OCR) and page rendering.

Everything here takes and returns picklable data so it can run in a process pool.
"""
import hashlib
import os
import re
from collections import Counter

import pymupdf

from src.ingestion.schemas import ExtractedDocument, PageRecord

RENDER_LONG_EDGE = int(os.getenv("PDF_RENDER_LONG_EDGE", "1456"))

_LECTURE_RE = re.compile(r"lecture[\s_-]*0*(\d+)", re.IGNORECASE)
_LEADING_NUM_RE = re.compile(r"^(\d+)")
_WORD_SPLIT_RE = re.compile(r"[^0-9a-z]+")

_CONTAINMENT_THRESHOLD = 0.95


def parse_lecture_number(file_name: str) -> int | None:
    match = _LECTURE_RE.search(file_name) or _LEADING_NUM_RE.match(file_name)
    return int(match.group(1)) if match else None


def extract_document(pdf_path: str, file_name: str) -> ExtractedDocument:
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    pages: list[PageRecord] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            rect = page.rect
            pages.append(
                PageRecord(
                    page_number=index + 1,
                    text=page.get_text("text", sort=True),
                    image_xrefs=[img[0] for img in page.get_images()],
                    drawing_count=len(page.get_drawings()),
                    width=rect.width,
                    height=rect.height,
                )
            )

    return ExtractedDocument(
        file_name=file_name,
        file_hash=file_hash,
        lecture_number=parse_lecture_number(file_name),
        page_count=len(pages),
        pages=pages,
    )


def render_page_png(pdf_path: str, page_number: int, long_edge: int = RENDER_LONG_EDGE) -> bytes:
    """Render a 1-based page so its longer side is `long_edge` pixels."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]
        scale = long_edge / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png")


def _words(text: str) -> Counter:
    return Counter(w for w in _WORD_SPLIT_RE.split(text.lower()) if w)


def _title_line(text: str) -> str:
    first = next((line for line in text.splitlines() if line.strip()), "")
    return " ".join(w for w in _WORD_SPLIT_RE.split(first.lower()) if w)


def _is_build_of(current: PageRecord, current_words: Counter, nxt: PageRecord, next_words: Counter) -> bool:
    total = sum(current_words.values())
    if total == 0:
        return False
    if not set(current.image_xrefs) <= set(nxt.image_xrefs):
        return False
    if nxt.drawing_count < current.drawing_count:
        return False
    # A build step keeps the slide title; this stops different slides with overlapping text from merging.
    if _title_line(current.text) != _title_line(nxt.text):
        return False
    # Short pure numbers (the slide-number footer) change on every slide, so they don't count.
    words = Counter({w: c for w, c in current_words.items() if not (w.isdigit() and len(w) <= 3)})
    total = sum(words.values())
    if total == 0:
        return False
    matched = sum(min(count, next_words[word]) for word, count in words.items())
    return matched / total >= _CONTAINMENT_THRESHOLD


def collapse_builds(pages: list[PageRecord]) -> list[tuple[PageRecord, list[int]]]:
    """Collapse build sequences (slide N+1 = slide N plus more) into their last page."""
    kept: list[tuple[PageRecord, list[int]]] = []
    pending: list[int] = []
    words = [_words(p.text) for p in pages]
    for i, page in enumerate(pages):
        pending.append(page.page_number)
        if i + 1 < len(pages) and _is_build_of(page, words[i], pages[i + 1], words[i + 1]):
            continue
        kept.append((page, pending))
        pending = []
    return kept
