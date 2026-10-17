"""Per-page Haiku vision transcription with a disk cache."""
import hashlib
import os
import re
import uuid
from pathlib import Path

from src.config import CACHE_DIR, LLM_MODEL
from src.ingestion.schemas import PageContent, PageRecord, PageTranscription
from src.llm import LLMUsage, generate_structured, image_block, text_block

PROMPT_VERSION = "v2"

SYSTEM = """You transcribe one lecture slide into clean markdown for a study database that quiz questions will be generated from.
The rendered slide image is the source of truth. The text layer provided with it is only a hint: it may be garbled, out of order, or missing text that is inside images.
Rules:
- Start the markdown with the slide title as a heading. Keep all meaningful slide text, in reading order. Use markdown headings, bullets and emphasis as on the slide.
- Write every equation in LaTeX: $...$ inline, $$...$$ for display equations.
- Write tables as markdown tables. If a table has more than 12 rows, transcribe the header and the rows the slide highlights or that matter most, then summarize the rest in one sentence.
- Do not include slide numbers, logos, or repeated footer text. Keep citations such as [Vaswani et al., 2017].
- figures: one entry per informative diagram, chart, plot or illustration. Describe what it shows (structure, labels, axes, trends) and the takeaway a student should learn from it. Skip purely decorative images (stock photos, memes, illustrations that carry no course information). Tables and equations already transcribed in markdown are not figures. Describe diagrams only in figures, never in the markdown, and never mention skipped decorative images anywhere.
- title: the slide's title, or a short descriptive title if it has none.
- page_type: "navigation" for title slides, outline/agenda slides and section dividers; "admin" for course logistics (deadlines, assignments, grading, office hours, project/exam announcements); otherwise "content". A content slide that ends with a brief logistics reminder is still "content"."""

_MATH_CHARS_RE = re.compile("[\U0001D400-\U0001D7FF∑∫√≤≥≈∞]")


def _cache_path(png: bytes) -> Path:
    key = hashlib.sha256(png + LLM_MODEL.encode() + PROMPT_VERSION.encode()).hexdigest()
    return CACHE_DIR / "vision" / f"{key}.json"


def _read_cache(path: Path) -> PageTranscription | None:
    try:
        return PageTranscription.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, transcription: PageTranscription) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(transcription.model_dump_json(), encoding="utf-8")
    os.replace(tmp, path)


async def transcribe_page(png: bytes, text_layer: str) -> tuple[PageTranscription, LLMUsage | None]:
    """Transcribe one rendered slide. Returns (transcription, usage); usage is None on a cache hit."""
    path = _cache_path(png)
    cached = _read_cache(path)
    if cached is not None:
        return cached, None

    result = await generate_structured(
        PageTranscription,
        system=SYSTEM,
        content=[image_block(png), text_block(f"Text layer (hint):\n{text_layer}")],
        max_tokens=8000,
        effort="low",
    )
    _write_cache(path, result.output)
    return result.output, result.usage


def build_page_content(
    page: PageRecord, covered: list[int], transcription: PageTranscription | None
) -> PageContent:
    if transcription is not None:
        markdown = transcription.markdown.strip()
        if not markdown.startswith("#"):
            markdown = f"# {transcription.title}\n\n{markdown}"
        text = markdown + "".join(
            "\n\n[Figure] " + figure.description for figure in transcription.figures
        )
        return PageContent(
            page_number=page.page_number,
            covered_pages=covered,
            page_type=transcription.page_type,
            title=transcription.title,
            text=text,
            used_vision=True,
            has_math="$" in text,
        )

    text = page.text.strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return PageContent(
        page_number=page.page_number,
        covered_pages=covered,
        page_type="content",
        title=first_line[:120] or f"Page {page.page_number}",
        text=text,
        used_vision=False,
        has_math=bool(_MATH_CHARS_RE.search(text)),
    )
