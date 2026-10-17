"""Whole-lecture analysis (title, summary, topics) with Haiku and a disk cache."""
import hashlib
import os
import uuid
from pathlib import Path

from src.config import CACHE_DIR, LLM_MODEL
from src.ingestion.schemas import DocumentAnalysis, PageContent
from src.llm import LLMUsage, generate_structured

PROMPT_VERSION = "v1"

CHAR_BUDGET = 240_000

SYSTEM = (
    "You analyze one lecture's slides (transcribed per page) for a course study database. "
    "Return: title — the lecture's title (from the title slide if present); "
    "summary — 150-250 words covering what the lecture teaches, in order, naming the key concepts, "
    "methods, equations and results, so a quiz planner can see what this lecture covers; "
    "topics — 5-15 short topic names, in lecture order, each a concept a quiz question could test. "
    "Ignore navigation and admin pages."
)

_TRUNCATION_NOTE = (
    "Note: the lecture was too long, so figure descriptions and/or page text below have been truncated.\n\n"
)


def _render(pages: list[PageContent], texts: list[str]) -> str:
    return "\n\n".join(
        f"=== Page {p.page_number} ({p.page_type}) — {p.title} ===\n{t}" for p, t in zip(pages, texts)
    )


def _drop_figures(text: str) -> str:
    return "\n\n".join(par for par in text.split("\n\n") if not par.startswith("[Figure] "))


def build_analysis_input(pages: list[PageContent]) -> str:
    texts = [p.text for p in pages]
    rendered = _render(pages, texts)
    if len(rendered) <= CHAR_BUDGET:
        return rendered

    texts = [_drop_figures(t) for t in texts]
    rendered = _render(pages, texts)
    if len(rendered) > CHAR_BUDGET:
        overhead = len(rendered) - sum(len(t) for t in texts)
        ratio = max(CHAR_BUDGET - overhead, 0) / max(sum(len(t) for t in texts), 1)
        texts = [t[: int(len(t) * ratio)] for t in texts]
        rendered = _render(pages, texts)
    return _TRUNCATION_NOTE + rendered


def _cache_path(content: str) -> Path:
    key = hashlib.sha256((content + LLM_MODEL + PROMPT_VERSION).encode()).hexdigest()
    return CACHE_DIR / "analyze" / f"{key}.json"


def _read_cache(path: Path) -> DocumentAnalysis | None:
    try:
        return DocumentAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, analysis: DocumentAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(analysis.model_dump_json(), encoding="utf-8")
    os.replace(tmp, path)


async def analyze_document(
    file_name: str, pages: list[PageContent]
) -> tuple[DocumentAnalysis, LLMUsage | None]:
    """Returns (analysis, usage); usage is None on a cache hit."""
    content = f"File: {file_name}\n\n" + build_analysis_input(pages)
    path = _cache_path(content)
    cached = _read_cache(path)
    if cached is not None:
        return cached, None

    result = await generate_structured(
        DocumentAnalysis, system=SYSTEM, content=content, max_tokens=4000, effort="low"
    )
    _write_cache(path, result.output)
    return result.output, result.usage
