"""Deterministic page-aware chunking. Navigation/admin pages never share a chunk with content pages."""
import os
import re
from dataclasses import dataclass

from src.ingestion.schemas import Chunk, PageContent

CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1500"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "400"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2500"))

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_LINE_RE = re.compile(r"\n")
_BLANK_LINES_RE = re.compile(r"\n\s*\n")
_WS_RE = re.compile(r"\s+")


@dataclass
class _Piece:
    text: str
    page: PageContent


def _split_keep(text: str, boundary: re.Pattern) -> list[str]:
    """Split at boundary matches; each segment keeps its trailing separator so nothing is lost."""
    segments: list[str] = []
    start = 0
    for match in boundary.finditer(text):
        segments.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        segments.append(text[start:])
    return segments


def _merge_open_math(segments: list[str]) -> list[str]:
    """Re-join segments so no $$...$$ block is cut in the middle."""
    merged: list[str] = []
    buffer = ""
    for seg in segments:
        buffer += seg
        if buffer.count("$$") % 2 == 0:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def _hard_split(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    rest = text
    while len(rest.strip()) > max_chars:
        window = rest[: max_chars + 1]
        cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        if cut <= 0:
            cut = max_chars
        parts.append(rest[:cut])
        rest = rest[cut:]
    parts.append(rest)
    return parts


def _atomize(text: str, max_chars: int) -> list[str]:
    """Break text into units each <= max_chars: paragraphs, then sentences, then lines, then hard split."""
    units = _merge_open_math(_split_keep(text, _PARAGRAPH_RE))
    for splitter in (_SENTENCE_RE, _LINE_RE):
        refined: list[str] = []
        for unit in units:
            if len(unit.strip()) > max_chars:
                refined.extend(_merge_open_math(_split_keep(unit, splitter)))
            else:
                refined.append(unit)
        units = refined
    final: list[str] = []
    for unit in units:
        if len(unit.strip()) > max_chars:
            final.extend(_hard_split(unit, max_chars))
        else:
            final.append(unit)
    return final


def _split_page_text(text: str, target: int, max_chars: int) -> list[str]:
    """Split one oversized page into parts (<= max_chars), packing atoms up to target. Never drops text."""
    limit = min(target, max_chars)
    parts: list[str] = []
    current = ""
    for unit in _atomize(text, max_chars):
        if current and len((current + unit).strip()) > limit:
            parts.append(current.strip())
            current = unit
        else:
            current += unit
    if current.strip():
        parts.append(current.strip())
    return parts


def _pack(pieces: list[_Piece], target: int, min_chars: int, max_chars: int) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_len = 0
    for piece in pieces:
        if not current:
            current, current_len = [piece], len(piece.text)
            continue
        new_len = current_len + 2 + len(piece.text)
        if new_len <= target or (current_len < min_chars and new_len <= max_chars):
            current.append(piece)
            current_len = new_len
        else:
            groups.append(current)
            current, current_len = [piece], len(piece.text)
    if current:
        groups.append(current)
    return groups


def _runs(pages: list[PageContent]) -> list[list[PageContent]]:
    """Consecutive same-type non-empty pages."""
    runs: list[list[PageContent]] = []
    for page in pages:
        if not page.text.strip():
            continue
        if runs and runs[-1][0].page_type == page.page_type:
            runs[-1].append(page)
        else:
            runs.append([page])
    return runs


def chunk_document(
    pages: list[PageContent],
    *,
    course_id: str,
    file_name: str,
    lecture_number: int | None,
    lecture_title: str,
    target: int = CHUNK_TARGET_CHARS,
    min_chars: int = CHUNK_MIN_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[Chunk]:
    header_title = _WS_RE.sub(" ", lecture_title).strip()
    chunks: list[Chunk] = []
    for run in _runs(pages):
        pieces: list[_Piece] = []
        for page in run:
            text = page.text.strip()
            if len(text) <= max_chars:
                pieces.append(_Piece(text, page))
            else:
                pieces.extend(_Piece(part, page) for part in _split_page_text(text, target, max_chars))

        for group in _pack(pieces, target, min_chars, max_chars):
            covered = [n for piece in group for n in piece.page.covered_pages]
            page_start, page_end = min(covered), max(covered)
            pages_label = f"p{page_start}" + (f"-{page_end}" if page_end != page_start else "")
            # Slide title in the header keeps split continuation chunks self-describing.
            topic = _WS_RE.sub(" ", group[0].page.title).strip()
            header = f"[{header_title} — {pages_label} — {topic}]\n"
            text = header + "\n\n".join(piece.text for piece in group)
            page_type = group[0].page.page_type
            chunks.append(
                Chunk(
                    chunk_id=f"{course_id}:{file_name}:p{page_start}-{page_end}:{len(chunks)}",
                    course_id=course_id,
                    file_name=file_name,
                    lecture_number=lecture_number,
                    lecture_title=lecture_title,
                    page_start=page_start,
                    page_end=page_end,
                    page_type=page_type,
                    is_administrative=page_type == "admin",
                    quiz_eligible=page_type == "content",
                    has_math=any(piece.page.has_math for piece in group),
                    used_vision=any(piece.page.used_vision for piece in group),
                    topic=group[0].page.title,
                    text=text,
                    char_count=len(text),
                )
            )
    return chunks


def check_chunks(pages: list[PageContent], chunks: list[Chunk], max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Return a list of invariant violations (empty list means all good)."""
    violations: list[str] = []
    bodies = [(c, c.text.split("\n", 1)[1] if "\n" in c.text else "") for c in chunks]

    for chunk, body in bodies:
        if len(body) > max_chars:
            violations.append(f"{chunk.chunk_id}: body is {len(body)} chars > max {max_chars}")

    for page in pages:
        if not page.text.strip():
            continue
        # whitespace is removed entirely so a hard split inside a whitespace-free run still compares equal
        joined = _WS_RE.sub("", "".join(b for c, b in bodies if c.page_start <= page.page_number <= c.page_end))
        for paragraph in _BLANK_LINES_RE.split(page.text.strip()):
            normalized = _WS_RE.sub("", paragraph)
            if normalized and normalized not in joined:
                violations.append(f"page {page.page_number}: paragraph missing from chunks: {normalized[:80]!r}")

    for page in pages:
        if not page.text.strip():
            continue
        for n in page.covered_pages:
            if not any(c.page_start <= n <= c.page_end for c in chunks):
                violations.append(f"page {n} (covered by {page.page_number}) is in no chunk range")
    return violations
