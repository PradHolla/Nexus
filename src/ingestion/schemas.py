"""Pydantic models for the PDF ingestion pipeline (extraction, vision, analysis, chunking)."""
from typing import Literal

from pydantic import BaseModel

PageType = Literal["content", "navigation", "admin"]


class PageRecord(BaseModel):
    """Output of extraction, one per PDF page."""

    page_number: int  # 1-based
    text: str  # native text layer (no OCR)
    image_xrefs: list[int]  # xrefs from page.get_images()
    drawing_count: int  # len(page.get_drawings())
    width: float
    height: float


class ExtractedDocument(BaseModel):
    file_name: str
    file_hash: str  # sha256 hex of the file bytes
    lecture_number: int | None
    page_count: int
    pages: list[PageRecord]


class Figure(BaseModel):
    description: str


class PageTranscription(BaseModel):
    """Haiku structured output for one page (the LLM schema)."""

    page_type: PageType
    title: str
    markdown: str
    figures: list[Figure]


class PageContent(BaseModel):
    """Final per-page content after vision/fallback."""

    page_number: int  # the kept page
    covered_pages: list[int]  # this page plus any earlier build-pages collapsed into it, ascending
    page_type: PageType
    title: str
    text: str  # markdown + figure descriptions, ready for chunking
    used_vision: bool
    has_math: bool


class DocumentAnalysis(BaseModel):
    """Haiku structured output for the whole lecture."""

    title: str
    summary: str
    topics: list[str]


class Chunk(BaseModel):
    chunk_id: str
    course_id: str
    file_name: str
    lecture_number: int | None
    lecture_title: str
    page_start: int
    page_end: int
    page_type: PageType
    is_administrative: bool  # page_type == "admin"
    quiz_eligible: bool  # page_type == "content"
    has_math: bool
    used_vision: bool
    topic: str  # title of the first page in the chunk
    text: str
    char_count: int
