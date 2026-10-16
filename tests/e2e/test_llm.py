"""End-to-end tests for src.llm against the real Anthropic API (needs ANTHROPIC_API_KEY)."""
import asyncio
from pathlib import Path

import pymupdf
import pytest
from pydantic import BaseModel

from src.llm import (
    LLMTruncatedError,
    generate_structured,
    image_block,
    text_block,
)

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "data" / "samples" / "1-introduction.pdf"


class Answer(BaseModel):
    capital: str
    country: str


class SlideInfo(BaseModel):
    institution: str
    mentions_nlp: bool


class Essay(BaseModel):
    paragraphs: list[str]


def test_text_structured_output():
    result = asyncio.run(
        generate_structured(
            Answer,
            system="Answer concisely.",
            content="What is the capital of France? Reply with the country too.",
            max_tokens=1000,
            effort="low",
        )
    )
    assert "paris" in result.output.capital.lower()
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.stop_reason == "end_turn"
    assert result.model.startswith("claude-haiku-5-5")


def test_image_input():
    # Page 1 text (verified via get_text()): "Introduction / CS 584 Natural Language
    # Processing / Ping Wang / Department of Computer Science / Stevens Institute of Technology"
    doc = pymupdf.open(SAMPLE_PDF)
    png = doc[0].get_pixmap(dpi=72).tobytes("png")
    result = asyncio.run(
        generate_structured(
            SlideInfo,
            system="You read lecture slides. Give the institution shown on the slide and whether it mentions NLP.",
            content=[image_block(png), text_block("Read this slide.")],
            max_tokens=1000,
            effort="low",
        )
    )
    # Asserting on the institution rather than the title: the model sometimes reads the
    # course name ("CS 584 Natural Language Processing") as the title, which is also valid.
    assert "stevens" in result.output.institution.lower()
    assert result.output.mentions_nlp is True


def test_truncation_raises():
    with pytest.raises(LLMTruncatedError):
        asyncio.run(
            generate_structured(
                Essay,
                system="Write exactly what is asked.",
                content="Write a 10-paragraph essay about the history of the printing press.",
                max_tokens=16,
            )
        )


def test_two_event_loops():
    for _ in range(2):
        result = asyncio.run(
            generate_structured(
                Answer,
                system="Answer concisely.",
                content="What is the capital of Japan? Reply with the country too.",
                max_tokens=200,
            )
        )
        assert "tokyo" in result.output.capital.lower()
