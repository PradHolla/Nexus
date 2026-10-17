"""Pydantic models for the quiz pipeline: LLM structured outputs (blueprint, draft, critique, dedup) and API outputs."""
from typing import Literal

from pydantic import BaseModel

Difficulty = Literal["easy", "medium", "hard"]
Style = Literal["definition", "application", "comparison", "diagnosis", "calculation"]


class Slot(BaseModel):  # LLM output (blueprint)
    topic: str
    target_files: list[str]
    search_query: str
    difficulty: Difficulty
    style: Style


class Blueprint(BaseModel):  # LLM output
    slots: list[Slot]
    reserve_slots: list[Slot]


class DraftQuestion(BaseModel):  # LLM output (generator)
    question_text: str
    options: list[str]
    correct_index: int
    explanation: str
    cited: list[str]  # excerpt labels like "C1"


class Critique(BaseModel):  # LLM output (critic)
    grounded: bool
    single_correct: bool
    distractors_plausible: bool
    requires_course_material: bool
    matches_difficulty: bool
    no_answer_leak: bool
    verdict: Literal["approve", "revise", "reject"]
    feedback: str


class DuplicateGroups(BaseModel):  # LLM output (dedup)
    groups: list[list[int]]


class Source(BaseModel):
    file_name: str
    lecture_number: int | None
    page_start: int
    page_end: int


class QuizQuestion(BaseModel):  # API output
    question_number: int
    question_text: str
    options: list[str]  # 4, already shuffled
    correct_answer: str  # equals one option exactly
    explanation: str
    source_chunk_ids: list[str]
    sources: list[Source]
    topic: str
    difficulty: Difficulty


class QuarantinedQuestion(BaseModel):  # API output
    question_number: int
    drafted_question: str  # last draft's question_text, or "" if none
    rejection_reason: str
    topic: str
