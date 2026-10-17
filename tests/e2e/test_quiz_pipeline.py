"""End-to-end tests for the quiz pipeline: real Qdrant test collections, real Haiku, real local embeddings (no mocks).

Ingests two small lectures into course TESTQ once per module, then drives /api/generate-quiz and parses its SSE stream.
Needs Qdrant on QDRANT_HOST:QDRANT_PORT (default localhost:6333) and ANTHROPIC_API_KEY. Run with -s to see the quizzes.
"""
import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from qdrant_client import QdrantClient
from qdrant_client.http import models

from main import app
from src.config import CHECKPOINTS_DB_PATH, CHUNKS_COLLECTION, DOCUMENTS_COLLECTION, UPLOADS_DIR, ensure_var_dirs
from src.ingestion.graph import build_graph
from src.ingestion.index import ensure_collections
from src.ingestion.runner import IngestionRunner
from src.job_status_store import create_job, get_job_status, init_db

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "data" / "samples" / "1-introduction.pdf"
COURSE_ID = "TESTQ"

# Pages of 1-introduction.pdf (1-based) for the two test lectures. Chosen from the deck's text layer as slides with
# real teaching content (190+ characters), skipping the syllabus/grading/policy slides (3-23) and sparse
# title/divider slides, so every page can seed a quiz question. The two halves cover different material.
# Slides 7-8 (course outcomes / "what we learn") are left out on purpose: they only list topics taught later
# (gradient descent, seq2seq, ...), so the planner would plan questions the test material cannot support.
# Each question gets its own chunks (a chunk is assigned to at most one slot) and slides are merged into chunks of
# ~1000 characters. With ~10 pages per file there were only ~15 chunks in total, so slots (and the reserves) lost the
# chunks that matched their topic to earlier slots, got off-topic leftovers and were quarantined. So the content-rich
# stretches of each half are used whole (~25 pages each), which leaves enough chunks per topic.
#   Part A, first half (NLP definition, applications, text/QA/extraction examples, bias, LLMs): pages 25-42
#   Part B, second half (challenges and ambiguity, NLP pipeline: tokenization, POS tagging, semantic analysis, NER,
#     then Python basics): pages 43-60 plus 63-65, 70-72
# Skipped on purpose: 1-23 (title, syllabus, grading, policies; 7-8 only list topics taught later, so the planner
# would plan questions the material cannot support), 61-62 (thank-you and title slides), 66-69 and 73-84 (code snippets).
PART_A = ("quiz-part-a.pdf", list(range(25, 43)))
PART_B = ("quiz-part-b.pdf", [*range(43, 61), 63, 64, 65, 70, 71, 72])

qdrant = QdrantClient(f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}")


@pytest.fixture(scope="module", autouse=True)
def quiz_course(tmp_path_factory):
    pdf_dir = tmp_path_factory.mktemp("quiz_pdfs")
    pdfs = []
    for file_name, pages in (PART_A, PART_B):
        path = pdf_dir / file_name
        with pymupdf.open(SAMPLE_PDF) as source, pymupdf.open() as out:
            for page_number in pages:
                out.insert_pdf(source, from_page=page_number - 1, to_page=page_number - 1)
            out.save(path)
        pdfs.append(path)

    async def ingest_all() -> None:
        ensure_var_dirs()
        init_db()
        ensure_collections(qdrant)
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS_DB_PATH)) as saver:
            runner = IngestionRunner(build_graph(saver), qdrant)
            job_ids = []
            for pdf in pdfs:
                job_id = str(uuid.uuid4())
                upload = UPLOADS_DIR / f"{job_id}_{pdf.name}"
                shutil.copy(pdf, upload)
                create_job(job_id, COURSE_ID, pdf.name, str(upload))
                runner.submit(job_id, COURSE_ID, pdf.name, str(upload))
                job_ids.append(job_id)
            await runner.wait_idle()
            statuses = [get_job_status(j) for j in job_ids]
            assert statuses == ["completed", "completed"], statuses

    asyncio.run(ingest_all())
    yield
    selector = models.FilterSelector(
        filter=models.Filter(
            must=[models.FieldCondition(key="course_id", match=models.MatchValue(value=COURSE_ID))]
        )
    )
    for collection in (CHUNKS_COLLECTION, DOCUMENTS_COLLECTION):
        qdrant.delete(collection_name=collection, points_selector=selector)


def generate_quiz(request: dict) -> tuple[list[tuple[str, dict]], float]:
    """POST /api/generate-quiz and parse the SSE body into [(event, data)]; also returns the wall time."""
    started = time.monotonic()
    response = TestClient(app).post("/api/generate-quiz", json=request)  # no lifespan needed for this endpoint
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    events = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        events.append((name, json.loads("\n".join(data)) if data else {}))
    return events, elapsed


def chunk_payloads(chunk_ids: list[str]) -> dict[str, dict]:
    records, _ = qdrant.scroll(
        collection_name=CHUNKS_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids))]
        ),
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=False,
    )
    return {r.payload["chunk_id"]: r.payload for r in records}


def print_quiz(title: str, complete: dict, elapsed: float) -> None:
    print(f"\n===== {title} ({elapsed:.1f}s, usage {complete['usage']}) =====")
    for q in complete["questions"]:
        print(f"\nQ{q['question_number']} [{q['difficulty']}] ({q['topic']}): {q['question_text']}")
        for i, option in enumerate(q["options"]):
            print(f"  {chr(65 + i)}. {option}{'   <-- correct' if option == q['correct_answer'] else ''}")
        print(f"  Explanation: {q['explanation']}")
        for s in q["sources"]:
            print(f"  Source: {s['file_name']} pp.{s['page_start']}-{s['page_end']}")
    for q in complete["quarantined_questions"]:
        print(f"\nQUARANTINED Q{q['question_number']} ({q['topic']}): {q['drafted_question']!r} -> {q['rejection_reason']}")


def single_complete(events: list[tuple[str, dict]]) -> dict:
    names = [name for name, _ in events]
    assert "error" not in names, [d for n, d in events if n == "error"]
    assert names.count("complete") == 1, names
    return next(d for n, d in events if n == "complete")


def check_question(q: dict) -> None:
    assert len(q["options"]) == 4 and len(set(q["options"])) == 4, q["options"]
    assert q["correct_answer"] in q["options"]
    assert q["explanation"].strip()
    assert q["difficulty"] in {"easy", "medium", "hard"}
    assert q["source_chunk_ids"]


def test_quiz_end_to_end():
    events, elapsed = generate_quiz(
        {
            "course_id": COURSE_ID,
            "num_questions": 4,
            "user_prompt": "A quiz covering the main ideas of these lectures",
            "difficulty": "mixed",
        }
    )
    complete = single_complete(events)
    questions, quarantined = complete["questions"], complete["quarantined_questions"]
    print_quiz("test_quiz_end_to_end", complete, elapsed)
    print("event sequence:", [name for name, _ in events])
    print("log messages:", *[d["message"] for n, d in events if n == "log"], sep="\n  ")

    assert len(questions) <= 4
    assert len(questions) + len(quarantined) >= 4
    # Quality (delivery and quarantine rates) is measured by eval/run_quiz_eval.py on the real course; this e2e test
    # checks the pipeline's invariants on a small sample deck, where quarantines are more frequent.
    assert len(questions) >= 2, f"quarantine reasons: {[q['rejection_reason'] for q in quarantined]}"
    assert len({q["question_number"] for q in questions}) == len(questions)
    assert complete["usage"]["cost_usd"] > 0

    chunks = chunk_payloads([cid for q in questions for cid in q["source_chunk_ids"]])
    for q in questions:
        check_question(q)
        assert len(q["sources"]) == len(q["source_chunk_ids"])
        for chunk_id, source in zip(q["source_chunk_ids"], q["sources"]):
            payload = chunks[chunk_id]
            assert payload["course_id"] == COURSE_ID
            assert payload["quiz_eligible"] is True
            assert source == {
                "file_name": payload["file_name"],
                "lecture_number": payload["lecture_number"],
                "page_start": payload["page_start"],
                "page_end": payload["page_end"],
            }


def test_quiz_file_filter():
    events, elapsed = generate_quiz(
        {
            "course_id": COURSE_ID,
            "num_questions": 2,
            "user_prompt": "A quiz on the key ideas of this lecture",
            "file_filters": ["quiz-part-b.pdf"],
            "difficulty": "easy",
        }
    )
    complete = single_complete(events)
    print_quiz("test_quiz_file_filter", complete, elapsed)
    assert complete["questions"], "no questions delivered"
    for q in complete["questions"]:
        check_question(q)
        assert q["difficulty"] == "easy"
        assert {s["file_name"] for s in q["sources"]} == {"quiz-part-b.pdf"}


def test_quiz_unknown_course():
    events, _ = generate_quiz(
        {"course_id": "NOPE", "num_questions": 2, "user_prompt": "Any quiz", "difficulty": "mixed"}
    )
    names = [name for name, _ in events]
    assert "error" in names
    assert "complete" not in names
