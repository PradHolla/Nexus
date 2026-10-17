"""End-to-end tests for PDF ingestion: real Qdrant, real Haiku, real local embeddings (no mocks).

Needs Qdrant on QDRANT_HOST:QDRANT_PORT (default localhost:6333) and ANTHROPIC_API_KEY. tests/conftest.py points
config at test_nexus_* collections and a fresh temp var dir, so each run makes real (cheap) Haiku calls.
"""
import asyncio
import collections
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
from src.ingestion.chunk import CHUNK_MAX_CHARS
from src.ingestion.graph import NODE_RUNS, build_graph
from src.ingestion.index import ensure_collections
from src.ingestion.runner import IngestionRunner
from src.job_status_store import create_job, get_job_status, init_db, upsert_job_status

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "data" / "samples" / "1-introduction.pdf"
FILE_NAME = "sample-lecture1.pdf"

# Pages of 1-introduction.pdf (1-based) used for the test PDF, in this order:
#   1  title slide (navigation page, little text)
#   8  "Course Outcomes", a plain bulleted-text slide
#  25  "Wikipedia: Natural language processing", a normal text slide
#  34  "Question Answering", a system architecture diagram plus a phone screenshot (diagram/image page)
#  38  long text slide (~1000 chars, exercises chunk sizing)
#  52  "Challenges - Scale", a table
SAMPLE_PAGES = [1, 8, 25, 34, 38, 52]

qdrant = QdrantClient(f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}")


@pytest.fixture(scope="session", autouse=True)
def test_collections():
    assert CHUNKS_COLLECTION == "test_nexus_chunks" and DOCUMENTS_COLLECTION == "test_nexus_documents"

    def drop() -> None:
        for name in (CHUNKS_COLLECTION, DOCUMENTS_COLLECTION):
            if qdrant.collection_exists(name):
                qdrant.delete_collection(name)

    drop()
    yield
    drop()


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdf") / FILE_NAME
    with pymupdf.open(SAMPLE_PDF) as source, pymupdf.open() as out:
        for page_number in SAMPLE_PAGES:
            out.insert_pdf(source, from_page=page_number - 1, to_page=page_number - 1)
        out.save(path)
    return path


def course_points(collection: str, course_id: str) -> list:
    records, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return records


def ingest(client: TestClient, pdf: Path, course_id: str) -> str:
    response = client.post(
        "/api/ingest",
        params={"course_id": course_id},
        files=[("files", (pdf.name, pdf.read_bytes(), "application/pdf"))],
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["jobs"][0]["job_id"]
    deadline = time.time() + 300
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status == "completed" or status.startswith("failed"):
            break
        time.sleep(1)
    else:
        pytest.fail("ingestion timed out after 300 s")
    logs = client.get(f"/api/jobs/{job_id}").json()["logs"]
    assert status == "completed", f"{status}\n" + "\n".join(logs)
    return job_id


def test_ingest_via_api(sample_pdf):
    with pymupdf.open(sample_pdf) as doc:
        pages_with_text = [i + 1 for i, page in enumerate(doc) if page.get_text().strip()]
        assert len(doc) == 6

    with TestClient(app) as client:
        ingest(client, sample_pdf, "TEST")

        chunks = course_points(CHUNKS_COLLECTION, "TEST")
        assert len(chunks) > 0
        covered: set[int] = set()
        for point in chunks:
            payload = point.payload
            assert payload["course_id"] == "TEST"
            assert payload["file_name"] == FILE_NAME
            assert payload["page_number"] == payload["page_start"]
            body = payload["text"].split("\n", 1)[1] if "\n" in payload["text"] else ""
            assert len(body) <= CHUNK_MAX_CHARS
            covered.update(range(payload["page_start"], payload["page_end"] + 1))
        assert set(pages_with_text) <= covered

        documents = course_points(DOCUMENTS_COLLECTION, "TEST")
        assert len(documents) == 1
        assert documents[0].payload["summary"].strip()
        assert documents[0].payload["chunk_count"] == len(chunks)

        response = client.get("/api/courses/TEST/documents")
        assert response.status_code == 200
        listed = response.json()["documents"]
        assert len(listed) == 1
        assert listed[0]["file_name"] == FILE_NAME
        assert listed[0]["lecture_number"] == 1

        assert FILE_NAME in client.get("/api/courses/TEST/files").json()["files"]

        # Re-ingesting the same file replaces its points: same ids, same counts.
        ids_before = {str(p.id) for p in chunks}
        ingest(client, sample_pdf, "TEST")
        chunks_after = course_points(CHUNKS_COLLECTION, "TEST")
        assert len(chunks_after) == len(chunks)
        assert {str(p.id) for p in chunks_after} == ids_before
        assert len(course_points(DOCUMENTS_COLLECTION, "TEST")) == 1


def test_rejects_non_pdf():
    with TestClient(app) as client:
        before = set(UPLOADS_DIR.glob("*"))
        response = client.post(
            "/api/ingest",
            params={"course_id": "TEST"},
            files=[("files", ("slides.pptx", b"not a pdf", "application/octet-stream"))],
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Only PDF files are supported"
        assert set(UPLOADS_DIR.glob("*")) == before


def test_crash_resume(sample_pdf, monkeypatch, tmp_path):
    course_id = "TEST_RESUME"
    marker = tmp_path / "fail_index_once"
    marker.write_text("x")
    monkeypatch.setenv("NEXUS_FAIL_INDEX_ONCE", str(marker))

    async def scenario() -> None:
        ensure_var_dirs()
        init_db()
        ensure_collections(qdrant)

        job_id = str(uuid.uuid4())
        upload = UPLOADS_DIR / f"{job_id}_{FILE_NAME}"
        shutil.copy(sample_pdf, upload)
        create_job(job_id, course_id, FILE_NAME, str(upload))

        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS_DB_PATH)) as saver:
            runner = IngestionRunner(build_graph(saver), qdrant)
            runner.submit(job_id, course_id, FILE_NAME, str(upload))
            await runner.wait_idle()
            status = get_job_status(job_id)
            assert status.startswith("failed") and "injected index failure" in status, status
            assert not marker.exists()
            assert upload.exists()
            assert course_points(CHUNKS_COLLECTION, course_id) == []

            # Simulate a crash mid-run: the job is left "processing" and a fresh runner recovers it.
            upsert_job_status(job_id, "processing")
            before = collections.Counter(NODE_RUNS)
            runner = IngestionRunner(build_graph(saver), qdrant)
            recovered = await runner.recover()
            await runner.wait_idle()

            assert job_id in recovered
            assert get_job_status(job_id) == "completed"
            delta = {node: NODE_RUNS[node] - before[node] for node in NODE_RUNS}
            assert delta.get("extract", 0) == 0
            assert delta.get("transcribe", 0) == 0
            assert delta.get("analyze", 0) == 0
            assert delta["index"] == 1
            assert len(course_points(CHUNKS_COLLECTION, course_id)) > 0
            assert not upload.exists()

    asyncio.run(scenario())
