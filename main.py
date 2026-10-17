from dotenv import load_dotenv

# Load .env before importing modules that create API clients at import time
load_dotenv()

import uuid
import json
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.config import CHECKPOINTS_DB_PATH, DOCUMENTS_COLLECTION, UPLOADS_DIR, ensure_var_dirs
from src.ingestion.graph import build_graph
from src.ingestion.index import ensure_collections
from src.ingestion.runner import IngestionRunner
from src.job_status_store import (
    init_db,
    create_job,
    get_job_status_and_logs,
)
from src.retrieval.sampler import QuizSampler
from src.quiz.generator import run_planner_agent, generate_validated_quiz
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http import models


# Pull from env so Docker can route to the 'qdrant' container, fallback to localhost for uv runs
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
qdrant_client = QdrantClient(f"http://{qdrant_host}:{qdrant_port}")
sampler = QuizSampler(qdrant_client)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_var_dirs()
    init_db()
    await asyncio.to_thread(ensure_collections, qdrant_client)
    print("Starting Nexus Backend...")
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS_DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)
        app.state.runner = IngestionRunner(graph, qdrant_client)
        recovered = await app.state.runner.recover()
        if recovered:
            print(f"Recovered unfinished ingestion jobs: {recovered}")
        try:
            yield
        finally:
            print("Shutting down...")
            # Stop running jobs before the checkpointer connection closes; they resume on the next start.
            await app.state.runner.shutdown()

app = FastAPI(title="Nexus API", lifespan=lifespan)

# Allow frontend origins dynamically via env var, keep localhost for dev
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INGESTION ENDPOINTS ---
@app.post("/api/ingest")
async def ingest_documents(
    course_id: str,
    files: List[UploadFile] = File(...)
):
    if any(not file.filename.lower().endswith(".pdf") for file in files):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    job_ids = []
    for file in files:
        job_id = str(uuid.uuid4())
        file_name = Path(file.filename).name
        temp_file_path = str(UPLOADS_DIR / f"{job_id}_{file_name}")

        content = await file.read()
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        create_job(job_id, course_id, file_name, temp_file_path)
        app.state.runner.submit(job_id, course_id, file_name, temp_file_path)
        job_ids.append({"filename": file.filename, "job_id": job_id})

    return {"message": "Ingestion started", "jobs": job_ids}

@app.get("/api/jobs/{job_id}")
async def get_ingest_job_status(job_id: str):
    res = get_job_status_and_logs(job_id)
    if not res:
        raise HTTPException(status_code=404, detail="Job not found")
    status, logs = res
    # Convert newline-separated logs into a list of strings for easier frontend consumption
    log_list = [line for line in logs.split('\n') if line.strip()]
    return {"job_id": job_id, "status": status, "logs": log_list}

# --- GENERATION ENDPOINTS ---

@app.get("/api/courses/{course_id}/files")
async def get_course_files(course_id: str):
    files = sampler.get_course_files(course_id)
    return {"files": files}

@app.get("/api/courses/{course_id}/documents")
async def get_course_documents(course_id: str):
    def scroll_all() -> list[dict]:
        documents = []
        offset = None
        while True:
            records, offset = qdrant_client.scroll(
                collection_name=DOCUMENTS_COLLECTION,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id))]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(r.payload for r in records)
            if offset is None:
                return documents

    documents = await asyncio.to_thread(scroll_all)
    documents.sort(key=lambda d: (d.get("lecture_number") is None, d.get("lecture_number") or 0, d["file_name"]))
    return {"documents": documents}

class QuizRequest(BaseModel):
    course_id: str
    num_questions: int = 5
    user_prompt: str  
    keywords: Optional[List[str]] = None
    file_filters: Optional[List[str]] = None 
    retrieval_mode: str = "hybrid"
    max_context_chunks: Optional[int] = None
    include_retrieval_diagnostics: bool = False


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

@app.post("/api/generate-quiz")
async def create_quiz(request: QuizRequest):
    async def event_stream():
        try:
            yield _sse_event("planner_started", {})

            # ==========================================
            # STAGE 1: THE PLANNER
            # ==========================================
            for event in run_planner_agent(request.user_prompt, request.keywords):
                if isinstance(event, str):
                    yield event
                else:
                    plan = event
            
            yield _sse_event("log", {"message": f"Planner queries: {plan.vector_queries}"})
            yield _sse_event("planner_complete", {"vector_queries": plan.vector_queries})

            # ==========================================
            # STAGE 2: RETRIEVAL
            # ==========================================
            yield _sse_event("retrieval_started", {})
            for event in sampler.get_quiz_chunks(
                course_id=request.course_id,
                num_questions=request.num_questions,
                file_filters=request.file_filters,
                vector_queries=plan.vector_queries,
                retrieval_mode=request.retrieval_mode,
                max_context_chunks=request.max_context_chunks,
                include_retrieval_diagnostics=request.include_retrieval_diagnostics,
            ):
                if isinstance(event, str):
                    yield event
                else:
                    context_chunks = event

            if not context_chunks:
                yield _sse_event("error", {"message": "No course materials found for this request."})
                return

            yield _sse_event("retrieval_complete", {"chunk_count": len(context_chunks)})

            # ==========================================
            # STAGE 3: GENERATOR & CRITIC LOOP
            # ==========================================
            for event in generate_validated_quiz(
                plan=plan,
                chunks=context_chunks,
                num_questions=request.num_questions,
            ):
                yield event
                await asyncio.sleep(0)

        except Exception as e:
            yield _sse_event("log", {"message": f"API Error: {str(e)}"})
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
