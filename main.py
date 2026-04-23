import re
import uuid
import json
import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import concurrent.futures
from fastapi.responses import StreamingResponse
from src.ingestion.embedder import process_and_ingest_document
from src.ingestion.ppt_parser import parse_ppt
from src.ingestion.pdf_parser import parse_pdf
from src.job_status_store import (
    init_db,
    upsert_job_status,
    get_job_status as get_stored_job_status,
)
from src.retrieval.sampler import QuizSampler
from src.quiz.generator import run_planner_agent, generate_validated_quiz
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

# Pull from env so Docker can route to the 'qdrant' container, fallback to localhost for uv runs
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
qdrant_client = QdrantClient(f"http://{qdrant_host}:{qdrant_port}")
sampler = QuizSampler(qdrant_client)

# --- GLOBAL EXECUTOR ---
# Reusing the process pool is more efficient than creating one per request.
process_pool = concurrent.futures.ProcessPoolExecutor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Starting Nexus Backend...")
    yield
    print("Shutting down...")
    process_pool.shutdown()

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

def _cpu_bound_parse(file_path: str, filename: str, course_id: str) -> list:
    match = re.search(r'\d+', filename)
    actual_lecture_num = int(match.group()) if match else 0
    
    if filename.lower().endswith('.pptx'):
        return parse_ppt(file_path, filename, course_id, lecture_number=actual_lecture_num)
    else:
        return parse_pdf(file_path, filename, course_id, lecture_number=actual_lecture_num)

async def batch_ingest_task(jobs: list):
    loop = asyncio.get_running_loop()
    
    for job in jobs:
        upsert_job_status(job["job_id"], "processing")
        
    print(f"Starting parallel CPU parsing for {len(jobs)} files...")

    tasks = [
        loop.run_in_executor(
            process_pool, 
            _cpu_bound_parse, 
            job["file_path"], 
            job["filename"], 
            job["course_id"]
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_academic_chunks = []

    for i, result in enumerate(results):
        job_id = jobs[i]["job_id"]
        if isinstance(result, Exception):
            upsert_job_status(job_id, f"failed: {str(result)}")
            print(f"Job {job_id} failed: {result}")
        else:
            academic_chunks = [c for c in result if not c.get("is_administrative")]
            all_academic_chunks.extend(academic_chunks)

    if all_academic_chunks:
        print(f"Sending {len(all_academic_chunks)} total chunks to the Bedrock Embedder...")
        try:
            process_and_ingest_document(all_academic_chunks, qdrant_client)
        except Exception as e:
            print(f"Embedding/Ingestion failed: {e}")
            for job in jobs:
                if get_stored_job_status(job["job_id"]) == "processing":
                    upsert_job_status(job["job_id"], f"failed: ingestion error")

    # Finalize statuses and CLEANUP
    for i, result in enumerate(results):
        job_id = jobs[i]["job_id"]
        file_path = jobs[i]["file_path"]
        
        if not isinstance(result, Exception) and get_stored_job_status(job_id) == "processing":
            upsert_job_status(job_id, "completed")
        
        # --- CLEANUP: Remove temp file ---
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                print(f"Failed to delete temp file {file_path}: {e}")

    print("Batch ingestion finished.")

@app.post("/api/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    course_id: str,
    files: List[UploadFile] = File(...)
):
    jobs_data = []
    job_ids = []
    
    for file in files:
        if not file.filename.endswith(('.pdf', '.pptx')):
            raise HTTPException(status_code=400, detail="Unsupported file type")
            
        job_id = str(uuid.uuid4())
        temp_file_path = f"/tmp/{job_id}_{file.filename}" 
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        upsert_job_status(job_id, "queued")
        
        jobs_data.append({
            "job_id": job_id,
            "file_path": temp_file_path,
            "filename": file.filename,
            "course_id": course_id
        })
        job_ids.append({"filename": file.filename, "job_id": job_id})
        
    background_tasks.add_task(batch_ingest_task, jobs_data)
        
    return {"message": "Ingestion started", "jobs": job_ids}

@app.get("/api/jobs/{job_id}")
async def get_ingest_job_status(job_id: str):
    status = get_stored_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": status}

# --- GENERATION ENDPOINTS ---

@app.get("/api/courses/{course_id}/files")
async def get_course_files(course_id: str):
    files = sampler.get_course_files(course_id)
    return {"files": files}

class QuizRequest(BaseModel):
    course_id: str
    num_questions: int = 5
    user_prompt: str  
    keywords: Optional[List[str]] = None
    file_filters: Optional[List[str]] = None 


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
            plan = run_planner_agent(request.user_prompt, request.keywords)
            print(f"Planner queries: {plan.vector_queries}")
            yield _sse_event("planner_complete", {"vector_queries": plan.vector_queries})

            # ==========================================
            # STAGE 2: RETRIEVAL
            # ==========================================
            yield _sse_event("retrieval_started", {})
            context_chunks = sampler.get_quiz_chunks(
                course_id=request.course_id,
                num_questions=request.num_questions,
                file_filters=request.file_filters,
                vector_queries=plan.vector_queries,
            )

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
            print(f"API Error: {str(e)}")
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