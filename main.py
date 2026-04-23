import re
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import concurrent.futures
from src.ingestion.embedder import process_and_ingest_document
from src.ingestion.ppt_parser import parse_ppt
from src.ingestion.pdf_parser import parse_pdf
from src.retrieval.sampler import QuizSampler
from src.quiz.generator import run_planner_agent, generate_validated_quiz
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

qdrant_client = QdrantClient("http://localhost:6333")
sampler = QuizSampler(qdrant_client)

job_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Nexus Backend...")
    yield
    print("Shutting down...")

app = FastAPI(title="Nexus API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
        job_status[job["job_id"]] = "processing"
        
    print(f"Starting parallel CPU parsing for {len(jobs)} files...")

    with concurrent.futures.ProcessPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(
                pool, 
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
            job_status[job_id] = f"failed: {str(result)}"
            print(f"Job {job_id} failed: {result}")
        else:
            academic_chunks = [c for c in result if not c.get("is_administrative")]
            all_academic_chunks.extend(academic_chunks)

    if all_academic_chunks:
        print(f"Sending {len(all_academic_chunks)} total chunks to the Bedrock Embedder...")
        process_and_ingest_document(all_academic_chunks, qdrant_client)

    for i, result in enumerate(results):
        if not isinstance(result, Exception):
            job_status[jobs[i]["job_id"]] = "completed"

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
            
        job_status[job_id] = "queued"
        
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
async def get_job_status(job_id: str):
    status = job_status.get(job_id)
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

@app.post("/api/generate-quiz")
async def create_quiz(request: QuizRequest):
    try:
        # ==========================================
        # STAGE 1: THE PLANNER
        # ==========================================
        plan = run_planner_agent(request.user_prompt, request.keywords)
        print(f"Planner queries: {plan.vector_queries}")
        
        # ==========================================
        # STAGE 2: RETRIEVAL
        # ==========================================
        context_chunks = sampler.get_quiz_chunks(
            course_id=request.course_id,
            num_questions=request.num_questions,
            file_filters=request.file_filters,
            vector_queries=plan.vector_queries
        )
        
        if not context_chunks:
            raise HTTPException(status_code=404, detail="No course materials found for this request.")

        # ==========================================
        # STAGE 3: GENERATOR & CRITIC LOOP
        # ==========================================
        quiz_draft = generate_validated_quiz(
            plan=plan, 
            chunks=context_chunks, 
            num_questions=request.num_questions
        )
            
        return {
            "questions": [q.model_dump() for q in quiz_draft.questions],
            "quarantined_questions": [q.model_dump() for q in quiz_draft.quarantined_questions]
        }
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)