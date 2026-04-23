# Nexus

Nexus is a full-stack, local-first retrieval-augmented generation application designed to synthesize high-difficulty, graduate-level academic assessments from raw course materials. It transforms unstructured PDFs and presentation slides into strictly validated JSON quiz structures using a multi-agent orchestration loop.

## Architecture Highlights

### 1. Semantic Chunking & Resilient Ingestion
Nexus implements a hybrid concurrency model to eliminate bottlenecks:
* **CPU-Bound Parsing:** A global `ProcessPoolExecutor` distributes document parsing (PDF to Markdown) across all available CPU cores.
* **I/O-Bound Embedding:** A `ThreadPoolExecutor` handles concurrent network requests to AWS Bedrock for Titan embeddings and Vision-LLM slide analysis.
* **Automatic Cleanup:** Temporary files are automatically purged from `/tmp` immediately after being indexed into the vector database.

### 2. Multi-Agent Orchestration (SSE Streaming)
The system uses a 3-stage agentic workflow (Planner -> Generator -> Critic) powered by Server-Sent Events (SSE). Users see real-time progress as:
* **The Planner** formulates a semantic search strategy and vector queries.
* **The Generator** drafts questions using a forced Chain-of-Thought reasoning scratchpad.
* **The Critic** ruthlessly reviews drafts for hallucinations or administrative trivia, triggering automated retry loops for failed questions.
* **Quarantine Zone:** Questions that fail the critic's checks 3 times are moved to a UI "Quarantine Zone" for full transparency.

### 3. Production-Ready State Management
* **SQLite Persistence:** Job statuses are stored in a local SQLite database (`jobs.db`), ensuring ingestion progress survives process restarts and remains consistent across multiple backend workers.
* **Namespace Isolation:** Every document is tagged with a Course ID, acting as a hard filter in Qdrant to prevent cross-contamination between different academic subjects.

## Tech Stack

**Backend**
* FastAPI (Python 3.12)
* Pydantic (Strict Schema Validation)
* SQLite (State Management)
* AWS Bedrock (Titan & Claude-3)
* PyMuPDF4LLM (Markdown Parsing)

**Frontend**
* React 19 + Vite
* Tailwind CSS v4
* Lucide React

**Infrastructure**
* Qdrant (Vector Database with INT8 Quantization)
* Docker & Docker Compose

## Deployment

### 1. Production Deployment (Docker Compose)
The fastest way to deploy Nexus to an EC2 instance or local server:

1. Create a `.env` file with your credentials:
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

2. Build and start the orchestration:
```bash
# Replace <IP> with your EC2 Public IP for the frontend build
docker-compose build --build-arg VITE_API_BASE_URL=http://<IP>:8000/api
docker-compose up -d
```

### 2. Local Development Setup
If you prefer running components manually:

**Start Qdrant:**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Start Backend:**
```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Start Frontend:**
```bash
cd frontend
npm install
npm run dev
```
