# Nexus

Nexus is a full-stack, local-first retrieval-augmented generation application designed to synthesize high-difficulty, graduate-level academic assessments from raw course materials. It transforms unstructured PDFs and presentation slides into strictly validated JSON quiz structures using a multi-agent orchestration loop.

---

**Live Subdomain**: [nexus.pradholla.com](https://nexus.pradholla.com/)

**Demo Link**: [Link](https://stevens.zoom.us/rec/share/NqTLErWLaNAzajxuTSO45eL9NhDgaUpZccyvzfor2QUdE63i5CF3tHHBBdqeRNEE.1gNIqSoQoD2Z3s6E?startTime=1777000389000)

## Architecture Highlights

### Semantic Chunking & Resilient Ingestion

Nexus implements a hybrid concurrency model to eliminate bottlenecks:

- **CPU-Bound Parsing**  
  A global `ProcessPoolExecutor` distributes document parsing (PDF → Markdown) across all available CPU cores.

- **I/O-Bound Embedding**  
  A `ThreadPoolExecutor` handles concurrent network requests to AWS Bedrock for Titan embeddings and Vision-LLM slide analysis.

- **Automatic Cleanup**  
  Temporary files are automatically purged from `/tmp` immediately after being indexed into the vector database.

---

### Multi-Agent Orchestration (SSE Streaming)

The system uses a 3-stage agentic workflow:

**Planner → Generator → Critic**

Powered by Server-Sent Events (SSE), users see real-time progress as:

- The **Planner** formulates a semantic search strategy and vector queries.
- The **Generator** drafts questions using a forced Chain-of-Thought reasoning scratchpad.
- The **Critic** ruthlessly reviews drafts for hallucinations or administrative trivia, triggering automated retry loops for failed questions.

**Quarantine Zone**  
Questions that fail the critic's checks 3 times are moved to a UI *Quarantine Zone* for full transparency.

---

### Production-Ready State Management

- **SQLite Persistence**  
  Job statuses are stored in a local SQLite database (`jobs.db`), ensuring ingestion progress survives process restarts and remains consistent across multiple backend workers.

- **Namespace Isolation**  
  Every document is tagged with a Course ID, acting as a hard filter in Qdrant to prevent cross-contamination between different academic subjects.

---

## Unbuilt Feature Explanation: The AI Tutor

While the Quiz Generator is implemented, the AI Tutor is designed to run on the exact same foundation. Because the Ingestion Engine already processes documents into semantic markdown chunks and stores them in Qdrant with INT8 scalar quantization, the Tutor does not require a separate database or re-ingestion.

### Planned Workflow

- **Query Rewriting**  
  A lightweight router LLM rewrites conversational queries into isolated semantic search queries.  
  _Example_: “How does Week 2 relate to backpropagation?”

- **Two-Stage Retrieval**  
  1. Fetch top broad chunks from Qdrant via vector similarity  
  2. Rerank using a local Cross-Encoder (`ms-marco`) for high-fidelity context

- **Synthesized Generation**  
  A synthesis LLM generates answers using reranked chunks and appends `chunk_id` footnotes, enabling users to trace answers back to original slides.

---

## Assumptions & Limitations

- **Diagrams and Visuals**  
  Academic slides rely heavily on diagrams. Instead of a fully multimodal vector store (expensive), Nexus uses an **Asynchronous Vision Fallback**:

  - `PyMuPDF4LLM` detects images or sparse text
  - Routes them to a Vision LLM
  - Generates dense semantic markdown descriptions for embedding

- **Formulas**  
  Formulas are extracted as LaTeX or markdown math blocks.

  - Works well for standard notation
  - Complex multi-line proofs may lose geometric context when flattened

  **Future Work:**  
  Integrate a math-aware embedding model (e.g., ColBERT) to preserve structural relationships.

---

## Tech Stack

### Backend
- FastAPI (Python 3.12)
- Pydantic (Strict Schema Validation)
- SQLite (State Management)
- AWS Bedrock (Titan, `zai.glm-5` and `moonshotai.kimi-k2.5`)
- PyMuPDF4LLM (Markdown Parsing)

### Frontend
- React 19 + Vite
- Tailwind CSS v4
- Lucide React

### Infrastructure
- Qdrant (Vector DB with INT8 Quantization)
- Docker & Docker Compose

---

## Local Setup & Deployment

##  Quick Start (Docker - Recommended)

The fastest way to get Nexus running locally is via Docker Compose.

1. Clone the repository and navigate into the root directory.

2. Create a `.env` file in the root directory. Docker will automatically pass these to the backend container:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

3. Build and start the orchestration (Note: we use `localhost` for the API base URL during local development):

```
docker compose build --build-arg VITE_API_BASE_URL=http://localhost:8000/api
docker compose up -d
```

4. Navigate to:

* UI: [http://localhost:5173](http://localhost:5173)
* API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Manual Local Development

If you prefer running components individually for debugging:

### 1. AWS Authentication

Ensure AWS credentials are configured securely. You can either:

* Ensure your AWS credentials are secure in your local environment. You can either export the variables from your `.env` file directly into your terminal session, or use the AWS CLI configuration tool:

```bash
aws configure
```

---

### 2. Start Qdrant (Vector Database)

```
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

---

### 3. Start Backend (FastAPI)

```
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 4. Start Frontend (React)

```
cd frontend
npm install
npm run dev
```

---

## Production Deployment (EC2)

When deploying to an AWS EC2 instance:

* You do **not** need to hardcode AWS credentials in a `.env` file.
* Instead, attach an **IAM Role** to the EC2 instance with Bedrock permissions.

Then build the frontend using your public IP:

```
docker compose build --build-arg VITE_API_BASE_URL=http://<YOUR_EC2_PUBLIC_IP>:8000/api

docker compose up -d
```
