# Scholera AI

Scholera AI is a full-stack, local-first retrieval-augmented generation application designed to synthesize high-difficulty, graduate-level academic assessments from raw course materials. It transforms unstructured PDFs and presentation slides into strictly validated JSON quiz structures.

The system abandons standard single-pass LLM wrappers in favor of a dual-concurrency ingestion engine and a multi-agent orchestration loop. This architecture ensures strict factual grounding, minimizes hallucinations, and provides tenant-isolated semantic search.

## Architecture Highlights

### 1. Dual-Concurrency Ingestion Pipeline
Processing dozens of academic PDFs and slide decks sequentially is a severe bottleneck. Scholera solves this by implementing a dual-concurrency architecture.
* **CPU-Bound Processing:** A ProcessPoolExecutor distributes document parsing across multiple CPU cores, converting raw PDFs into clean Markdown simultaneously.
* **I/O-Bound Processing:** A ThreadPoolExecutor handles network calls, concurrently routing sparse diagrams to a Vision LLM for text extraction and batching academic chunks to AWS Bedrock for Titan embeddings.

### 2. Multi-Tenant Vector Retrieval
The system uses Qdrant for semantic search with strict deterministic boundaries. Every uploaded document is tagged with a Course ID, acting as a hard namespace filter. This prevents the vector database from cross-contaminating contexts, ensuring that an NLP query will never retrieve chunks from a Biology syllabus. It also features dynamic file-level filtering, allowing users to restrict the AI's knowledge base to specific lecture files.

### 3. The 3-Stage Agentic Workflow
To prevent the model from generating lazy questions or relying on its pre-training memory, the generation pipeline is split into three isolated personas managed by the `instructor` library.

* **Phase 1: The Planner**
Instead of relying on rigid UI dropdowns, the Planner Agent interprets a raw user prompt. It extracts the core academic intent, generates specific vector search queries, and outputs strict instructions detailing what the downstream generator should focus on and what it must ignore.

* **Phase 2: The Generator**
Operating strictly within the retrieved Qdrant context, the Generator drafts the assessment. It utilizes a forced Chain of Thought approach via a reasoning scratchpad. The model must explicitly write out its logic, map its distractors, and cite the exact source chunk ID before outputting the final question.

* **Phase 3: The Critic**
The Critic acts as an independent quality control layer. It evaluates the Generator's draft against the source chunks to detect hallucinations, administrative trivia, or poorly phrased options. If a question fails the rubric, the Critic rejects it and feeds a specific failure reason back to the Generator, initiating an automated retry loop.

## Tech Stack

**Backend System**
* FastAPI (Python)
* Pydantic (Strict schema validation)
* Instructor (Structured LLM outputs)
* PyMuPDF / pymupdf4llm (Document parsing)

**Frontend System**
* React 18
* Vite
* Tailwind CSS v4
* Lucide React

**AI Infrastructure**
* Qdrant (Local Docker instance for Vector Search)
* AWS Bedrock (Orchestration and Embeddings)

## Local Development Setup

### Prerequisites
* Docker
* Node.js v20+
* Python 3.10+ (uv recommended)
* AWS CLI configured with Bedrock access

### 1. Start the Vector Database
Initialize a local instance of Qdrant using Docker.
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Start the Backend API
Navigate to the root directory, install dependencies, and start the FastAPI server.
```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Frontend Client
Navigate to the frontend directory, install dependencies, and start the Vite development server.
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`.
