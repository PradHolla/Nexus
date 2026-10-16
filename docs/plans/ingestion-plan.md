# Ingestion Rework — Plan

Goal: fix ingestion completely, then wipe and re-ingest **once**. Runs before the quiz pipeline redesign (see `quiz-pipeline-plan.md`).

Decided:
- Ingestion becomes a LangGraph graph with a checkpointer.
- Generate a summary per lecture at ingestion.
- Keep only `lecture14-agents-shikhar-updated.pdf`.
- Embeddings stay on Titan v2.

## Verified facts this plan relies on (2026-10-15)

**Current data**
- 360 chunks over 25 files.
- At least 10% of chunks sit exactly at the 8,000-char cap; the text past the cap is discarded (`embedder.py:101-107`).
- 22/25 files are tagged lecture 224.
- 0 chunks are flagged administrative.
- Collection format is a legacy unnamed dense vector, so hybrid search is unavailable.

**Titan v2** accepts 8,192 tokens / 50,000 characters ([AWS docs](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)).

**pymupdf4llm 1.28.2**
- It takes the layout path by default.
- `use_ocr` defaults to `True`, and `OCRMode` ranges from `NEVER` to `FORCE_*`.
- It OCRs about 82 of 84 pages on the sample deck. 1.27.2.2 did the same, so this is not a regression.

**LangGraph 1.2.15**, tested with a script:
- `invoke(None, config)` with the same `thread_id` resumes after a node failure.
- Completed parallel `Send` tasks and upstream nodes are **not** re-run; only the failed task re-runs.
- `durability` is `"sync" | "async" | "exit"`.
- `RetryPolicy` defaults: 3 attempts, 0.5 s initial interval, backoff ×2, jitter.
- Its default `retry_on` retries every exception *except* ValueError, TypeError, LookupError, RuntimeError, OSError and a few others. So a 400 from Anthropic would be retried, which means we need our own `retry_on`.

**qdrant-client 1.19.1**: `create_payload_index(collection_name, field_name, field_schema)` and `delete(collection_name, points_selector)`.

**Frontend contract**
- `POST /api/ingest?course_id=` (multipart `files`) returns `{jobs: [{filename, job_id}]}`.
- The UI polls `GET /api/jobs/{id}` and expects `{status, logs[]}`.
- The UI treats `completed` or a `failed…` prefix as terminal.

**PowerPoint today**: `.ppt` is converted to `.pptx` via LibreOffice (installed in the Docker image, **not** on this Mac); `.pptx` goes through python-pptx.

## Target design

One graph run per file: `thread_id = job_id`, checkpointer = `AsyncSqliteSaver` (`langgraph-checkpoint-sqlite`).

```
START → extract ──Send per visual page──▶ describe_page ─┐
            └────────(no visual pages)──────────────────┴─▶ analyze_document → chunk → index → finalize → END
```

| Node | What it does |
|---|---|
| `extract` | Runs in the process pool, since it's CPU-heavy. `pymupdf4llm.to_markdown(page_chunks=True)` produces one record per page: `{page_number, text, needs_vision, has_math}`, plus the file hash and lecture number. The pool is **rebuilt automatically** after `BrokenProcessPool`, and the error is re-raised so `RetryPolicy` retries. |
| `describe_page` | One parallel task per visual page. It renders the page at `PDF_RENDER_DPI` and calls Haiku 5.5 vision with structured output `{description, topic}`. Results are cached on disk by image hash, model and prompt version. If it fails permanently, the page gets **no** description, and that is logged; failure text is never written into chunks. |
| `analyze_document` | One Haiku call per file with all page texts and visual descriptions, marked by page. Structured output: `{title, summary, topics[], admin_pages[]}`. This replaces keyword-based admin detection and produces the lecture summary. |
| `chunk` | Pure code (details below). |
| `index` | Titan embeddings, cached on disk by a hash of the text, with bounded threads. Failures **raise** and are retried. It then deletes the file's existing points (`course_id` + `file_name`), upserts chunks in batches, and upserts the lecture summary into the documents collection. It's safe to re-run: point IDs are deterministic (uuid5 of a stable `chunk_id`) and old points are deleted first. Vectors never go into graph state. |
| `finalize` | Marks the job completed with stats and deletes the uploaded file. |

**Retries:** `RetryPolicy(max_attempts=3, retry_on=transient)` on `describe_page`, `analyze_document` and `index`. `transient` covers: Anthropic connection, timeout, 429, 5xx and overloaded errors; botocore throttling and service-unavailable `ClientError`s and connection errors; Qdrant connection errors.

**State** contains only file paths, page text, descriptions, metadata and chunk text. It never holds image bytes or vectors, so checkpoints stay small.

### Chunking (`chunk` node)
- Work in page order over all pages. Each page's text has its visual description appended (`[Figure] …`) when one exists.
- Group consecutive pages until a target size is reached, and start a new chunk at a page that has a header.
- Split a page that's too long at paragraph, then sentence boundaries, with a small overlap.
- Merge a page that's too short into its neighbour.
- **Never truncate.** If a chunk is still over the limit, split it again.
- Admin pages become chunks too, flagged `is_administrative=True`. The quiz already filters them out, and the future Tutor can answer "when is the exam?".
- `chunk_id = f"{course_id}:{file_name}:p{start}-{end}:{idx}"`.
- Payload:
  - `chunk_id, course_id, file_name, lecture_number, lecture_title`
  - `page_start, page_end, page_number` (= page_start, for compatibility)
  - `topic, is_administrative, has_math, used_vision, text, char_count`
- The size targets (default guess: target 1,500, minimum 400, maximum 2,500 characters) are **set from the dry-run report**, not guessed.

### Metadata
- **Lecture number:** the regex `lecture[\s_-]*0*(\d+)`, case-insensitive. If that doesn't match, a leading `^(\d+)`. Otherwise `null`. Never the first digit run in the name, which is how `cs224n` became 224.
- **Documents collection `Nexus_course_documents`:** one point per file, holding a dense Titan vector of the summary plus `{course_id, file_name, lecture_number, title, summary, topics, page_count, chunk_count, file_hash, ingested_at}`. It feeds the quiz planner's course outline and, later, the Tutor's lecture routing.
- **Chunk collection `Nexus_course_materials`:**
  - Created in hybrid format: named `dense` vector (1024, cosine, INT8 quantization) plus `sparse` BM25 with IDF.
  - Payload indexes on `course_id`, `file_name`, `is_administrative` and `lecture_number`.
  - Collection names come from env vars, so tests use their own collections.

### Jobs, recovery, API
- **Runtime state moves to `var/`** (gitignored and dockerignored; compose mounts `./var:/app/var`): `jobs.db`, `checkpoints.db`, `cache/`, `uploads/`.
- **Uploads** are written to `var/uploads/` instead of `/tmp`, so a resumed job can still find its file.
- **`JobStatus`** gains `course_id`, `file_name` and `file_path` columns. The API contract doesn't change (`queued` / `processing` / `completed` / `failed: …`).
- **Running jobs:** files are processed concurrently under a semaphore (`INGEST_MAX_FILES`, default 2). Parallel vision calls are capped with `max_concurrency` (`INGEST_VISION_WORKERS`).
- **Startup recovery:** for each job still `queued` or `processing`:
  - a checkpoint exists → resume it with `invoke(None)`;
  - otherwise, the upload still exists → start the job over;
  - otherwise → `failed: interrupted`.
- **Event loop:** ingestion no longer blocks the server, because blocking calls run in threads or the process pool.
- **New endpoint:** `GET /api/courses/{course_id}/documents` lists the lecture summaries.
- **PowerPoint: dropped for now** (user decision, 2026-10-15). Ingestion accepts PDF only; `POST /api/ingest` rejects `.ppt`/`.pptx` with a 400 and the frontend dropzone accepts PDF only. The python-pptx path and LibreOffice conversion are removed. It can be added back later by converting to PDF in front of the same graph.
- **Old code removed:** `parse_pdf`/`parse_ppt`, the Kimi vision calls and the old embedder flow are replaced. The quiz side's `get_titan_embedding` stays, now raising on errors.

## Steps
Each step is a self-contained spec for the `implementer`. I review each diff, run the verification myself, then commit and push to `main`.

1. **Foundation**
   - `src/llm.py`: `AsyncAnthropic` with an async structured-output helper (text and image blocks; refusal and truncation errors; usage returned).
   - Env-configurable collection names.
   - The `var/` runtime directory.
   - Add the `langgraph-checkpoint-sqlite` dependency (verify the `AsyncSqliteSaver` import path in the installed package).
2. **Extraction and chunking, without LLM calls**, plus `scripts/ingest_folder.py --dry-run`:
   - Runs extract and chunk over a folder.
   - Reports per-file page counts, visual-page counts, the chunk size distribution and sample chunks.
   - `--ocr on|off` compares text yield and timing.
   - **Checkpoint:** I review the report with you and we fix the chunk sizes and the OCR setting.
3. **LLM nodes:** `describe_page` and `analyze_document` with their caches.
4. **Graph, checkpointer, runner, job store, startup recovery, API wiring**, the documents collection and payload indexes.
5. **`scripts/ingest_folder.py`** (real mode: `--course`, `--dir`, `--exclude`, `--wipe`), plus **end-to-end tests** in `tests/e2e/test_ingestion.py` against real Qdrant (test collections), Titan and Haiku, using a 6-page PDF cut from `data/samples/1-introduction.pdf` that includes a visual page:
   - The job completes through `POST /api/ingest`.
   - No chunk exceeds the size limit, and no text is lost (chunk text covers the extracted text).
   - Every non-empty page appears in some chunk's page range.
   - The lecture summary exists.
   - Re-ingesting the same file leaves the point count unchanged.
   - Crash-resume: a one-time injected failure in `index`, then a resume completes without new vision calls (served from the cache).
6. **Wipe and re-ingest**:
   - Snapshot already taken (`data/qdrant-snapshots/`).
   - Delete both collections.
   - Ingest the 21 PDFs in `nlp_stanford/` (excluding the non-updated lecture 14) as `CS224`.
   - Check data stats: no chunk at a cap, correct lecture numbers, admin pages flagged, 21 summaries.
7. **Eval checkpoint:** run the old quiz pipeline on the new data (`--label new-data`) and compare it with the baseline.
