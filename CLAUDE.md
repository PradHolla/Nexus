# CLAUDE.md

Nexus: RAG quiz generator (FastAPI + SSE backend, Qdrant, React/Vite frontend). Take-home for Scholera's AI Engineer role — brief in `docs/ai-engineer.md`.

## Working rules

- **Don't assume — verify.** Before using any library API, model ID, SDK parameter, config field or CLI flag, confirm it against official docs, the installed package source (`.venv`), or the code in this repo. If you can't verify it, say so instead of guessing. Applies to plans as much as code.
- **Delegate grunt coding to the `implementer` subagent** (`.claude/agents/implementer.md`, Sonnet 5.5, high effort). It does not read this file, so every delegation must be a self-contained spec: files to touch, exact behavior, interfaces/signatures, constraints, and how to verify. Review every diff it produces before accepting it; you own correctness.
- **Tests: end-to-end only.** No mock-heavy or trivial unit tests. Tests exercise the real pipeline against real services (Qdrant, Bedrock Titan, Anthropic API) and assert on meaningful outcomes. Existing mock-based tests in `tests/` are legacy.

## Environment

- Python via `uv` (`uv sync`, `uv run ...`). Run API: `uv run uvicorn main:app --port 8000 --reload`. Qdrant: `docker compose up -d qdrant`. Runtime state (jobs/checkpoint DBs, caches, uploads) lives in `var/` (gitignored).
- LLM: Claude Haiku 5.5 (`claude-haiku-5-5`) via the `anthropic` SDK; `ANTHROPIC_API_KEY` is in `.env`.
- Embeddings: local `ibm-granite/granite-embedding-small-english-r2` via `fastembed` (384-dim, CPU, use `batch_size=8`). Replaces Titan; the app has no AWS dependency once the old pipeline is removed.
- Never print or commit `.env` contents.

## Current focus

Ingestion rework first (`docs/plans/ingestion-plan.md`), then one wipe + re-ingest of `nlp_stanford/`, then the quiz pipeline redesign (`docs/plans/quiz-pipeline-plan.md`). PDF only for now — PowerPoint support is dropped. Commit and push straight to `main`; no branches.
