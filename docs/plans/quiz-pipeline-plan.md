# Quiz Pipeline Redesign — Plan

> **Order changed (2026-10-15):** the ingestion rework (`ingestion-plan.md`) runs first and re-ingests once; this plan follows on the final data. The Haiku client (Phase 1 below) is built during ingestion step 1.

Scope: quiz generation only. Switch the LLM calls to Claude Haiku 5.5, rebuild the pipeline as a LangGraph graph, and fix the quality problems in the design itself. Ingestion fixes and the re-ingest come later.

Verified facts this plan relies on (checked 2026-10-15):
- All Python and frontend dependencies upgraded to latest (done 2026-10-15). `anthropic` 1.15.0 and `langgraph` 1.2.15 are now direct dependencies.
- Haiku 5.5 is `claude-haiku-5-5`. It rejects non-default `temperature`/`top_p`/`top_k` and assistant prefill with a 400. Thinking is adaptive, effort defaults to `medium` (`output_config={"effort": ...}`), and responses can end with `stop_reason == "refusal"`, with no server-side fallback.
- Structured output: `client.messages.parse(..., output_format=PydanticModel)` → `.parsed_output`.
- `python-dotenv` is a dependency but `load_dotenv()` is never called. (`instructor` was unused and has been removed.)
- Frontend (`QuizTab.jsx`) handles these events: `planner_started`, `planner_complete`, `retrieval_started`, `retrieval_complete`, `generation_started`, `approved`, `quarantined`, `complete`, `error`, plus `log` in the log panel. It displays these question fields: `question_text`, `options`, `correct_answer`, `explanation`, `reasoning_scratchpad`, `source_chunk_ids`; and for quarantined questions, `drafted_question`, `rejection_reason`.

## Target flow

```
START → build_outline → plan_blueprint → retrieve_slots ──Send per slot──▶ question subgraph ──▶ assemble → END
                                                                                ▲                  │
                                                                                └── reserve slots ─┘ (top-up if short)

question subgraph:
  generate → validate (code) ─fail→ generate  (code failures: max 2, don't use critic strikes)
                 │ ok
              critique ─┬─ approve  → done
                        ├─ revise   → generate (with previous draft + feedback)
                        ├─ resample → swap in reserve chunks → generate
                        └─ reject / 3 critic strikes / unrecoverable error → quarantine
```

## Phases

Each phase gets an explicit spec for the `implementer` subagent. I review every diff and run the verification myself before moving on.

### Phase 0 — Setup and baseline
1. ~~Dependencies~~ (done: upgraded to latest; `anthropic` and `langgraph` added; `pytest` in the `dev` group; ragas/datasets/langchain-aws in the `eval` group; `instructor` removed).
2. Call `load_dotenv()` at the top of `main.py`, before any client is created.
3. Find the ingested `course_id`(s) in the local Qdrant and choose the eval course.
4. **Eval harness** `eval/run_quiz_eval.py` (cases in `eval/quiz_cases.json`): sends about 8 fixed requests (varied prompts, file filters, question counts) to `POST /api/generate-quiz`, parses the SSE stream, and writes metrics to JSON:
   - first-attempt approval rate, quarantine rate, delivered count vs. requested
   - duplicate rate (Titan cosine similarity between question stems)
   - coverage (distinct files/lectures/chunks used)
   - answer-position distribution
   - latency, input/output tokens, cost
   - groundedness, scored by an LLM judge (Haiku 5.5)
5. Run it against the **current** pipeline (Bedrock GLM-5) → `eval/results/baseline-*.json`.

### Phase 1 — LLM layer (`src/llm.py`)
- One `AsyncAnthropic` client, `MODEL = "claude-haiku-5-5"`, model name overridable by env var.
- `async structured_call(system, user, schema, effort, max_tokens) -> schema instance`, built on `messages.parse`:
  - Put `cache_control` on the system block.
  - Raise `LLMRefusal` on `stop_reason == "refusal"` and `LLMTruncated` on `"max_tokens"`.
  - Return the token usage alongside the result, for the eval.
- No temperature, no prefill. Effort per role: blueprint `medium`, generator `medium`, critic `medium`, judge `low`. We'll tune these with the eval.

### Phase 2 — Schemas (`src/quiz/schemas.py`)
- `QuestionSlot{slot_id, topic, search_query, difficulty, style, target_files?}` and `QuizBlueprint{slots, reserve_slots, generator_notes}`.
- `GeneratedQuestion`:
  - Drop `reasoning_scratchpad`, since Haiku thinks natively.
  - The model returns `options[4]`, `correct_index` and `cited_chunks` as short aliases (`C1`…`C6`). Code maps the aliases back to real chunk IDs and fills in `correct_answer`, so the API response shape stays the same for the frontend.
  - Add `difficulty`, `topic` and `sources[{file_name, page_number, lecture_number}]`.
- `CriticReview`:
  - Per-criterion booleans: grounded, single_correct, distractors_plausible, no_answer_leak, matches_difficulty, matches_request, not_administrative.
  - `verdict: approve|revise|resample|reject`, plus `feedback`.
- `QuizRequest` gets `difficulty: easy|medium|hard|mixed = "medium"`. Each level is defined in the prompts using Bloom's levels: easy = recall/understand, medium = apply, hard = analyze/evaluate.

### Phase 3 — Course outline and retrieval per slot (`src/retrieval/sampler.py`)
- `get_course_outline(course_id, file_filters)`: scroll payloads only (no vectors) and aggregate into `{file → lecture_number, topics[]}`, limited to a fixed size. This is the context the planner currently lacks.
- `retrieve_for_slots(slots, ...)`:
  - One node, sequential over slots, so chunks can be assigned **exclusively** before the fan-out.
  - Each slot gets 4 primary chunks plus up to 4 reserve chunks (used by `resample`).
  - Reuse the existing hybrid/dense search and reranker.
  - Drop chunks below a minimum text length or a rerank score cutoff (both env-configurable).
  - Blocking calls (Qdrant, Titan, FlashRank) run in `asyncio.to_thread`.
- Fix the reranker timeout. The `with ThreadPoolExecutor` exit blocks until the rerank finishes, so the timeout never takes effect.
- Remove the fake "stratified" fallback. The outline-driven blueprint replaces it.

### Phase 4 — Graph (`src/quiz/graph.py`; old functions removed from `generator.py`)
- **Parent `QuizState`** (TypedDict):
  - request fields, `outline`, `blueprint`, `slot_contexts`
  - `results: Annotated[list[SlotResult], operator.add]`
  - `round`, `questions`, `quarantined`
- **Question subgraph `SlotState`**: `slot`, `chunks`, `reserve`, `draft`, `feedback`, `critic_strikes`, `validation_failures`, `status`.
- **Nodes:**
  - `generate`: on a revision, it receives the previous draft plus the feedback and revises rather than starting over.
  - `validate` (pure code): exactly 4 distinct options, a valid `correct_index`, cited aliases that resolve, no placeholder text.
  - `critique`: sees the original professor request, the slot spec, the draft, the explanation and the cited chunks.
  - `quarantine`.
- **Retries:**
  - `RetryPolicy(max_attempts=3, retry_on=<anthropic connection/429/5xx/timeout errors>)` on the LLM nodes. Check the default `retry_on` in the installed source before relying on it.
  - If an error persists after retries, the node catches it and routes to `quarantine` with a system-error reason. A question never silently disappears.
- **Assemble:**
  - Drop near-duplicate stems (Titan cosine ≥ 0.9, env-configurable) and record them as quarantined with reason "duplicate".
  - Shuffle option order in code.
  - If delivered < requested and reserve slots remain, run **one** top-up round via `Send`.
  - Emit `complete` with the questions in slot order.
- **Parallelism:** `config={"max_concurrency": QUIZ_MAX_CONCURRENCY}` (default 5).
- **Streaming:** nodes emit the existing SSE event names through `get_stream_writer()`. `planner_complete.vector_queries` carries the slot queries. Events from different slots arrive interleaved; the frontend already handles that because it appends.
- No checkpointer. Quiz runs are short and repeatable, so persistence isn't worth it here. We'll revisit for ingestion and human review.

### Phase 5 — API (`main.py`)
- `/api/generate-quiz` streams `graph.astream(..., stream_mode="custom")` and maps each chunk to SSE. The sync generators and `isinstance` juggling go away, and the event loop no longer blocks.
- `retrieval_mode` and `include_retrieval_diagnostics` pass through to the retrieval node unchanged.

### Phase 6 — Frontend (minimal)
- Add a difficulty selector.
- Remove the `reasoning_scratchpad` display. Show sources as file / lecture / page instead of raw chunk IDs, with a topic/difficulty badge.
- Change the default prompt so it no longer says "highly difficult".

### Phase 7 — E2E tests and eval comparison
- `tests/e2e/test_quiz_pipeline.py` runs against the real app (ASGI client), real Qdrant, Titan and Haiku. It skips with a clear message if a service or the ingested course is missing. It asserts:
  - a `complete` event arrives
  - approved + quarantined accounts for every requested slot
  - every question has 4 distinct options, `correct_answer` is one of them, and every source chunk exists in Qdrant and respects `course_id`/`file_filters`
  - no duplicate stems
  - the `difficulty` is echoed back
- Re-run the eval → `eval/results/after-*.json` and compare with the baseline.

### Phase 8 (optional) — Solver check
Add it only if the eval shows bad questions the critic misses. Two extra Haiku calls per question: one open-book (an answer that disagrees with the key → revise) and one closed-book (correct without the chunks on a hard slot → revise), behind a request flag.

## Defaults I've chosen (change if you disagree)
- Difficulty: easy/medium/hard/mixed in the UI, defined internally by Bloom's levels.
- The eval and baseline come first (Phase 0), so improvements are measured, not claimed.
- The solver check waits until after the eval.
- Out of scope for now: ingestion (Kimi vision and Titan stay on Bedrock), `scripts/evaluate_rag.py`, the AI Tutor, checkpointing.

## Ingestion
Moved to `ingestion-plan.md`, which now runs before this plan.
