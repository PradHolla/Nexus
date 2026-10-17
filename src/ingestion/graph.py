"""LangGraph ingestion graph: one run per PDF file.

extract -> (fan-out) transcribe x kept page -> analyze -> chunk -> index -> finalize

State holds plain JSON-able dicts only (pydantic models are dumped/validated at node boundaries) so the
checkpointer can persist it. Each finished node and each finished parallel transcribe task is checkpointed,
so a crashed or failed run resumes without redoing completed work.
"""
import asyncio
import collections
import dataclasses
import os
import weakref
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from qdrant_client import QdrantClient

from src.ingestion.analyze import analyze_document
from src.ingestion.chunk import CHUNK_MAX_CHARS, check_chunks, chunk_document
from src.ingestion.extract import RENDER_LONG_EDGE, collapse_builds, extract_document, render_page_png
from src.ingestion.index import ensure_collections, index_document
from src.ingestion.schemas import (
    Chunk,
    DocumentAnalysis,
    ExtractedDocument,
    PageContent,
    PageRecord,
    PageTranscription,
)
from src.ingestion.vision import build_page_content, transcribe_page
from src.job_status_store import append_job_log
from src.llm import LLMError
from src.retry import RETRY, is_transient

INPUT_USD_PER_MTOK = 0.10
OUTPUT_USD_PER_MTOK = 0.50

# Incremented at the start of every node run; tests use it to prove a resume does not re-run finished nodes.
NODE_RUNS: collections.Counter = collections.Counter()

_vision_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class IngestState(TypedDict, total=False):
    job_id: str
    course_id: str
    file_name: str
    file_path: str
    extracted: dict
    kept: list[dict]  # {"page": PageRecord dict, "covered": [int]}
    transcriptions: Annotated[dict, merge_dicts]  # str(page_number) -> {"transcription", "error", "usage"}
    pages: list[dict]  # PageContent dicts
    analysis: dict
    analysis_usage: dict | None
    chunks: list[dict]
    stats: dict


@dataclasses.dataclass(frozen=True)
class IngestContext:
    """Run-scoped dependencies injected into nodes (not checkpointed)."""

    qdrant: QdrantClient


def _vision_semaphore() -> asyncio.Semaphore:
    """Process-wide cap on concurrent vision calls, one semaphore per running event loop."""
    loop = asyncio.get_running_loop()
    sem = _vision_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(int(os.getenv("INGEST_VISION_WORKERS", "8")))
        _vision_semaphores[loop] = sem
    return sem


async def _log(job_id: str, message: str) -> None:
    await asyncio.to_thread(append_job_log, job_id, message)


async def extract(state: IngestState) -> dict:
    NODE_RUNS["extract"] += 1
    extracted = await asyncio.to_thread(extract_document, state["file_path"], state["file_name"])
    kept = collapse_builds(extracted.pages)
    await _log(
        state["job_id"],
        f"Extracted {extracted.page_count} pages; kept {len(kept)} after collapsing "
        f"{extracted.page_count - len(kept)} build pages.",
    )
    return {
        "extracted": extracted.model_dump(),
        "kept": [{"page": page.model_dump(), "covered": covered} for page, covered in kept],
    }


def fan_out(state: IngestState):
    if not state["kept"]:
        return "analyze"
    return [
        Send(
            "transcribe",
            {
                "job_id": state["job_id"],
                "file_path": state["file_path"],
                "page_number": item["page"]["page_number"],
                "text": item["page"]["text"],
            },
        )
        for item in state["kept"]
    ]


async def transcribe(payload: dict) -> dict:
    NODE_RUNS["transcribe"] += 1
    page_number = payload["page_number"]
    png = await asyncio.to_thread(render_page_png, payload["file_path"], page_number, RENDER_LONG_EDGE)
    try:
        async with _vision_semaphore():
            transcription, usage = await transcribe_page(png, payload["text"])
    except (LLMError, anthropic.APIError) as e:
        if is_transient(e):
            raise  # RetryPolicy retries; if retries run out the run fails and can be resumed
        error = f"{type(e).__name__}: {e}"
        await _log(payload["job_id"], f"Page {page_number}: vision failed, using text layer ({error})")
        entry = {"transcription": None, "error": error, "usage": None}
    else:
        entry = {
            "transcription": transcription.model_dump(),
            "error": None,
            "usage": dataclasses.asdict(usage) if usage else None,
        }
    return {"transcriptions": {str(page_number): entry}}


async def analyze(state: IngestState) -> dict:
    NODE_RUNS["analyze"] += 1
    pages: list[PageContent] = []
    for item in state["kept"]:
        page = PageRecord.model_validate(item["page"])
        raw = state.get("transcriptions", {}).get(str(page.page_number), {}).get("transcription")
        transcription = PageTranscription.model_validate(raw) if raw else None
        pages.append(build_page_content(page, item["covered"], transcription))

    usage = None
    try:
        analysis, usage = await analyze_document(state["file_name"], pages)
    except (LLMError, anthropic.APIError) as e:
        if is_transient(e):
            raise
        await _log(state["job_id"], f"Lecture analysis failed, using file name as title ({type(e).__name__}: {e})")
        analysis = DocumentAnalysis(title=Path(state["file_name"]).stem, summary="", topics=[])
    await _log(state["job_id"], f"Analyzed lecture: {analysis.title!r}, {len(analysis.topics)} topics.")
    return {
        "pages": [p.model_dump() for p in pages],
        "analysis": analysis.model_dump(),
        "analysis_usage": dataclasses.asdict(usage) if usage else None,
    }


async def chunk(state: IngestState) -> dict:
    NODE_RUNS["chunk"] += 1
    pages = [PageContent.model_validate(p) for p in state["pages"]]
    analysis = DocumentAnalysis.model_validate(state["analysis"])
    chunks = chunk_document(
        pages,
        course_id=state["course_id"],
        file_name=state["file_name"],
        lecture_number=state["extracted"]["lecture_number"],
        lecture_title=analysis.title,
    )
    violations = check_chunks(pages, chunks, CHUNK_MAX_CHARS)
    if violations:
        raise ValueError(f"Chunking invariants violated for {state['file_name']}: " + "; ".join(violations))
    await _log(state["job_id"], f"Created {len(chunks)} chunks.")
    return {"chunks": [c.model_dump() for c in chunks]}


async def index(state: IngestState, runtime: Runtime[IngestContext]) -> dict:
    NODE_RUNS["index"] += 1
    # Test hook for crash-resume tests: if NEXUS_FAIL_INDEX_ONCE names an existing file, delete it and fail once.
    marker = os.getenv("NEXUS_FAIL_INDEX_ONCE")
    if marker and os.path.exists(marker):
        os.remove(marker)
        raise RuntimeError("injected index failure")

    client = runtime.context.qdrant
    chunks = [Chunk.model_validate(c) for c in state["chunks"]]

    def write() -> dict:
        ensure_collections(client)
        return index_document(
            client,
            course_id=state["course_id"],
            extracted=ExtractedDocument.model_validate(state["extracted"]),
            analysis=DocumentAnalysis.model_validate(state["analysis"]),
            chunks=chunks,
        )

    result = await asyncio.to_thread(write)
    await _log(state["job_id"], f"Indexed {result['chunks']} chunks ({result['quiz_eligible']} quiz-eligible).")
    return {}


async def finalize(state: IngestState) -> dict:
    NODE_RUNS["finalize"] += 1
    entries = list(state.get("transcriptions", {}).values())
    usages = [e["usage"] for e in entries if e.get("usage")]
    if state.get("analysis_usage"):
        usages.append(state["analysis_usage"])
    input_tokens = sum(u["input_tokens"] for u in usages)
    output_tokens = sum(u["output_tokens"] for u in usages)
    vision_ok = sum(1 for e in entries if e.get("transcription"))
    stats = {
        "pages_total": state["extracted"]["page_count"],
        "pages_kept": len(state["kept"]),
        "vision_ok": vision_ok,
        "vision_fallback": len(state["kept"]) - vision_ok,
        "chunks": len(state["chunks"]),
        "chunks_by_page_type": dict(collections.Counter(c["page_type"] for c in state["chunks"])),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(
            input_tokens * INPUT_USD_PER_MTOK / 1e6 + output_tokens * OUTPUT_USD_PER_MTOK / 1e6, 6
        ),
    }
    await _log(
        state["job_id"],
        f"Done: {stats['pages_total']} pages ({stats['pages_kept']} kept), vision {vision_ok} ok / "
        f"{stats['vision_fallback']} fallback, {stats['chunks']} chunks {stats['chunks_by_page_type']}, "
        f"{input_tokens} in / {output_tokens} out tokens, ~${stats['cost_usd']:.4f}.",
    )
    return {"stats": stats}


def build_graph(checkpointer):
    builder = StateGraph(IngestState, context_schema=IngestContext)
    builder.add_node("extract", extract)
    builder.add_node("transcribe", transcribe, retry_policy=RETRY)
    builder.add_node("analyze", analyze, retry_policy=RETRY)
    builder.add_node("chunk", chunk)
    builder.add_node("index", index, retry_policy=RETRY)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", fan_out, ["transcribe", "analyze"])
    builder.add_edge("transcribe", "analyze")
    builder.add_edge("analyze", "chunk")
    builder.add_edge("chunk", "index")
    builder.add_edge("index", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
