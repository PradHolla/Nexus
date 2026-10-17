"""Run the new PDF ingestion core on real PDFs and write a human-readable preview (no Qdrant).

Usage: uv run python scripts/ingest_preview.py --pdf PATH [--pdf PATH ...] --course CS224
       [--out DIR] [--concurrency 8] [--no-vision]
"""
import argparse
import asyncio
import dataclasses
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # before importing src.* so config picks up env overrides

from src.config import ensure_var_dirs  # noqa: E402
from src.ingestion.analyze import analyze_document  # noqa: E402
from src.ingestion.chunk import CHUNK_MAX_CHARS, check_chunks, chunk_document  # noqa: E402
from src.ingestion.extract import collapse_builds, extract_document, render_page_png  # noqa: E402
from src.ingestion.schemas import DocumentAnalysis, PageContent  # noqa: E402
from src.ingestion.vision import build_page_content, transcribe_page  # noqa: E402
from src.llm import LLMError, LLMUsage  # noqa: E402

INPUT_USD_PER_MTOK = 0.10
OUTPUT_USD_PER_MTOK = 0.50


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * INPUT_USD_PER_MTOK / 1e6 + output_tokens * OUTPUT_USD_PER_MTOK / 1e6


def usage_dict(usage: LLMUsage | None) -> dict | None:
    return dataclasses.asdict(usage) if usage else None


async def process_pdf(pdf: Path, course: str, out_root: Path, concurrency: int, no_vision: bool) -> dict:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    t = time.perf_counter()
    doc = await asyncio.to_thread(extract_document, str(pdf), pdf.name)
    timings["extract"] = time.perf_counter() - t

    t = time.perf_counter()
    kept = collapse_builds(doc.pages)
    timings["collapse"] = time.perf_counter() - t

    errors: dict[int, str] = {}
    usages: dict[int, LLMUsage | None] = {}
    sem = asyncio.Semaphore(concurrency)

    async def vision_one(page):
        async with sem:
            try:
                png = await asyncio.to_thread(render_page_png, str(pdf), page.page_number)
                transcription, usage = await transcribe_page(png, page.text)
            except (LLMError, anthropic.APIError) as e:
                errors[page.page_number] = f"{type(e).__name__}: {e}"
                print(f"  [{pdf.name}] page {page.page_number} vision failed: {errors[page.page_number]}", file=sys.stderr)
                return None
            usages[page.page_number] = usage
            return transcription

    t = time.perf_counter()
    if no_vision:
        transcriptions = [None] * len(kept)
    else:
        transcriptions = await asyncio.gather(*(vision_one(page) for page, _ in kept))
    timings["vision"] = time.perf_counter() - t

    contents: list[PageContent] = [
        build_page_content(page, covered, tr) for (page, covered), tr in zip(kept, transcriptions)
    ]

    t = time.perf_counter()
    analysis_usage: LLMUsage | None = None
    analysis_error: str | None = None
    try:
        analysis, analysis_usage = await analyze_document(pdf.name, contents)
    except (LLMError, anthropic.APIError) as e:
        analysis_error = f"{type(e).__name__}: {e}"
        print(f"  [{pdf.name}] analysis failed: {analysis_error}", file=sys.stderr)
        analysis = DocumentAnalysis(title=pdf.stem, summary="", topics=[])
    timings["analyze"] = time.perf_counter() - t

    t = time.perf_counter()
    chunks = chunk_document(
        contents,
        course_id=course,
        file_name=pdf.name,
        lecture_number=doc.lecture_number,
        lecture_title=analysis.title,
    )
    timings["chunk"] = time.perf_counter() - t
    violations = check_chunks(contents, chunks, CHUNK_MAX_CHARS)
    total_time = time.perf_counter() - t_start

    # ---- outputs
    out = out_root / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    by_number = {c.page_number: c for c in contents}
    collapsed_into = {n: c.page_number for c in contents for n in c.covered_pages if n != c.page_number}
    pages_json = []
    for p in doc.pages:
        row: dict = {"page": p.page_number}
        c = by_number.get(p.page_number)
        if c is None:
            row.update(status="collapsed", collapsed_into=collapsed_into[p.page_number])
        else:
            row.update(
                status="kept",
                covered_pages=c.covered_pages,
                page_type=c.page_type,
                title=c.title,
                used_vision=c.used_vision,
                has_math=c.has_math,
                text_length=len(c.text),
                usage=usage_dict(usages.get(p.page_number)),
                vision_error=errors.get(p.page_number),
            )
        pages_json.append(row)
    (out / "pages.json").write_text(json.dumps(pages_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")

    all_usage = [u for u in usages.values() if u] + ([analysis_usage] if analysis_usage else [])
    in_tok = sum(u.input_tokens for u in all_usage)
    out_tok = sum(u.output_tokens for u in all_usage)
    cost = cost_usd(in_tok, out_tok)
    vision_ok = sum(1 for c in contents if c.used_vision)
    fallback = len(contents) - vision_ok
    lens = sorted(c.char_count for c in chunks) or [0]
    p90 = lens[min(len(lens) - 1, int(0.9 * len(lens)))]
    empty_pages = [c.page_number for c in contents if not c.text.strip()]
    groups = [c.covered_pages for c in contents if len(c.covered_pages) > 1]

    lines = [
        f"# Ingest preview: {pdf.name}",
        "",
        f"- course: {course}, lecture_number: {doc.lecture_number}, file_hash: {doc.file_hash[:16]}...",
        "",
        "## Pages",
        f"- total: {doc.page_count}",
        f"- kept: {len(contents)}",
        f"- collapsed: {doc.page_count - len(contents)} in {len(groups)} group(s): {groups if groups else 'none'}",
        f"- by page_type: {dict(Counter(c.page_type for c in contents))}",
        f"- vision ok: {vision_ok}, fallback: {fallback}",
        f"- empty-text kept pages (no chunk): {empty_pages or 'none'}",
    ]
    if no_vision:
        lines.append("- vision disabled (--no-vision): every page uses the native text fallback")
    if errors:
        lines.append("- vision errors:")
        lines += [f"  - page {n}: {msg}" for n, msg in sorted(errors.items())]
    lines += [
        "",
        "## Chunks",
        f"- count: {len(chunks)}",
        f"- chars min/median/p90/max: {lens[0]} / {int(statistics.median(lens))} / {p90} / {lens[-1]}",
        f"- by page_type: {dict(Counter(c.page_type for c in chunks))}",
        "",
        "## check_chunks violations",
    ]
    lines += [f"- {v}" for v in violations] or ["none"]
    lines += [
        "",
        "## Analysis",
        f"- title: {analysis.title}",
        f"- summary: {analysis.summary}",
        "- topics:",
    ]
    lines += [f"  - {tp}" for tp in analysis.topics]
    if analysis_error:
        lines.append(f"- analysis error: {analysis_error}")
    uncached_pages = sum(1 for u in usages.values() if u)
    lines += [
        "",
        "## Token usage (uncached calls only)",
        f"- LLM calls: {uncached_pages} page calls + {1 if analysis_usage else 0} analysis call",
        f"- input tokens: {in_tok}, output tokens: {out_tok}",
        f"- estimated cost: ${cost:.4f} (at ${INPUT_USD_PER_MTOK}/MTok in, ${OUTPUT_USD_PER_MTOK}/MTok out)",
        "",
        "## Wall time",
    ]
    lines += [f"- {k}: {v:.2f}s" for k, v in timings.items()]
    lines += [f"- total: {total_time:.2f}s", "", "## Chunk texts"]
    for c in chunks:
        lines += ["", f"### {c.chunk_id}", f"({c.page_type}, {c.char_count} chars, math={c.has_math}, topic={c.topic!r})", "", c.text]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "pdf": pdf.name,
        "pages": doc.page_count,
        "kept": len(contents),
        "vision_ok": vision_ok,
        "fallback": fallback,
        "chunks": len(chunks),
        "violations": len(violations),
        "llm_calls": uncached_pages + (1 if analysis_usage else 0),
        "tokens": f"{in_tok} in / {out_tok} out",
        "cost": f"${cost:.4f}",
        "wall": f"{total_time:.1f}s",
    }
    print("  " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="append", required=True, type=Path)
    parser.add_argument("--course", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "var" / "preview")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()

    ensure_var_dirs()
    for pdf in args.pdf:
        print(f"{pdf.name}:")
        await process_pdf(pdf, args.course, args.out, args.concurrency, args.no_vision)


if __name__ == "__main__":
    asyncio.run(main())
