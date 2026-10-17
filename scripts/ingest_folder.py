"""Ingest every PDF in a folder through the ingestion graph and print a report.

Usage: uv run python scripts/ingest_folder.py --course CS224 --dir nlp_stanford [--exclude NAME ...] [--wipe] [--yes]
"""
import argparse
import asyncio
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # before importing src.* so config picks up env overrides

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models  # noqa: E402

from src.config import CHECKPOINTS_DB_PATH, CHUNKS_COLLECTION, DOCUMENTS_COLLECTION, UPLOADS_DIR, ensure_var_dirs  # noqa: E402
from src.ingestion.graph import build_graph  # noqa: E402
from src.ingestion.index import ensure_collections  # noqa: E402
from src.ingestion.runner import IngestionRunner  # noqa: E402
from src.job_status_store import create_job, get_job_status_and_logs, init_db  # noqa: E402

TOKENS_RE = re.compile(r"(\d+) in / (\d+) out tokens, ~\$([\d.]+)\.")


def course_count(client: QdrantClient, collection: str, course: str) -> int:
    flt = models.Filter(must=[models.FieldCondition(key="course_id", match=models.MatchValue(value=course))])
    return client.count(collection, count_filter=flt, exact=True).count


def wipe_collections(client: QdrantClient, assume_yes: bool) -> None:
    existing = [c for c in (CHUNKS_COLLECTION, DOCUMENTS_COLLECTION) if client.collection_exists(c)]
    for name in existing:
        print(f"Will delete collection {name} ({client.count(name, exact=True).count} points)")
    if not existing:
        return
    if not assume_yes and input("Type 'wipe' to confirm: ").strip() != "wipe":
        sys.exit("Aborted, nothing changed.")
    for name in existing:
        client.delete_collection(name)


def job_summary(job_id: str) -> tuple[str, list[str]]:
    status, logs = get_job_status_and_logs(job_id)
    return status, [line for line in logs.split("\n") if line.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", required=True)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[], metavar="NAME")
    parser.add_argument("--wipe", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    ensure_var_dirs()
    init_db()
    client = QdrantClient(f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}")

    pdfs = sorted(args.dir.glob("*.pdf"), key=lambda p: p.name)
    missing = [n for n in args.exclude if n not in {p.name for p in pdfs}]
    if missing:
        sys.exit(f"Error: --exclude names not found in {args.dir}: {missing}")
    pdfs = [p for p in pdfs if p.name not in args.exclude]
    print(f"{len(pdfs)} PDFs to ingest into course {args.course}:")
    for pdf in pdfs:
        print(f"  {pdf.name}")

    if args.wipe:
        wipe_collections(client, args.yes)
    ensure_collections(client)

    start = time.perf_counter()
    jobs: dict[str, str] = {}  # job_id -> file name
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS_DB_PATH)) as checkpointer:
        runner = IngestionRunner(build_graph(checkpointer), client)
        for pdf in pdfs:
            job_id = str(uuid.uuid4())
            upload_path = UPLOADS_DIR / f"{job_id}_{pdf.name}"  # the runner deletes this copy on success
            shutil.copyfile(pdf, upload_path)
            create_job(job_id, args.course, pdf.name, str(upload_path))
            runner.submit(job_id, args.course, pdf.name, str(upload_path))
            jobs[job_id] = pdf.name

        reported: set[str] = set()
        waiter = asyncio.create_task(runner.wait_idle())
        while not waiter.done():
            await asyncio.wait([waiter], timeout=10)
            statuses = {j: job_summary(j)[0] for j in jobs}
            finished = [j for j, s in statuses.items() if s == "completed" or s.startswith("failed")]
            failed = sum(1 for j in finished if statuses[j] != "completed")
            running = sum(1 for s in statuses.values() if s == "processing")
            print(f"[{time.perf_counter() - start:.0f}s] completed {len(finished) - failed}, failed {failed}, running {running}")
            for j in finished:
                if j not in reported:
                    reported.add(j)
                    print(f"  finished: {jobs[j]} ({statuses[j].split(':')[0]})")
        await waiter

    print("\n=== Report ===")
    ok = failed = in_tok = out_tok = 0
    cost = 0.0
    for job_id, name in jobs.items():
        status, lines = job_summary(job_id)
        if status == "completed":
            ok += 1
            done = next((line for line in reversed(lines) if line.startswith("Done:")), "(no Done line)")
            print(f"{name}: {status}\n  {done}")
            if m := TOKENS_RE.search(done):
                in_tok, out_tok, cost = in_tok + int(m[1]), out_tok + int(m[2]), cost + float(m[3])
        else:
            failed += 1
            print(f"{name}: {status}")
            for line in lines[-5:]:
                print(f"  {line}")

    print(f"\nFiles: {ok} ok, {failed} failed")
    print(f"Chunks in {CHUNKS_COLLECTION} for {args.course}: {course_count(client, CHUNKS_COLLECTION, args.course)}")
    print(f"Documents in {DOCUMENTS_COLLECTION} for {args.course}: {course_count(client, DOCUMENTS_COLLECTION, args.course)}")
    print(f"Tokens: {in_tok} in / {out_tok} out, ~${cost:.4f}")
    print(f"Wall time: {time.perf_counter() - start:.1f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
