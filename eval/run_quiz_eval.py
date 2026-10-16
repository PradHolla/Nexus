"""End-to-end evaluation harness for the quiz-generation API.

Run:      uv run python eval/run_quiz_eval.py --label <name> [--cases id1,id2] [--no-judge]
Compare:  uv run python eval/run_quiz_eval.py --compare a.json b.json
Rescore:  uv run python eval/run_quiz_eval.py --rescore results.json [--dup-threshold 0.75]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal

import anthropic
import boto3
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
CASES_PATH = EVAL_DIR / "quiz_cases.json"
RESULTS_DIR = EVAL_DIR / "results"

COLLECTION = os.getenv("QDRANT_CHUNKS_COLLECTION", "Nexus_course_materials")
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_CONCURRENCY = 5
DUP_METHOD = "titan>=threshold + haiku confirm"
DIFFICULTIES = ["easy", "medium", "hard"]
JUDGE_BOOLS = [
    "answer_supported_by_sources",
    "single_correct_answer",
    "distractors_plausible",
    "requires_course_material",
]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_quiz(base_url, request, timeout):
    """POST the request and parse the SSE stream. Returns (events, error_reason)."""
    events = []
    error_reason = None
    start = time.monotonic()
    current_event, data_lines = None, []

    def dispatch():
        nonlocal current_event, data_lines
        if current_event is not None or data_lines:
            raw = "\n".join(data_lines)
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = raw
            events.append({
                "t": round(time.monotonic() - start, 3),
                "event": current_event or "message",
                "data": data,
            })
        current_event, data_lines = None, []

    try:
        http_timeout = httpx.Timeout(connect=10, read=timeout, write=30, pool=10)
        with httpx.stream("POST", f"{base_url}/api/generate-quiz", json=request, timeout=http_timeout) as resp:
            if resp.status_code != 200:
                resp.read()
                return events, f"HTTP {resp.status_code}: {resp.text[:500]}"
            for line in resp.iter_lines():
                if time.monotonic() - start > timeout:
                    error_reason = f"timeout after {timeout}s"
                    break
                if line == "":
                    dispatch()
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip(" "))
            dispatch()
    except httpx.TimeoutException as e:
        error_reason = f"timeout: {type(e).__name__}"
    except httpx.HTTPError as e:
        error_reason = f"http error: {type(e).__name__}: {e}"
    return events, error_reason


def attribute_events(events):
    """Return [(question_number, event)] for approved/quarantined/rejected events."""
    current = None
    out = []
    for ev in events:
        data = ev["data"] if isinstance(ev["data"], dict) else {}
        if ev["event"] == "question_started":
            current = data.get("question_number")
        elif ev["event"] in ("approved", "quarantined", "rejected"):
            qn = data.get("question_number", current)
            out.append((qn, ev))
    return out


# ---------------------------------------------------------------------------
# Qdrant lookups and structural checks
# ---------------------------------------------------------------------------

def fetch_chunks(qdrant, chunk_ids):
    """Return {chunk_id: payload} for the chunk ids that exist in Qdrant."""
    chunk_ids = sorted(set(chunk_ids))
    if not chunk_ids:
        return {}
    points, _ = qdrant.scroll(
        COLLECTION,
        scroll_filter=qmodels.Filter(must=[
            qmodels.FieldCondition(key="chunk_id", match=qmodels.MatchAny(any=chunk_ids))
        ]),
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=False,
    )
    return {p.payload["chunk_id"]: p.payload for p in points}


def norm(text):
    return " ".join(str(text).lower().split())


def check_question(q, chunks, course_id, file_filters):
    """Return a list of structural failure strings (empty means valid)."""
    failures = []
    options = q.get("options") or []
    if len(options) != 4:
        failures.append(f"expected 4 options, got {len(options)}")
    if len({norm(o) for o in options}) != len(options):
        failures.append("options are not distinct")
    if q.get("correct_answer") not in options:
        failures.append("correct_answer does not exactly equal an option")
    if not (q.get("explanation") or "").strip():
        failures.append("explanation is empty")
    ids = q.get("source_chunk_ids") or []
    if not ids:
        failures.append("source_chunk_ids is empty")
    for cid in ids:
        payload = chunks.get(cid)
        if payload is None:
            failures.append(f"chunk {cid} not found in Qdrant")
            continue
        if payload.get("course_id") != course_id:
            failures.append(f"chunk {cid} has course_id {payload.get('course_id')}")
        if payload.get("is_administrative"):
            failures.append(f"chunk {cid} is administrative")
        if file_filters and payload.get("file_name") not in file_filters:
            failures.append(f"chunk {cid} file {payload.get('file_name')} not in file_filters")
    return failures


# ---------------------------------------------------------------------------
# Duplicates (Titan embedding candidates, confirmed by Haiku)
# ---------------------------------------------------------------------------

def titan_embed(bedrock, text):
    resp = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def find_candidates(bedrock, questions, threshold):
    texts = ["\n".join([q.get("question_text", ""), *(q.get("options") or [])]) for q in questions]
    embeddings = [titan_embed(bedrock, t) for t in texts]
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine(embeddings[i], embeddings[j])
            if sim >= threshold:
                pairs.append({"i": i, "j": j, "similarity": round(sim, 4)})
    return pairs


class DupVerdict(BaseModel):
    duplicate: bool
    reason: str


DUP_SYSTEM = (
    "You compare two multiple-choice quiz questions from the same quiz. "
    "Answer duplicate=true if they test the same piece of knowledge, so a student who can answer one can answer "
    "the other without learning anything new, even when the wording, framing or options differ. "
    "Answer duplicate=false if they test different facts, concepts or skills, even when they are on the same topic. "
    "reason: one sentence."
)


def confirm_duplicate(client, qa, qb):
    """Return (duplicate_bool_or_None, reason)."""
    def render(label, q):
        options = "\n".join(f"{i}. {o}" for i, o in enumerate(q.get("options") or []))
        return (f"Question {label}: {q.get('question_text')}\nOptions:\n{options}\n"
                f"Marked correct answer: {q.get('correct_answer')}")
    content = render("A", qa) + "\n\n" + render("B", qb)
    try:
        resp = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=1000,
            system=DUP_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_format=DupVerdict,
            output_config={"effort": "low"},
        )
    except anthropic.APIError as e:
        print(f"  duplicate verdict API error: {type(e).__name__}: {e}", file=sys.stderr)
        return None, f"api error: {type(e).__name__}: {e}"
    if resp.stop_reason == "refusal":
        return None, "refusal"
    if resp.parsed_output is None:
        return None, f"no parsed output (stop_reason={resp.stop_reason})"
    return resp.parsed_output.duplicate, resp.parsed_output.reason


def detect_duplicates(bedrock, anthropic_client, questions, threshold):
    """Return (candidates, confirmed_pairs, error). error is set if Titan failed or any verdict is None."""
    try:
        raw = find_candidates(bedrock, questions, threshold)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"  duplicate check failed: {error}", file=sys.stderr)
        return [], [], error

    def verdict(pair):
        dup, reason = confirm_duplicate(anthropic_client, questions[pair["i"]], questions[pair["j"]])
        return {**pair, "duplicate": dup, "reason": reason}
    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as pool:
        candidates = list(pool.map(verdict, raw))

    pairs = [
        {"i": c["i"], "j": c["j"], "similarity": c["similarity"], "reason": c["reason"]}
        for c in candidates if c["duplicate"] is True
    ]
    n_null = sum(c["duplicate"] is None for c in candidates)
    error = f"{n_null} of {len(candidates)} duplicate verdicts unavailable" if n_null else None
    return candidates, pairs, error


def duplicate_rate(pairs, delivered, error):
    in_dup = {p["i"] for p in pairs} | {p["j"] for p in pairs}
    return len(in_dup) / delivered if delivered and error is None else None


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    answer_supported_by_sources: bool
    single_correct_answer: bool
    distractors_plausible: bool
    requires_course_material: bool
    difficulty: Literal["easy", "medium", "hard"]
    notes: str


JUDGE_SYSTEM = """You are a strict evaluator of multiple-choice quiz questions generated from course materials.
Evaluate the question strictly against the provided source text, not against your own background knowledge.

Fill in these fields:
- answer_supported_by_sources: true only if the marked correct answer is clearly supported by the source text.
- single_correct_answer: true only if exactly one option is correct given the source text (no other option is also defensible).
- distractors_plausible: true if the wrong options are plausible to a student who has not mastered the material (not absurd or obviously wrong).
- requires_course_material: false if a knowledgeable person could answer correctly from general knowledge alone, without the course material.
- difficulty: easy = recall/understand, medium = apply, hard = analyze/evaluate.
- notes: one sentence explaining the most important issue or confirming the question is sound."""


def judge_question(client, user_prompt, question, chunks):
    """Return (judge_result_dict_or_None, reason_or_None)."""
    source_text = "\n\n".join(
        f"[{cid}] ({chunks[cid].get('file_name')}, page {chunks[cid].get('page_number')})\n{chunks[cid].get('text', '')}"
        for cid in question.get("source_chunk_ids") or []
        if cid in chunks
    )
    options = "\n".join(f"{i}. {o}" for i, o in enumerate(question.get("options") or []))
    content = (
        f"Quiz request context: {user_prompt}\n\n"
        f"Source text cited by the question:\n{source_text or '(no source text found)'}\n\n"
        f"Question: {question.get('question_text')}\n"
        f"Options:\n{options}\n"
        f"Marked correct answer: {question.get('correct_answer')}\n"
        f"Explanation given: {question.get('explanation')}"
    )
    try:
        resp = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=2000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_format=JudgeResult,
            output_config={"effort": "medium"},
        )
    except anthropic.APIError as e:
        print(f"  judge API error: {type(e).__name__}: {e}", file=sys.stderr)
        return None, f"api error: {type(e).__name__}: {e}"
    if resp.stop_reason == "refusal":
        return None, "refusal"
    if resp.parsed_output is None:
        return None, f"no parsed output (stop_reason={resp.stop_reason})"
    return resp.parsed_output.model_dump(), None


def summarize_judge(results):
    """results: list of {"result": dict|None, "reason": str|None}."""
    valid = [r["result"] for r in results if r["result"] is not None]
    summary = {"n_judged": len(valid), "n_null": len(results) - len(valid)}
    for key in JUDGE_BOOLS:
        summary[key + "_rate"] = sum(r[key] for r in valid) / len(valid) if valid else None
    summary["difficulty"] = {d: sum(r["difficulty"] == d for r in valid) for d in DIFFICULTIES}
    return summary


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def run_case(case, args, qdrant, bedrock, anthropic_client):
    request = {k: case[k] for k in ("course_id", "num_questions", "user_prompt", "keywords", "file_filters")}
    print(f"[{case['id']}] requesting {case['num_questions']} questions ...", flush=True)
    events, error_reason = stream_quiz(args.base_url, request, args.case_timeout)
    total_seconds = events[-1]["t"] if events else 0.0

    complete = next((e["data"] for e in events if e["event"] == "complete"), None)
    error_events = [e["data"] for e in events if e["event"] == "error"]
    if error_events and not error_reason:
        error_reason = f"error event: {error_events[0]}"
    if complete is None and not error_reason:
        error_reason = "no complete event"
    complete = complete if isinstance(complete, dict) else {}

    requested = case["num_questions"]
    delivered_qs = complete.get("questions") or []
    quarantined_qs = complete.get("quarantined_questions") or []
    delivered, quarantined = len(delivered_qs), len(quarantined_qs)

    # Attempts: match each delivered question to its approved event to get its question number.
    attributed = attribute_events(events)
    rejected_counts = {}
    for qn, ev in attributed:
        if ev["event"] == "rejected":
            rejected_counts[qn] = rejected_counts.get(qn, 0) + 1
    approved_by_text = {ev["data"].get("question_text"): qn for qn, ev in attributed if ev["event"] == "approved"}
    question_numbers, attempts = [], []
    for q in delivered_qs:
        qn = approved_by_text.get(q.get("question_text"), q.get("question_number"))
        question_numbers.append(qn)
        attempts.append(1 + rejected_counts.get(qn, 0))

    # Structural validity
    all_ids = [cid for q in delivered_qs for cid in q.get("source_chunk_ids") or []]
    chunks = fetch_chunks(qdrant, all_ids)
    validity = []
    for i, q in enumerate(delivered_qs):
        failures = check_question(q, chunks, case["course_id"], case["file_filters"])
        validity.append({"index": i, "valid": not failures, "failures": failures})

    # Coverage
    files_cited = sorted({chunks[c]["file_name"] for c in set(all_ids) if c in chunks})

    # Duplicates (a Titan or verdict failure is recorded on the case, not fatal to the run)
    dup_candidates, dup_pairs, dup_error = detect_duplicates(bedrock, anthropic_client, delivered_qs, args.dup_threshold)

    first_approved = next((e["t"] for e in events if e["event"] == "approved"), None)
    metrics = {
        "requested": requested,
        "delivered": delivered,
        "quarantined": quarantined,
        "silently_missing": max(0, requested - delivered - quarantined),
        "errored": error_reason is not None,
        "error_reason": error_reason,
        "question_numbers": question_numbers,
        "attempts": attempts,
        "first_attempt_approval_rate": sum(a == 1 for a in attempts) / requested,
        "mean_attempts": sum(attempts) / len(attempts) if attempts else None,
        "validity": validity,
        "valid_rate": sum(v["valid"] for v in validity) / delivered if delivered else None,
        "answer_positions": [
            q["options"].index(q["correct_answer"]) if q.get("correct_answer") in (q.get("options") or []) else None
            for q in delivered_qs
        ],
        "duplicate_candidates": dup_candidates,
        "duplicate_pairs": dup_pairs,
        "duplicate_rate": duplicate_rate(dup_pairs, delivered, dup_error),
        "duplicate_error": dup_error,
        "distinct_files_cited": len(files_cited),
        "distinct_chunks_cited": len(set(all_ids)),
        "files_cited": files_cited,
        "total_seconds": total_seconds,
        "seconds_to_first_approved": first_approved,
        "usage": complete.get("usage"),
    }

    # Judge
    judge = []
    judge_summary = None
    if not args.no_judge and delivered_qs:
        def judge_one(q):
            result, reason = judge_question(anthropic_client, case["user_prompt"], q, chunks)
            return {"result": result, "reason": reason}
        with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as pool:
            judge = list(pool.map(judge_one, delivered_qs))
        judge_summary = summarize_judge(judge)

    return {
        "id": case["id"],
        "request": request,
        "events": events,
        "delivered_questions": delivered_qs,
        "quarantined_questions": quarantined_qs,
        "metrics": metrics,
        "judge": judge,
        "judge_summary": judge_summary,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(case_results):
    ms = [c["metrics"] for c in case_results]
    requested = sum(m["requested"] for m in ms)
    delivered = sum(m["delivered"] for m in ms)
    all_attempts = [a for m in ms for a in m["attempts"]]
    all_validity = [v["valid"] for m in ms for v in m["validity"]]
    n_dup = sum(round(m["duplicate_rate"] * m["delivered"]) for m in ms if m["duplicate_rate"] is not None)
    positions = {str(i): sum(p == i for m in ms for p in m["answer_positions"]) for i in range(4)}
    judged = [j["result"] for c in case_results for j in c["judge"] if j["result"] is not None]
    judge_total = sum(len(c["judge"]) for c in case_results)
    usages = [m["usage"] for m in ms if m["usage"]]

    summary = {
        "n_cases": len(ms),
        "requested": requested,
        "delivered": delivered,
        "quarantined": sum(m["quarantined"] for m in ms),
        "silently_missing": sum(m["silently_missing"] for m in ms),
        "errored_cases": sum(m["errored"] for m in ms),
        "first_attempt_approval_rate": sum(a == 1 for a in all_attempts) / requested if requested else None,
        "mean_attempts": mean(all_attempts),
        "valid_rate": sum(all_validity) / len(all_validity) if all_validity else None,
        "duplicate_rate": n_dup / delivered if delivered else None,
        "answer_position_histogram": positions,
        "mean_distinct_files_cited": mean([m["distinct_files_cited"] for m in ms]),
        "mean_total_seconds": mean([m["total_seconds"] for m in ms]),
        "judge_n_judged": len(judged),
        "judge_n_null": judge_total - len(judged),
        "usage": None,
    }
    for key in JUDGE_BOOLS:
        summary["judge_" + key + "_rate"] = sum(r[key] for r in judged) / len(judged) if judged else None
    for d in DIFFICULTIES:
        summary["judge_difficulty_" + d] = sum(r["difficulty"] == d for r in judged)
    if usages:
        summary["usage"] = {
            k: sum(u.get(k) or 0 for u in usages) for k in ("input_tokens", "output_tokens", "cost_usd")
        }
    return summary


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_summary(case_results, summary):
    header = f"{'case':28} {'req':>4} {'del':>4} {'qua':>4} {'miss':>4} {'err':>4} {'fa_rate':>8} {'valid':>6} {'dup':>6} {'files':>5} {'secs':>7} {'supp':>6}"
    print("\n" + header)
    print("-" * len(header))
    for c in case_results:
        m = c["metrics"]
        supp = c["judge_summary"]["answer_supported_by_sources_rate"] if c["judge_summary"] else None
        print(f"{c['id']:28} {m['requested']:>4} {m['delivered']:>4} {m['quarantined']:>4} {m['silently_missing']:>4} "
              f"{'Y' if m['errored'] else 'n':>4} {fmt(m['first_attempt_approval_rate']):>8} {fmt(m['valid_rate']):>6} "
              f"{fmt(m['duplicate_rate']):>6} {m['distinct_files_cited']:>5} {m['total_seconds']:>7.1f} {fmt(supp):>6}")
        if m["error_reason"]:
            print(f"    error: {m['error_reason']}")
        for v in m["validity"]:
            if not v["valid"]:
                print(f"    q{v['index']} invalid: {'; '.join(v['failures'])}")
    print("\nAggregate")
    for k, v in summary.items():
        print(f"  {k:42} {v if isinstance(v, (dict, str)) else fmt(v)}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def git_info():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def run_mode(args):
    load_dotenv(ROOT / ".env")
    cases = json.loads(CASES_PATH.read_text())
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",")]
        unknown = set(wanted) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in wanted]

    qdrant = QdrantClient(host="localhost", port=6333, check_compatibility=False)
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    anthropic_client = anthropic.Anthropic()

    started = datetime.now()
    git = git_info()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{args.label}-{started.strftime('%Y%m%d-%H%M%S')}.json"

    def save(case_results):
        output = {
            "label": args.label,
            "timestamp": started.isoformat(timespec="seconds"),
            "base_url": args.base_url,
            "git": git,
            "dup_threshold": args.dup_threshold,
            "dup_method": DUP_METHOD,
            "judge_model": None if args.no_judge else JUDGE_MODEL,
            "cases": case_results,
            "summary": aggregate(case_results),
        }
        path.write_text(json.dumps(output, indent=2))
        return output["summary"]

    case_results = []
    for case in cases:
        result = run_case(case, args, qdrant, bedrock, anthropic_client)
        case_results.append(result)
        save(case_results)  # partial results survive a crash mid-run
        m = result["metrics"]
        print(f"[{case['id']}] done: delivered={m['delivered']} quarantined={m['quarantined']} "
              f"errored={m['errored']} {m['total_seconds']:.1f}s", flush=True)

    summary = save(case_results)
    print_summary(case_results, summary)
    print(f"\nWrote {path}")


def short(text, n=100):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def rescore_mode(args):
    load_dotenv(ROOT / ".env")
    in_path = Path(args.rescore)
    data = json.loads(in_path.read_text())
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    anthropic_client = anthropic.Anthropic()

    for case in data["cases"]:
        questions = case["delivered_questions"]
        print(f"[{case['id']}] rescoring {len(questions)} questions ...", flush=True)
        candidates, pairs, error = detect_duplicates(bedrock, anthropic_client, questions, args.dup_threshold)
        m = case["metrics"]
        m["duplicate_candidates"] = candidates
        m["duplicate_pairs"] = pairs
        m["duplicate_rate"] = duplicate_rate(pairs, len(questions), error)
        m["duplicate_error"] = error

    data["summary"] = aggregate(data["cases"])
    data["dup_threshold"] = args.dup_threshold
    data["dup_method"] = DUP_METHOD
    data["rescored_at"] = datetime.now().isoformat(timespec="seconds")
    out_path = in_path.with_name(f"{in_path.stem}-rescored.json")
    out_path.write_text(json.dumps(data, indent=2))

    print_summary(data["cases"], data["summary"])
    print("\nDuplicate candidates")
    for case in data["cases"]:
        m = case["metrics"]
        questions = case["delivered_questions"]
        print(f"[{case['id']}] {len(m['duplicate_candidates'])} candidate(s)"
              + (f"  error: {m['duplicate_error']}" if m["duplicate_error"] else ""))
        for c in m["duplicate_candidates"]:
            verdict = {True: "DUPLICATE", False: "distinct", None: "no verdict"}[c["duplicate"]]
            print(f"  q{c['i']} vs q{c['j']}  sim={c['similarity']:.3f}  {verdict}: {c['reason']}")
            print(f"    q{c['i']}: {short(questions[c['i']].get('question_text'))}")
            print(f"    q{c['j']}: {short(questions[c['j']].get('question_text'))}")
    print(f"\nWrote {out_path}")


def flatten(summary):
    flat = {}
    for k, v in summary.items():
        if k == "answer_position_histogram":
            for pos, n in v.items():
                flat[f"answer_position_{pos}"] = n
        elif k == "usage":
            for uk in ("input_tokens", "output_tokens", "cost_usd"):
                flat[f"usage_{uk}"] = v.get(uk) if v else None
        else:
            flat[k] = v
    return flat


def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return fmt(b - a) if isinstance(a, float) or isinstance(b, float) else str(b - a)
    return "-"


def compare_mode(path_a, path_b):
    a, b = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    print(f"A: {a['label']} ({a['timestamp']}, {a['git']['commit']}{'+dirty' if a['git']['dirty'] else ''})")
    print(f"B: {b['label']} ({b['timestamp']}, {b['git']['commit']}{'+dirty' if b['git']['dirty'] else ''})\n")

    fa, fb = flatten(a["summary"]), flatten(b["summary"])
    print(f"{'metric':46} {'A':>12} {'B':>12} {'delta':>12}")
    print("-" * 85)
    for k in fa:
        print(f"{k:46} {fmt(fa[k]):>12} {fmt(fb.get(k)):>12} {delta(fa[k], fb.get(k)):>12}")

    def case_row(c):
        m, js = c["metrics"], c["judge_summary"]
        return {
            "delivered": m["delivered"],
            "first_attempt": m["first_attempt_approval_rate"],
            "dup_rate": m["duplicate_rate"],
            "valid_rate": m["valid_rate"],
            "supported": js["answer_supported_by_sources_rate"] if js else None,
        }
    ca = {c["id"]: case_row(c) for c in a["cases"]}
    cb = {c["id"]: case_row(c) for c in b["cases"]}
    print("\nPer case (A | B)")
    cols = ["delivered", "first_attempt", "dup_rate", "valid_rate", "supported"]
    print(f"{'case':28} " + " ".join(f"{c:>19}" for c in cols))
    for cid in list(ca) + [c for c in cb if c not in ca]:
        cells = []
        for col in cols:
            va = ca.get(cid, {}).get(col)
            vb = cb.get(cid, {}).get(col)
            cells.append(f"{fmt(va):>8} | {fmt(vb):>8}")
        print(f"{cid:28} " + " ".join(f"{c:>19}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--cases")
    p.add_argument("--case-timeout", type=float, default=900)
    p.add_argument("--dup-threshold", type=float, default=0.75)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    p.add_argument("--rescore", metavar="IN_JSON")
    args = p.parse_args()

    if sum(bool(x) for x in (args.label, args.compare, args.rescore)) != 1:
        p.error("exactly one of --label, --compare or --rescore is required")
    if args.compare:
        compare_mode(*args.compare)
    elif args.rescore:
        rescore_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()
