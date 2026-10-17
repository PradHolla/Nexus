"""LangGraph quiz pipeline.

Parent graph:  plan -> retrieve -> (fan-out) question x slot -> assemble -> [question x reserve -> assemble] -> finish
Question subgraph (one per slot):  start -> generate -> validate -> critique -> done | quarantine,
with loops back to generate on a validation failure or a critic "revise".

Progress is emitted as custom stream events {"event": name, "data": {...}} (the SSE events the frontend and the
eval harness consume). Subgraph events only reach the parent's astream with subgraphs=True, see stream_quiz.
State holds plain JSON-able dicts only.
"""
import asyncio
import operator
import os
import random
import re
from typing import Annotated, TypedDict

import anthropic
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from src.llm import LLMError, generate_structured
from src.quiz.outline import get_course_outline
from src.quiz.prompts import (
    BLUEPRINT_SYSTEM,
    CRITIQUE_SYSTEM,
    DEDUP_SYSTEM,
    GENERATE_SYSTEM,
    blueprint_content,
    critique_content,
    dedup_content,
    generate_content,
)
from src.quiz.retrieve import assign_chunks
from src.quiz.schemas import (
    Blueprint,
    Critique,
    DraftQuestion,
    DuplicateGroups,
    QuarantinedQuestion,
    QuizQuestion,
    Slot,
    Source,
)
from src.retry import RETRY, is_transient

INPUT_USD_PER_MTOK = 0.10
OUTPUT_USD_PER_MTOK = 0.50

NUM_RESERVE_SLOTS = 2
MAX_ATTEMPTS = 3  # drafts per question: validation failures allowed, and critic revise rounds + 1
MAX_CRITIC_ROUNDS = 2


def _emit(event: str, data: dict) -> None:
    get_stream_writer()({"event": event, "data": data})


def _usage_dict(usage) -> dict:
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


# ---------------------------------------------------------------------------
# Question subgraph
# ---------------------------------------------------------------------------


class SlotState(TypedDict, total=False):
    request: dict
    question_number: int
    slot: dict
    chunks: list[dict]  # payloads, labelled C1..Cn in order
    draft: dict | None  # DraftQuestion dict (options already shuffled once validated)
    feedback: str  # feedback for the next generate call, or the final reason when quarantining
    feedback_kind: str  # "critic" | "validation" | "" (what `feedback` is, for the generate prompt)
    gen_error: str  # set by generate when the LLM call failed non-transiently; validate counts it as a failure
    validation_failures: int
    critic_rounds: int
    status: str  # "pending" | "approved" | "quarantined"
    result: dict | None
    usage: Annotated[list, operator.add]


def _label_index(label: str, n_chunks: int) -> int | None:
    match = re.fullmatch(r"C(\d+)", label.strip().upper())
    if match and 1 <= int(match.group(1)) <= n_chunks:
        return int(match.group(1)) - 1
    return None


def _problems(draft: DraftQuestion, n_chunks: int) -> list[str]:
    problems = []
    if len(draft.options) != 4 or any(not o.strip() for o in draft.options):
        problems.append("There must be exactly 4 non-empty options.")
    elif len({" ".join(o.lower().split()) for o in draft.options}) != 4:
        problems.append("The options must be distinct.")
    if not 0 <= draft.correct_index < 4:
        problems.append("correct_index must be between 0 and 3.")
    if not draft.cited:
        problems.append("cited must list at least one excerpt label.")
    else:
        bad = [c for c in draft.cited if _label_index(c, n_chunks) is None]
        if bad:
            problems.append(f"cited contains labels that are not excerpts: {bad}. Valid labels are C1..C{n_chunks}.")
    return problems


def _cited_chunks(draft: DraftQuestion, chunks: list[dict]) -> list[tuple[str, dict]]:
    seen: list[int] = []
    for label in draft.cited:
        index = _label_index(label, len(chunks))
        if index is not None and index not in seen:
            seen.append(index)
    return [(f"C{i + 1}", chunks[i]) for i in seen]


async def q_start(state: SlotState) -> dict:
    n = state["question_number"]
    _emit("question_started", {"question_number": n, "max_attempts": MAX_ATTEMPTS})
    _emit("log", {"message": f"Q{n}: {state['slot']['topic']} ({state['slot']['difficulty']}, {state['slot']['style']})"})
    if not state["chunks"]:
        return {"status": "quarantined", "feedback": "No matching course material found for this topic"}
    return {}


def route_start(state: SlotState) -> str:
    return "quarantine" if state["status"] == "quarantined" else "generate"


async def q_generate(state: SlotState) -> dict:
    content = generate_content(
        state["request"]["user_prompt"],
        state["slot"],
        state["chunks"],
        draft=state.get("draft"),
        feedback=state.get("feedback", ""),
        feedback_kind=state.get("feedback_kind", ""),
    )
    try:
        result = await generate_structured(
            DraftQuestion, system=GENERATE_SYSTEM, content=content, max_tokens=8000, effort="medium"
        )
    except (LLMError, anthropic.APIError) as e:
        if is_transient(e):
            raise  # RetryPolicy retries
        return {"gen_error": f"{type(e).__name__}: {e}"}
    return {
        "draft": result.output.model_dump(),
        "gen_error": "",
        "usage": [_usage_dict(result.usage)],
    }


async def q_validate(state: SlotState) -> dict:
    n = state["question_number"]
    problems: list[str]
    draft: DraftQuestion | None = None
    if state.get("gen_error"):
        problems = [f"The previous attempt failed: {state['gen_error']}"]
    else:
        draft = DraftQuestion.model_validate(state["draft"])
        problems = _problems(draft, len(state["chunks"]))

    if problems:
        error = " ".join(problems)
        failures = state.get("validation_failures", 0) + 1
        attempt = failures + state.get("critic_rounds", 0)
        _emit("rejected", {"question_number": n, "attempt": attempt, "feedback": error})
        update = {
            "validation_failures": failures,
            "feedback": error,
            "feedback_kind": "validation",
            "gen_error": "",
        }
        if failures >= MAX_ATTEMPTS:
            update["status"] = "quarantined"
            update["feedback"] = f"Could not produce a valid question: {error}"
        return update

    # Shuffle here so every later event shows the final option order.
    options = [o.strip() for o in draft.options]
    correct = options[draft.correct_index]
    random.shuffle(options)
    shuffled = {**draft.model_dump(), "options": options, "correct_index": options.index(correct)}
    return {"draft": shuffled, "feedback": "", "feedback_kind": ""}


def route_validate(state: SlotState) -> str:
    if state["status"] == "quarantined":
        return "quarantine"
    return "generate" if state.get("feedback_kind") == "validation" else "critique"


async def q_critique(state: SlotState) -> dict:
    n = state["question_number"]
    draft = DraftQuestion.model_validate(state["draft"])
    content = critique_content(
        state["request"]["user_prompt"],
        state["slot"],
        draft.model_dump(),
        draft.correct_index,
        _cited_chunks(draft, state["chunks"]),
    )
    try:
        result = await generate_structured(
            Critique, system=CRITIQUE_SYSTEM, content=content, max_tokens=6000, effort="medium"
        )
    except (LLMError, anthropic.APIError) as e:
        if is_transient(e):
            raise
        return {"status": "quarantined", "feedback": f"Review failed: {type(e).__name__}: {e}"}

    critique = result.output
    usage = [_usage_dict(result.usage)]
    verdict, feedback = critique.verdict, critique.feedback
    if verdict == "approve":
        failed = [name for name in ("grounded", "single_correct") if not getattr(critique, name)]
        if failed:
            verdict = "revise"
            feedback = f"{feedback} Failed checks: {', '.join(failed)}.".strip()

    if verdict == "approve":
        return {"status": "approved", "usage": usage}

    rounds = state.get("critic_rounds", 0)
    if verdict == "reject":
        _emit("rejected", {
            "question_number": n, "attempt": state.get("validation_failures", 0) + rounds + 1, "feedback": feedback
        })
        return {"status": "quarantined", "feedback": feedback, "usage": usage}

    rounds += 1
    _emit("rejected", {
        "question_number": n, "attempt": state.get("validation_failures", 0) + rounds, "feedback": feedback
    })
    if rounds > MAX_CRITIC_ROUNDS:
        return {"status": "quarantined", "feedback": feedback, "critic_rounds": rounds, "usage": usage}
    return {"feedback": feedback, "feedback_kind": "critic", "critic_rounds": rounds, "usage": usage}


def route_critique(state: SlotState) -> str:
    if state["status"] == "approved":
        return "done"
    return "quarantine" if state["status"] == "quarantined" else "generate"


async def q_done(state: SlotState) -> dict:
    draft = DraftQuestion.model_validate(state["draft"])
    slot = state["slot"]
    cited = _cited_chunks(draft, state["chunks"])
    question = QuizQuestion(
        question_number=state["question_number"],
        question_text=draft.question_text,
        options=draft.options,
        correct_answer=draft.options[draft.correct_index],
        explanation=draft.explanation,
        source_chunk_ids=[chunk["chunk_id"] for _, chunk in cited],
        sources=[
            Source(
                file_name=chunk["file_name"],
                lecture_number=chunk.get("lecture_number"),
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
            )
            for _, chunk in cited
        ],
        topic=slot["topic"],
        difficulty=slot["difficulty"],
    ).model_dump()
    _emit("approved", question)
    _emit("log", {"message": f"Q{state['question_number']}: approved"})
    return {
        "result": {
            "question_number": state["question_number"],
            "status": "approved",
            "question": question,
            "usage": state.get("usage", []),
        }
    }


async def q_quarantine(state: SlotState) -> dict:
    draft = state.get("draft")
    quarantined = QuarantinedQuestion(
        question_number=state["question_number"],
        drafted_question=draft["question_text"] if draft else "",
        rejection_reason=state.get("feedback", ""),
        topic=state["slot"]["topic"],
    ).model_dump()
    _emit("quarantined", quarantined)
    _emit("log", {"message": f"Q{state['question_number']}: quarantined ({quarantined['rejection_reason'][:200]})"})
    return {
        "status": "quarantined",
        "result": {
            "question_number": state["question_number"],
            "status": "quarantined",
            "quarantined": quarantined,
            "usage": state.get("usage", []),
        },
    }


def build_question_graph():
    builder = StateGraph(SlotState)
    builder.add_node("start", q_start)
    builder.add_node("generate", q_generate, retry_policy=RETRY)
    builder.add_node("validate", q_validate)
    builder.add_node("critique", q_critique, retry_policy=RETRY)
    builder.add_node("done", q_done)
    builder.add_node("quarantine", q_quarantine)

    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", route_start)
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges("validate", route_validate)
    builder.add_conditional_edges("critique", route_critique)
    builder.add_edge("done", END)
    builder.add_edge("quarantine", END)
    return builder.compile()


QUESTION_GRAPH = build_question_graph()


# ---------------------------------------------------------------------------
# Parent graph
# ---------------------------------------------------------------------------


class QuizState(TypedDict, total=False):
    request: dict  # course_id, num_questions, user_prompt, keywords, file_filters, difficulty
    outline: list[dict]
    slots: list[dict]  # {question_number, slot, chunks}
    reserve: list[dict]  # same shape
    results: Annotated[list, operator.add]  # {question_number, status, question | quarantined, usage}
    round: int
    usage: Annotated[list, operator.add]


def _qdrant(config: RunnableConfig):
    return config["configurable"]["qdrant"]


def _question_sends(state: QuizState, items: list[dict]) -> list[Send]:
    return [
        Send(
            "question",
            {
                "request": state["request"],
                "question_number": item["question_number"],
                "slot": item["slot"],
                "chunks": item["chunks"],
            },
        )
        for item in items
    ]


def _fail(message: str) -> dict:
    _emit("error", {"message": message})
    return {"slots": [], "reserve": []}


async def plan(state: QuizState, config: RunnableConfig) -> dict:
    request = state["request"]
    _emit("planner_started", {})

    outline = await asyncio.to_thread(
        get_course_outline, _qdrant(config), request["course_id"], request.get("file_filters")
    )
    if not outline:
        suffix = " and file filters" if request.get("file_filters") else ""
        return _fail("No ingested lectures found for this course" + suffix)

    num = request["num_questions"]
    content = blueprint_content(
        request["user_prompt"], request.get("keywords"), request["difficulty"], num, NUM_RESERVE_SLOTS, outline
    )
    try:
        result = await generate_structured(
            Blueprint, system=BLUEPRINT_SYSTEM, content=content, max_tokens=8000, effort="medium"
        )
    except (LLMError, anthropic.APIError) as e:
        if is_transient(e):
            raise
        return _fail(f"Quiz planning failed: {type(e).__name__}: {e}")

    valid_files = {d["file_name"] for d in outline}

    def clean(slot: Slot) -> Slot:
        return slot.model_copy(update={"target_files": [f for f in slot.target_files if f in valid_files]})

    slots = [clean(s) for s in result.output.slots][:num]
    reserves = [clean(s) for s in result.output.reserve_slots]
    if len(slots) < num:
        top_up = num - len(slots)
        slots += reserves[:top_up]
        reserves = reserves[top_up:]
    reserves = reserves[:NUM_RESERVE_SLOTS]
    if not slots:
        return _fail("Quiz planning failed: the planner returned no questions")

    slot_items = [
        {"question_number": i, "slot": s.model_dump(), "chunks": []} for i, s in enumerate(slots, 1)
    ]
    reserve_items = [
        {"question_number": len(slots) + i, "slot": s.model_dump(), "chunks": []}
        for i, s in enumerate(reserves, 1)
    ]
    _emit("planner_complete", {"vector_queries": [s.search_query for s in slots]})
    for item in slot_items:
        s = item["slot"]
        _emit("log", {
            "message": f"Slot {item['question_number']}: {s['difficulty']} {s['style']} - {s['topic']} "
            f"[{', '.join(s['target_files'])}]"
        })
    return {
        "outline": outline,
        "slots": slot_items,
        "reserve": reserve_items,
        "usage": [_usage_dict(result.usage)],
    }


def route_after_plan(state: QuizState) -> str:
    return "retrieve" if state.get("slots") else END


async def retrieve(state: QuizState, config: RunnableConfig) -> dict:
    request = state["request"]
    _emit("retrieval_started", {})
    items = state["slots"] + state["reserve"]  # main slots first, so they get first pick of chunks
    assigned = await asyncio.to_thread(
        assign_chunks,
        _qdrant(config),
        request["course_id"],
        [Slot.model_validate(i["slot"]) for i in items],
        request.get("file_filters"),
    )
    filled = [{**item, "chunks": chunks} for item, chunks in zip(items, assigned)]
    if not any(assigned):
        return _fail("No course materials found for this request.")

    n_main = len(state["slots"])
    _emit("retrieval_complete", {"chunk_count": sum(len(c) for c in assigned)})
    _emit("generation_started", {"num_questions": request["num_questions"]})
    return {"slots": filled[:n_main], "reserve": filled[n_main:]}


def fan_out(state: QuizState):
    if not state.get("slots"):
        return END
    return _question_sends(state, state["slots"])


async def question(payload: dict, config: RunnableConfig) -> dict:
    out = await QUESTION_GRAPH.ainvoke(
        {
            **payload,
            "draft": None,
            "feedback": "",
            "feedback_kind": "",
            "gen_error": "",
            "validation_failures": 0,
            "critic_rounds": 0,
            "status": "pending",
            "result": None,
            "usage": [],
        },
        config,
    )
    return {"results": [out["result"]], "usage": out["usage"]}


def _delivered(state: QuizState) -> list[dict]:
    """Approved questions that were not later moved out as duplicates."""
    duplicates = {r["question_number"] for r in state.get("results", []) if r["status"] == "duplicate"}
    return sorted(
        (r["question"] for r in state.get("results", []) if r["status"] == "approved" and r["question_number"] not in duplicates),
        key=lambda q: q["question_number"],
    )


async def assemble(state: QuizState):
    request = state["request"]
    num = request["num_questions"]
    approved = _delivered(state)
    new_results: list[dict] = []
    usage: list[dict] = []

    if len(approved) >= 2:
        try:
            result = await generate_structured(
                DuplicateGroups,
                system=DEDUP_SYSTEM,
                content=dedup_content(approved),
                max_tokens=3000,
                effort="low",
            )
        except (LLMError, anthropic.APIError) as e:
            if is_transient(e):
                raise
            _emit("log", {"message": f"Duplicate check skipped: {type(e).__name__}: {e}"})
        else:
            usage.append(_usage_dict(result.usage))
            by_number = {q["question_number"]: q for q in approved}
            removed: set[int] = set()
            for group in result.output.groups:
                numbers = sorted({n for n in group if n in by_number and n not in removed})
                if len(numbers) < 2:
                    continue
                kept = numbers[0]
                for n in numbers[1:]:
                    removed.add(n)
                    quarantined = QuarantinedQuestion(
                        question_number=n,
                        drafted_question=by_number[n]["question_text"],
                        rejection_reason=f"Duplicate of question {kept}",
                        topic=by_number[n]["topic"],
                    ).model_dump()
                    _emit("quarantined", quarantined)
                    new_results.append(
                        {"question_number": n, "status": "duplicate", "quarantined": quarantined, "usage": []}
                    )
            approved = [q for q in approved if q["question_number"] not in removed]

    update: dict = {"results": new_results, "usage": usage}
    reserves = [r for r in state.get("reserve", []) if r["chunks"]]
    if len(approved) < num and state.get("round", 0) == 0 and reserves:
        shortfall = num - len(approved)
        _emit("log", {"message": f"{shortfall} question(s) short, drafting from reserve slots"})
        return Command(update={**update, "round": 1}, goto=_question_sends(state, reserves[:shortfall]))
    return Command(update=update, goto="finish")


async def finish(state: QuizState) -> dict:
    request = state["request"]
    results = state.get("results", [])
    questions = _delivered(state)[: request["num_questions"]]
    quarantined = sorted(
        (r["quarantined"] for r in results if r["status"] in ("quarantined", "duplicate")),
        key=lambda q: q["question_number"],
    )
    input_tokens = sum(u["input_tokens"] for u in state.get("usage", []))
    output_tokens = sum(u["output_tokens"] for u in state.get("usage", []))
    _emit("complete", {
        "questions": questions,
        "quarantined_questions": quarantined,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(
                input_tokens * INPUT_USD_PER_MTOK / 1e6 + output_tokens * OUTPUT_USD_PER_MTOK / 1e6, 6
            ),
        },
        "difficulty": request["difficulty"],
    })
    return {}


def build_quiz_graph():
    builder = StateGraph(QuizState)
    builder.add_node("plan", plan, retry_policy=RETRY)
    builder.add_node("retrieve", retrieve, retry_policy=RETRY)
    builder.add_node("question", question)  # transient errors are retried per node inside the subgraph
    builder.add_node("assemble", assemble, retry_policy=RETRY, destinations=("question", "finish"))
    builder.add_node("finish", finish)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("retrieve", fan_out)
    builder.add_edge("question", "assemble")
    builder.add_edge("finish", END)
    return builder.compile()


QUIZ_GRAPH = build_quiz_graph()


async def stream_quiz(client, request: dict):
    """Run the quiz pipeline, yielding (event_name, data) tuples. Unexpected exceptions become a final error event."""
    config = {
        "max_concurrency": int(os.getenv("QUIZ_MAX_CONCURRENCY", "6")),
        "recursion_limit": 100,
        "configurable": {"qdrant": client},
    }
    initial = {"request": request, "outline": [], "slots": [], "reserve": [], "results": [], "round": 0, "usage": []}
    try:
        # subgraphs=True is required for custom events written inside the question subgraph to reach this stream;
        # chunks then arrive as (namespace_tuple, payload).
        async for _namespace, chunk in QUIZ_GRAPH.astream(
            initial, config, stream_mode="custom", subgraphs=True
        ):
            yield chunk["event"], chunk["data"]
    except Exception as e:
        yield "error", {"message": str(e)}
