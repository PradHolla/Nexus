"""System prompts and user-content builders for the quiz pipeline."""
from src.quiz.schemas import DraftQuestion

BLUEPRINT_SYSTEM = """You plan a multiple-choice quiz for a university course. You get the professor's request, optional keywords, the requested difficulty, the number of questions, and the course outline: one entry per lecture file with its title, summary and topics.
Produce exactly the requested number of slots, plus the requested number of reserve slots (reserves are used only if a question fails).
Rules:
- Each slot tests one distinct concept. No two slots, reserves included, may test the same concept or ask the same thing in different words.
- Follow the request's scope. If it names topics or lectures, stay within them. If it is broad (the whole course, a midterm), spread the slots across the lectures in scope instead of clustering on one or two.
- target_files: the 1-2 file names, copied exactly from the outline, whose material the question should come from.
- search_query: a specific 5-15 word query that will retrieve the slide content for this concept, in the course's terminology.
- difficulty: if the requested difficulty is easy, medium or hard, use it for every slot; if mixed, use roughly a third of each. easy = recall or explain a definition or fact from the slides; medium = apply a method, equation or idea to a concrete case; hard = analyze or compare, reason about trade-offs, or diagnose a failure.
- style: vary styles across slots.
- Never plan questions about course logistics (deadlines, grading, assignments, staff)."""

GENERATE_SYSTEM = """You write one multiple-choice question for a university course quiz from the provided slide excerpts (labelled C1, C2, ...).
Requirements:
- Test the slot's concept at the slot's difficulty, in the slot's style. Build the question mainly on excerpt C1; use the other excerpts for supporting detail.
- Difficulty levels: easy = recall or explain a definition or fact from the slides; medium = apply a method, equation or idea to a concrete case or example the question describes; hard = analyze or compare, reason about trade-offs, or diagnose why something fails. A medium or hard question must require reasoning, not just recognizing a phrase from the slides.
- The question must be answerable from the excerpts, and the correct answer must be clearly supported by them. Prefer what is specific to this course: the slides' definitions, equations, examples, results and claims, so a student who studied the lecture has an advantage over someone relying on general knowledge.
- Exactly 4 options and exactly one correct. Distractors must be plausible to a student who has not mastered the material (common misconceptions, near-miss variants of the correct idea) but unambiguously wrong according to the excerpts. Keep options similar in length and style. Never use "all of the above", "none of the above" or combinations of options.
- Do not give away the answer in the question.
- Do not ask about the slides themselves (slide numbers, layout, figure colors) or about course logistics.
- Use LaTeX ($...$) for math where needed.
- correct_index: 0-based index of the correct option.
- explanation: 1-3 sentences on why the correct option is right, grounded in the excerpts.
- cited: the labels of the excerpts that support the answer."""

CRITIQUE_SYSTEM = """You review one multiple-choice question generated for a university course quiz. You get the professor's request, the slot the question was planned for, the question with its marked answer and explanation, and the text of the slide excerpts it cites. Judge only against the cited excerpts, not your own knowledge.
- grounded: the marked answer is clearly supported by the cited excerpts.
- single_correct: exactly one option is correct; no other option is defensible from the excerpts.
- distractors_plausible: the wrong options are plausible to a student who has not mastered the material, not absurd.
- requires_course_material: false if someone with general knowledge could easily answer without having studied this lecture.
- matches_difficulty: the question fits the slot's difficulty (easy = recall/understand, medium = apply, hard = analyze/evaluate).
- no_answer_leak: the question does not give away the answer.
verdict: "approve" if all checks pass (requires_course_material and matches_difficulty may be slightly imperfect for an otherwise excellent question); "revise" if editing the question can fix the issues; "reject" if the excerpts cannot support a good question for this slot.
feedback: specific, actionable revision instructions; empty if approved."""

DEDUP_SYSTEM = """You check a quiz for duplicate questions. Two questions are duplicates if a student who can answer one can answer the other without learning anything new, even if they are worded differently. Return groups of question numbers that duplicate each other (each group has at least 2 numbers). Return no groups if there are none."""


def blueprint_content(
    user_prompt: str,
    keywords: list[str] | None,
    difficulty: str,
    num_questions: int,
    num_reserve: int,
    outline: list[dict],
) -> str:
    lectures = "\n\n".join(
        f"### {d['file_name']}"
        + (f" (lecture {d['lecture_number']})" if d["lecture_number"] is not None else "")
        + f": {d['title']}\n{d['summary']}\nTopics: {'; '.join(d['topics'])}"
        for d in outline
    )
    return (
        f"Professor's request: {user_prompt}\n"
        f"Keywords: {', '.join(keywords) if keywords else 'none'}\n"
        f"Difficulty: {difficulty}\n"
        f"Number of questions: {num_questions}\n"
        f"Number of reserve slots: {num_reserve}\n\n"
        f"Course outline:\n\n{lectures}"
    )


def _slot_text(slot: dict) -> str:
    return f"topic: {slot['topic']}\ndifficulty: {slot['difficulty']}\nstyle: {slot['style']}"


def _excerpt(label: str, chunk: dict) -> str:
    return f'<excerpt label="{label}">{chunk["text"]}</excerpt>'


def _question_text(question: str, options: list[str], correct_index: int | None, lettered: bool) -> str:
    # The marked answer goes on its own line, not inside the option text, so reviewers do not read the marker as part
    # of the option and flag it as an answer leak.
    lines = [f"{chr(65 + i) if lettered else i}. {option}" for i, option in enumerate(options)]
    marked = chr(65 + correct_index) if lettered else correct_index
    return f"Question: {question}\nOptions:\n" + "\n".join(lines) + f"\nMarked correct answer: {marked}"


def generate_content(
    user_prompt: str,
    slot: dict,
    chunks: list[dict],
    draft: dict | None = None,
    feedback: str = "",
    feedback_kind: str = "",
) -> str:
    """feedback_kind is "critic" (reviewer feedback on the previous draft) or "validation" (code validation error)."""
    excerpts = "\n\n".join(_excerpt(f"C{i}", c) for i, c in enumerate(chunks, 1))
    parts = [
        f"Slot:\n{_slot_text(slot)}",
        f"Professor's request: {user_prompt}",
        f"Excerpts:\n{excerpts}",
    ]
    if feedback:
        if draft is not None:
            previous = DraftQuestion.model_validate(draft)
            parts.append(
                "<previous_draft>\n"
                + _question_text(previous.question_text, previous.options, previous.correct_index, lettered=False)
                + f"\nExplanation: {previous.explanation}\n</previous_draft>"
            )
        if feedback_kind == "validation":
            parts.append(f"<validation_error>\n{feedback}\n</validation_error>")
            parts.append("Fix this and return a valid question.")
        else:
            parts.append(f"<reviewer_feedback>\n{feedback}\n</reviewer_feedback>")
            parts.append("Revise the question to fix every issue in the feedback.")
    return "\n\n".join(parts)


def critique_content(
    user_prompt: str, slot: dict, question: dict, correct_index: int, cited_chunks: list[tuple[str, dict]]
) -> str:
    """question holds question_text/options/explanation; cited_chunks is [(label, chunk payload)]."""
    excerpts = "\n\n".join(_excerpt(label, chunk) for label, chunk in cited_chunks)
    return (
        f"Professor's request: {user_prompt}\n\n"
        f"Slot:\n{_slot_text(slot)}\n\n"
        + _question_text(question["question_text"], question["options"], correct_index, lettered=True)
        + f"\nExplanation: {question['explanation']}\n\n"
        f"Cited excerpts:\n{excerpts}"
    )


def dedup_content(questions: list[dict]) -> str:
    return "\n\n".join(
        f"Question {q['question_number']}: {q['question_text']}\nCorrect answer: {q['correct_answer']}"
        for q in questions
    )
