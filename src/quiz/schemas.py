from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# PHASE 1: THE PLANNER AGENT
# ==========================================
class SearchPlan(BaseModel):
    """
    Translates the professor's raw text prompt into targeted database queries 
    and strict instructions for the generator.
    """
    vector_queries: List[str] = Field(
        description="2-3 highly specific academic search queries to run against the vector database based on the user prompt."
    )
    generator_instructions: str = Field(
        description="Strict instructions for the Question Generator. Explicitly state what concepts to focus on, and what to ignore (e.g., 'Do NOT ask about syllabus, grading, or midterms')."
    )

# ==========================================
# PHASE 2: THE GENERATOR AGENT
# ==========================================
class GeneratedQuestion(BaseModel):
    """
    A single generated question. Includes a reasoning scratchpad to force Chain-of-Thought.
    """
    reasoning_scratchpad: str = Field(
        description="BREATHING ROOM: Before writing the question, explain your thought process. Which chunk are you using? Are you avoiding administrative trivia? How will you make the distractors plausible?"
    )
    question_text: str
    options: List[str] = Field(description="Exactly 4 multiple choice options.")
    correct_answer: str
    explanation: str
    source_chunk_id: str

class QuizDraft(BaseModel):
    questions: List[GeneratedQuestion]

# ==========================================
# PHASE 3: THE CRITIC AGENT
# ==========================================
class CriticReview(BaseModel):
    """
    The output of the Reviewer Agent evaluating a single drafted question.
    """
    reasoning_scratchpad: str = Field(
        description="Evaluate the question against the source chunk. Is it testing academic concepts? Is the answer strictly provable using ONLY the provided chunk? Is it free of syllabus/administrative trivia?"
    )
    is_approved: bool = Field(
        description="True if the question passes all quality checks. False if it is administrative trivia, relies on outside knowledge, or is factually flawed."
    )
    feedback: str = Field(
        description="If is_approved is False, provide a brutally honest, 1-sentence explanation of why it failed so the Generator can try again."
    )