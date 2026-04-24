from pydantic import BaseModel, Field
from typing import List

# ==========================================
# PHASE 1: THE PLANNER AGENT
# ==========================================
class SearchPlan(BaseModel):
    vector_queries: List[str] = Field(
        description="2-3 highly specific academic search queries to run against the vector database based on the user prompt."
    )
    generator_instructions: str = Field(
        description="Strict instructions for the Question Generator. Explicitly state what concepts to focus on, and what to ignore."
    )

# ==========================================
# PHASE 2: THE GENERATOR AGENT
# ==========================================
class GeneratedQuestion(BaseModel):
    reasoning_scratchpad: str = Field(
        description="BREATHING ROOM: Before writing the question, explain your thought process. Which chunk are you using? Are you avoiding administrative trivia? How will you make the distractors plausible?"
    )
    question_text: str
    options: List[str] = Field(description="Exactly 4 multiple choice options.")
    correct_answer: str
    explanation: str
    source_chunk_ids: List[str]

class QuarantinedQuestion(BaseModel):
    """Stores a question that failed the Critic's checks after maximum retries."""
    drafted_question: str = Field(description="The text of the rejected question.")
    rejection_reason: str = Field(description="The brutally honest feedback from the Critic Agent explaining why this failed.")

class QuizDraft(BaseModel):
    questions: List[GeneratedQuestion]
    quarantined_questions: List[QuarantinedQuestion] = Field(default_factory=list)

# ==========================================
# PHASE 3: THE CRITIC AGENT
# ==========================================
class CriticReview(BaseModel):
    reasoning_scratchpad: str = Field(
        description="Evaluate the question against the source chunk. Is it testing academic concepts? Is the answer strictly provable using ONLY the provided chunk? Is it free of syllabus/administrative trivia?"
    )
    is_approved: bool = Field(
        description="True if the question passes all quality checks. False if it fails."
    )
    feedback: str = Field(
        description="If is_approved is False, provide a brutally honest, 1-sentence explanation of why it failed so the Generator can try again."
    )