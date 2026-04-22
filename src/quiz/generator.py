import os
import boto3
import instructor
from typing import List, Dict, Any

from src.quiz.schemas import SearchPlan, QuizDraft, GeneratedQuestion, QuarantinedQuestion, CriticReview

bedrock_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
client = instructor.from_bedrock(bedrock_client)

AGENT_MODEL_ID = "moonshotai.kimi-k2.5" # Or "moonshotai.kimi-k2.5"

# ==========================================
# AGENT 1: THE PLANNER
# ==========================================
# ==========================================
# AGENT 1: THE PLANNER
# ==========================================
def run_planner_agent(user_prompt: str) -> SearchPlan:
    print("[Planner Agent] Analyzing intent and formulating search plan...")
    
    # STRIPPED PROMPT: No JSON instructions, no examples. Let `instructor` do the heavy lifting.
    system_prompt = """You are the Orchestration Agent for an academic RAG system.
Your job is to read the professor's request and formulate a precise search plan.
1. Extract 2-3 specific academic concepts to query the vector database.
2. Write strict instructions for the downstream Question Generator, explicitly noting what to avoid.
"""
    
    return client.messages.create(
        model=AGENT_MODEL_ID,
        max_tokens=1024,
        temperature=0.1, 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Professor's Request: {user_prompt}"}
        ],
        response_model=SearchPlan
    )

# ==========================================
# AGENT 2: THE GENERATOR (Drafts a SINGLE question)
# ==========================================
def run_generator_agent(
    instructions: str, 
    context_chunks: List[Dict[str, Any]], 
    previous_feedback: str = ""
) -> GeneratedQuestion:
    
    context_string = ""
    for chunk in context_chunks:
        context_string += f"<chunk id='{chunk.get('chunk_id')}'>\n{chunk.get('text', '')}\n</chunk>\n"

    system_prompt = f"""
    <system_role>
    You are an expert academic assessment writer.
    </system_role>
    
    <planner_instructions>
    {instructions}
    </planner_instructions>
    
    <rules>
    1. Generate exactly 1 highly difficult question based ONLY on the provided chunks.
    2. Provide the exact source_chunk_id.
    3. Use the 'reasoning_scratchpad' to think step-by-step.
    </rules>
    """

    user_prompt = f"<context>\n{context_string}</context>\n"
    if previous_feedback:
        user_prompt += f"\n<critic_feedback>\nYOUR PREVIOUS ATTEMPT FAILED. FIX THIS:\n{previous_feedback}\n</critic_feedback>"

    return client.messages.create(
        model=AGENT_MODEL_ID,
        max_tokens=2048,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_model=GeneratedQuestion
    )

# ==========================================
# AGENT 3: THE CRITIC
# ==========================================
def run_critic_agent(question: GeneratedQuestion, chunk_text: str) -> CriticReview:
    system_prompt = """
    <system_role>
    You are a ruthless Academic Reviewer Agent. Your job is to catch bad LLM-generated questions.
    </system_role>
    
    <rules>
    FAIL THE QUESTION IF:
    1. It asks about course administration, syllabus details, exam dates, or textbook names.
    2. The correct answer cannot be 100% proven by reading the provided chunk.
    3. It is trivially easy or poorly phrased.
    Be brutally honest in your feedback.
    </rules>
    """
    
    prompt = f"""
    <source_chunk>
    {chunk_text}
    </source_chunk>
    
    <drafted_question>
    Question: {question.question_text}
    Options: {question.options}
    Answer: {question.correct_answer}
    </drafted_question>
    """

    return client.messages.create(
        model=AGENT_MODEL_ID,
        max_tokens=1024,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_model=CriticReview
    )

# ==========================================
# ORCHESTRATION: The 3-Strike Loop
# ==========================================
def generate_validated_quiz(plan: SearchPlan, chunks: List[Dict[str, Any]], num_questions: int) -> QuizDraft:
    """Manages the Generator-Critic loop and populates the Quarantine Zone."""
    approved_questions = []
    quarantined_questions = []
    
    print(f"\n[Orchestrator] Beginning generation of {num_questions} questions...")

    for i in range(num_questions):
        print(f"  -> Drafting Question {i+1}...")
        attempts = 0
        max_attempts = 3
        feedback = ""
        question_draft = None

        while attempts < max_attempts:
            # 1. Generate
            question_draft = run_generator_agent(plan.generator_instructions, chunks, feedback)
            
            # Find the matching text for the critic
            source_text = next((c.get("text") for c in chunks if c.get("chunk_id") == question_draft.source_chunk_id), "Source chunk not found.")
            
            # 2. Critique
            review = run_critic_agent(question_draft, source_text)
            
            if review.is_approved:
                print(f"Approved on attempt {attempts + 1}")
                approved_questions.append(question_draft)
                break
            else:
                attempts += 1
                feedback = review.feedback
                print(f"Rejected (Attempt {attempts}/{max_attempts}): {feedback}")

        # 3. Quarantine if it failed 3 times
        if attempts == max_attempts and question_draft:
            print("Quarantined: Maximum retries reached.")
            quarantined_questions.append(
                QuarantinedQuestion(
                    drafted_question=question_draft.question_text,
                    rejection_reason=feedback
                )
            )

    return QuizDraft(questions=approved_questions, quarantined_questions=quarantined_questions)