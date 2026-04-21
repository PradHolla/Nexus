import os
import boto3
import instructor
from typing import List, Dict, Any

# Import our new agent brains
from src.quiz.schemas import SearchPlan, QuizDraft, GeneratedQuestion, CriticReview

bedrock_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
client = instructor.from_bedrock(bedrock_client)

AGENT_MODEL_ID = "zai.glm-5"

# ==========================================
# AGENT 1: THE PLANNER
# ==========================================
def run_planner_agent(user_prompt: str) -> SearchPlan:
    """Translates the user's raw prompt into vector queries and strict generation instructions."""
    print("[Planner Agent] Analyzing intent and formulating search plan...")
    
    system_prompt = """
    You are the Orchestration Agent for an academic RAG system.
    Your job is to read the professor's request and output a precise JSON plan.
    1. Extract 2-3 specific academic concepts to query the vector database.
    2. Write strict instructions for the downstream Question Generator, explicitly noting what to avoid (e.g., admin trivia).
    
    CRITICAL: You MUST return ONLY raw, valid JSON. DO NOT include any conversational text.
    
    EXAMPLE OUTPUT:
    {
        "vector_queries": ["LSTM backpropagation", "Transformer attention"],
        "generator_instructions": "Focus heavily on the math behind vanishing gradients. Do NOT ask about the midterm or textbook."
    }
    """
    
    return client.messages.create(
        model=AGENT_MODEL_ID,
        max_tokens=1024,
        temperature=0.4, # <--- Bumping this introduces enough entropy to break the exclamation point loop
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Professor's Request: {user_prompt}"}
        ],
        response_model=SearchPlan
    )

# ==========================================
# AGENT 2: THE GENERATOR
# ==========================================
def run_generator_agent(
    instructions: str, 
    num_questions: int, 
    context_chunks: List[Dict[str, Any]], 
    previous_feedback: str = ""
) -> QuizDraft:
    """Drafts questions using the Planner's instructions and the retrieved Qdrant chunks."""
    print(f"[Generator Agent] Drafting {num_questions} questions...")
    
    context_string = ""
    for chunk in context_chunks:
        context_string += f"\n--- Start Chunk: {chunk.get('chunk_id')} ---\n{chunk.get('text', '')}\n--- End Chunk ---\n"

    system_prompt = f"""
    You are an expert academic assessment writer.
    
    PLANNER INSTRUCTIONS: {instructions}
    
    CRITICAL RULES:
    1. Generate exactly {num_questions} questions.
    2. Use ONLY the provided chunks. Provide the exact source_chunk_id.
    3. Use the 'reasoning_scratchpad' to think step-by-step before outputting the question.
    
    {previous_feedback}
    """

    return client.messages.create(
        model=AGENT_MODEL_ID,
        max_tokens=4096,
        temperature=0.3, # Slight creativity for plausible distractors
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT CHUNKS:\n{context_string}"}
        ],
        response_model=QuizDraft
    )

# ==========================================
# AGENT 3: THE CRITIC
# ==========================================
def run_critic_agent(question: GeneratedQuestion, chunk_text: str) -> CriticReview:
    """Evaluates a single drafted question to ensure it isn't syllabus trivia or a hallucination."""
    system_prompt = """
    You are a ruthless Academic Reviewer Agent. Your job is to catch bad LLM-generated questions.
    
    FAIL THE QUESTION IF:
    1. It asks about course administration, syllabus details, exam dates, or textbook names.
    2. The correct answer cannot be 100% proven by reading the provided chunk.
    3. It is trivially easy or poorly phrased.
    
    Be brutally honest in your feedback.
    """
    
    prompt = f"""
    SOURCE CHUNK: {chunk_text}
    
    DRAFTED QUESTION: {question.question_text}
    OPTIONS: {question.options}
    ANSWER: {question.correct_answer}
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