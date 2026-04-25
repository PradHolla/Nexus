import os
import boto3
import json
import re
from typing import List, Dict, Any, Generator
import random
from src.quiz.schemas import SearchPlan, GeneratedQuestion, QuarantinedQuestion, CriticReview

bedrock_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
AGENT_MODEL_ID = "zai.glm-5" # Or zai.glm-5

# ==========================================
# UTILS: The Custom JSON Parser
# ==========================================
def clean_json_response(content: str) -> dict:
    """Extracts JSON from an LLM response, ignoring conversational padding and fixing trailing commas."""
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
            # --- FIX: Remove trailing commas before parsing ---
            # This regex looks for a comma, optional whitespace, and then a closing bracket/brace
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return {}

def call_bedrock_json(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> dict:
    """Raw Bedrock Converse API call with regex JSON extraction."""
    response = bedrock_client.converse(
        modelId=AGENT_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": system_prompt}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
    )
    content = response['output']['message']['content'][0]['text']
    return clean_json_response(content)

# ==========================================
# AGENT 1: THE PLANNER
# ==========================================
def run_planner_agent(user_prompt: str, keywords: List[str] = None):
    yield _stream_event("log", {"message": "[Planner Agent] Analyzing intent and formulating search plan..."})

    keyword_instruction = ""
    if keywords and len(keywords) > 0:
        keyword_str = ", ".join(keywords)
        keyword_instruction = f"\n3. ANCHOR KEYWORDS: The user has explicitly requested to focus on these keywords: {keyword_str}. You MUST incorporate these deeply into your vector queries."
    
    system_prompt = f"""You are the Orchestration Agent for an academic RAG system.
Your job is to read the professor's request and formulate a precise search plan.
1. Extract 2-3 specific academic concepts to query the vector database.
2. Write strict instructions for the downstream Question Generator, explicitly noting what to avoid.{keyword_instruction}

CRITICAL: Return ONLY valid JSON matching this exact schema:
{{
    "vector_queries": ["query 1", "query 2"],
    "generator_instructions": "Focus on X, ignore Y."
}}
"""
    raw_json = call_bedrock_json(system_prompt, f"Professor's Request: {user_prompt}", temperature=0.1, max_tokens=1024)
    
    # Fallback in case of absolute failure
    if not raw_json:
        raw_json = {
            "vector_queries": keywords if keywords else ["core concepts"],
            "generator_instructions": "Generate a highly difficult academic quiz based strictly on the text."
        }
        
    yield SearchPlan(**raw_json)

# ==========================================
# AGENT 2: THE GENERATOR
# ==========================================
def run_generator_agent(
    instructions: str, 
    context_chunks: List[Dict[str, Any]], 
    previous_feedback: str = "",
    question_style: str = "Direct conceptual question."
) -> GeneratedQuestion:
    
    context_string = ""
    for chunk in context_chunks:
        context_string += f"<chunk id='{chunk.get('chunk_id')}'>\n{chunk.get('text', '')}\n</chunk>\n"

    system_prompt = f"""You are an expert academic assessment writer.

PLANNER INSTRUCTIONS: {instructions}

QUESTION STYLE REQUIREMENT: 
You MUST write the question using this specific format: {question_style}

RULES:
1. Generate exactly 1 highly difficult question based ONLY on the provided chunks.
2. Provide the exact source_chunk_ids.
3. Use the 'reasoning_scratchpad' to think step-by-step.
4. CRITICAL: DO NOT use "Roman Numeral" (I, II, III) or "Multiple True/False" statement formats.

CRITICAL: Return ONLY valid JSON matching this exact schema:
{{
    "reasoning_scratchpad": "Your step-by-step logic...",
    "question_text": "The actual question?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "The exact string of the correct option",
    "explanation": "Why this is correct...",
    "source_chunk_ids": ["ID of chunk 1", "ID of chunk 2"]
}}
"""

    user_prompt = f"<context>\n{context_string}</context>\n"
    if previous_feedback:
        user_prompt += f"\n<critic_feedback>\nYOUR PREVIOUS ATTEMPT FAILED. FIX THIS:\n{previous_feedback}\n</critic_feedback>"

    raw_json = call_bedrock_json(system_prompt, user_prompt, temperature=0.3, max_tokens=2048)
    
    # Ensure options array exists even if LLM slightly hallucinates schema
    if "options" not in raw_json:
        raw_json["options"] = ["A", "B", "C", "D"]
        
    return GeneratedQuestion(**raw_json)

# ==========================================
# AGENT 3: THE CRITIC
# ==========================================
def run_critic_agent(question: GeneratedQuestion, chunk_text: str) -> CriticReview:
    system_prompt = """You are a ruthless Academic Reviewer Agent. Your job is to catch bad LLM-generated questions.

FAIL THE QUESTION IF:
1. It asks about course administration, syllabus details, exam dates, or textbook names.
2. The correct answer cannot be 100% proven by reading the provided chunk.
3. It is trivially easy or poorly phrased.

CRITICAL: Return ONLY valid JSON matching this exact schema:
{
    "reasoning_scratchpad": "Evaluate the drafted question against the source chunk.",
    "is_approved": true,
    "feedback": "If rejected, explain why in one sentence."
}
"""
    
    prompt = f"""<source_chunk>
{chunk_text}
</source_chunk>

<drafted_question>
Question: {question.question_text}
Options: {question.options}
Answer: {question.correct_answer}
</drafted_question>
"""

    raw_json = call_bedrock_json(system_prompt, prompt, temperature=0.1, max_tokens=1024)
    
    # Fallback
    if "is_approved" not in raw_json:
        raw_json = {"reasoning_scratchpad": "Parse failed", "is_approved": False, "feedback": "JSON parse error."}
        
    return CriticReview(**raw_json)

# ==========================================
# ORCHESTRATION: The 3-Strike Loop
# ==========================================
def _stream_event(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def generate_validated_quiz(
    plan: SearchPlan,
    chunks: List[Dict[str, Any]],
    num_questions: int,
) -> Generator[str, None, None]:
    """Streams generation progress and final results as JSON chunks."""
    approved_questions = []
    quarantined_questions = []

    yield _stream_event("generation_started", {"num_questions": num_questions})
    
    yield _stream_event("log", {"message": f"[Orchestrator] Beginning generation of {num_questions} questions..."})

    question_styles = [
        "Scenario-based application: Present a hypothetical engineering or academic scenario where the student must apply the concepts to solve a problem.",
        "Compare and contrast: Ask the student to identify the core differences, trade-offs, or scaling laws between two concepts.",
        "Troubleshooting/Diagnostic: Present a system, equation, or architecture that is failing or acting unexpectedly, and ask the student to identify the root cause based on the text.",
        "Direct conceptual: Ask a straightforward but highly difficult question about a specific mechanism, definition, or architectural design."
    ]

    for i in range(num_questions):
        question_number = i + 1
        yield _stream_event("log", {"message": f"  -> Drafting Question {question_number}..."})
        attempts = 0
        max_attempts = 3
        feedback = ""
        question_draft = None
        
        current_style = random.choice(question_styles)
        
        # --- ADD THESE TWO LINES ---
        # Grab a random subset of up to 4 chunks from the total retrieved pool
        sample_size = min(4, len(chunks))
        chunk_subset = random.sample(chunks, sample_size)

        yield _stream_event(
            "question_started",
            {"question_number": question_number, "max_attempts": max_attempts},
        )

        while attempts < max_attempts:
            # 1. Generate
            try:
                # --- UPDATE THIS LINE to pass chunk_subset instead of chunks ---
                question_draft = run_generator_agent(plan.generator_instructions, chunk_subset, feedback, current_style)
                
                # Find the matching text for the critic (also using chunk_subset)
                cited_chunks = [
                    c.get("text") for c in chunk_subset 
                    if c.get("chunk_id") in question_draft.source_chunk_ids
                ]
                source_text = "\n\n--- NEXT SOURCE CHUNK ---\n\n".join(cited_chunks) if cited_chunks else "Source chunks not found."
                
                # 2. Critique
                review = run_critic_agent(question_draft, source_text)
                
                if review.is_approved:
                    yield _stream_event("log", {"message": f"     Approved on attempt {attempts + 1}"})
                    approved_questions.append(question_draft)
                    yield _stream_event("approved", question_draft.model_dump())
                    break
                else:
                    attempts += 1
                    feedback = review.feedback
                    yield _stream_event("log", {"message": f"     Rejected (Attempt {attempts}/{max_attempts}): {feedback}"})
                    yield _stream_event(
                        "rejected",
                        {
                            "question_number": question_number,
                            "attempt": attempts,
                            "feedback": feedback,
                        },
                    )
            except Exception as e:
                # Catch Pydantic mapping errors if the LLM completely botches the JSON
                attempts += 1
                feedback = f"System Error: {str(e)}"
                yield _stream_event("log", {"message": f"     Generation Error (Attempt {attempts}/{max_attempts}): {str(e)}"})
                yield _stream_event(
                    "rejected",
                    {
                        "question_number": question_number,
                        "attempt": attempts,
                        "feedback": feedback,
                    },
                )

        # 3. Quarantine if it failed 3 times
        if attempts == max_attempts and question_draft:
            yield _stream_event("log", {"message": "     Quarantined: Maximum retries reached."})
            # Safety check in case question_draft doesn't have the attribute
            q_text = getattr(question_draft, "question_text", "Failed to generate valid question text.")
            quarantined_question = QuarantinedQuestion(
                drafted_question=q_text,
                rejection_reason=feedback,
            )
            quarantined_questions.append(quarantined_question)
            yield _stream_event("quarantined", quarantined_question.model_dump())

    yield _stream_event(
        "complete",
        {
            "questions": [q.model_dump() for q in approved_questions],
            "quarantined_questions": [q.model_dump() for q in quarantined_questions],
        },
    )