import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    Faithfulness,
    AnswerRelevancy,
)
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_core.outputs import ChatResult

# Import your actual pipeline components
from src.retrieval.sampler import QuizSampler
from src.quiz.generator import run_planner_agent, run_generator_agent
from qdrant_client import QdrantClient

# 1. Initialize Judge models (LLM + embeddings)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_JUDGE_MODEL_ID = "zai.glm-5"
JUDGE_MODEL_ID = os.getenv("RAGAS_JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL_ID)

class JSONCleanChatBedrock(ChatBedrockConverse):
    """Wraps ChatBedrockConverse to strip markdown backticks from JSON outputs."""
    
    def _clean_json_string(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            if isinstance(gen.text, str):
                gen.text = self._clean_json_string(gen.text)
            if hasattr(gen, "message") and hasattr(gen.message, "content") and isinstance(gen.message.content, str):
                gen.message.content = self._clean_json_string(gen.message.content)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            if isinstance(gen.text, str):
                gen.text = self._clean_json_string(gen.text)
            if hasattr(gen, "message") and hasattr(gen.message, "content") and isinstance(gen.message.content, str):
                gen.message.content = self._clean_json_string(gen.message.content)
        return result

judge_llm = JSONCleanChatBedrock(
    model=JUDGE_MODEL_ID,
    region_name=AWS_REGION,
    temperature=0.0 # Force deterministic output
)

judge_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=AWS_REGION,
)

# Initialize your database connection
qdrant_client = QdrantClient("http://localhost:6333")
sampler = QuizSampler(qdrant_client)

def generate_test_data():
    """Runs a few test prompts through our pipeline to gather data for the Judge."""
    print("Gathering trace data from the Nexus pipeline...")
    
    # We define a few test prompts that a professor might type
    test_prompts = [
        "Create a question about the mathematical mechanism of LSTM forget gates.",
        "Generate a question comparing additive and multiplicative attention.",
        "Ask about the vanishing gradient problem in standard RNNs."
    ]
    
    data_samples = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "ground_truth": [] # Optional for these specific metrics, but good practice
    }

    course_id = "CS224"

    for prompt in test_prompts:
        # A. Run the Planner
        plan = run_planner_agent(prompt)
        
        # B. Run the Retriever
        chunks = sampler.get_quiz_chunks(
            course_id=course_id,
            num_questions=1,
            vector_queries=plan.vector_queries
        )
        context_texts = [c.get("text", "") for c in chunks]
        
        # C. Run the Generator
        draft = run_generator_agent(
            instructions=plan.generator_instructions,
            num_questions=1,
            context_chunks=chunks
        )
        
        # Format the output so Ragas understands it
        generated_q = draft.questions[0]
        full_answer = f"Question: {generated_q.question_text}\nCorrect Answer: {generated_q.correct_answer}\nExplanation: {generated_q.explanation}"
        
        data_samples["user_input"].append(prompt)
        data_samples["response"].append(full_answer)
        data_samples["retrieved_contexts"].append(context_texts)
        data_samples["ground_truth"].append("Not required for Faithfulness/Relevance")

    return Dataset.from_dict(data_samples)

def run_evaluation():
    # 1. Generate the raw data
    dataset = generate_test_data()
    
    print("\nStarting Ragas Evaluation (This may take a minute)...")
    
    # 2. Run the LLM-as-a-Judge evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(llm=judge_llm),                           # Did Qdrant retrieve highly relevant chunks?
            Faithfulness(llm=judge_llm),                              # Did the Generator hallucinate?
            AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),  # Did the output match the user's prompt?
        ],
    )

    score_df = result.to_pandas()

    def mean_or_none(column_name: str):
        if column_name not in score_df.columns:
            return None
        non_null = score_df[column_name].dropna()
        if non_null.empty:
            return None
        return float(non_null.mean())

    context_score = mean_or_none("context_precision")
    faithfulness_score = mean_or_none("faithfulness")
    answer_relevancy_score = mean_or_none("answer_relevancy")

    def format_score(score):
        return f"{score:.2f} / 1.0" if score is not None else "N/A"
    
    # 3. Print the scorecard
    print("\n" + "="*40)
    print("NEXUS - RAG TRIAD SCORECARD")
    print("="*40)
    print(f"Context Precision: {format_score(context_score)}")
    print(f"Faithfulness:      {format_score(faithfulness_score)}")
    print(f"Answer Relevance:  {format_score(answer_relevancy_score)}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()