import random
import uuid
import json
from typing import List, Dict, Any, Optional, Union, Generator
from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest

from src.ingestion.embedder import get_titan_embedding

def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

class QuizSampler:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str = "Nexus_course_materials"):
        self.client = qdrant_client
        self.collection_name = collection_name
        # Initialize the lightweight cross-encoder model
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")

    def get_course_files(self, course_id: str) -> List[str]:
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id))]
            ),
            limit=10000,
            with_payload=["file_name"],
            with_vectors=False
        )
        unique_files = {r.payload.get("file_name") for r in records if r.payload and "file_name" in r.payload}
        return sorted(list(unique_files))

    def get_quiz_chunks(
        self, 
        course_id: str, 
        num_questions: int, 
        file_filters: Optional[List[str]] = None,
        vector_queries: Optional[List[str]] = None
    ) -> Generator[Union[str, List[Dict[str, Any]]], None, None]:
        
        must_conditions = [
            models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)),
            models.FieldCondition(key="is_administrative", match=models.MatchValue(value=False))
        ]

        if file_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchAny(any=file_filters)
                )
            )

        q_filter = models.Filter(must=must_conditions)

        if vector_queries:
            yield _sse_event("log", {"message": f"Executing two-stage semantic search for agent queries: {vector_queries}"})
            all_results = []
            seen_ids = set()
            
            limit_per_query = max(2, (num_questions * 2) // len(vector_queries))

            for query in vector_queries:
                query_vector = get_titan_embedding(query)
                if not query_vector:
                    continue
                    
                # STAGE 1: Broad Dense Retrieval
                search_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=30, # Cast a wide net to catch potentially missed contexts
                    with_payload=True
                )
                
                if not search_response.points:
                    continue

                # STAGE 2: Cross-Encoder Reranking
                passages = []
                for hit in search_response.points:
                    passages.append({
                        "id": hit.payload.get("chunk_id", str(uuid.uuid4())),
                        "text": hit.payload.get("text", ""),
                        "payload": hit.payload # Keep reference for the final output
                    })

                rerank_request = RerankRequest(query=query, passages=passages)
                reranked_results = self.ranker.rerank(rerank_request)
                
                # Slice the top results post-reranking
                top_reranked = reranked_results[:limit_per_query]

                # Deduplicate chunks
                for hit in top_reranked:
                    chunk_id = hit["id"]
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_results.append(hit["payload"])
                        
            yield all_results
            return

        # Fallback: Stratified Random Sampling
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=q_filter,
            limit=10000, 
            with_payload=True,
            with_vectors=False 
        )
        yield [r.payload for r in records][:num_questions * 2]