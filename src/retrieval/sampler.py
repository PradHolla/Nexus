import random
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.embedder import get_titan_embedding

class QuizSampler:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str = "scholera_course_materials"):
        self.client = qdrant_client
        self.collection_name = collection_name

    def get_course_files(self, course_id: str) -> List[str]:
        """Fetches a list of unique filenames currently stored in Qdrant for a course."""
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
    ) -> List[Dict[str, Any]]:
        """
        Retrieves context chunks based on Planner Agent queries and strict file filters.
        """
        
        # 1. Build the base filter
        must_conditions = [
            models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)),
            models.FieldCondition(key="is_administrative", match=models.MatchValue(value=False))
        ]

        # 2. Add explicit file filters if the professor selected them
        if file_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchAny(any=file_filters)
                )
            )

        q_filter = models.Filter(must=must_conditions)

        # 3. Agentic Semantic Search (Using queries from the Planner)
        if vector_queries:
            print(f"Executing semantic search for agent queries: {vector_queries}")
            all_results = []
            seen_ids = set()
            
            # Divide the chunk allowance evenly among the queries
            limit_per_query = max(2, (num_questions * 2) // len(vector_queries))

            for query in vector_queries:
                query_vector = get_titan_embedding(query)
                if not query_vector:
                    continue
                    
                search_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=limit_per_query, 
                    with_payload=True
                )
                
                # Deduplicate chunks
                for hit in search_response.points:
                    chunk_id = hit.payload.get("chunk_id")
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_results.append(hit.payload)
                        
            return all_results

        # 4. Fallback: Stratified Random Sampling
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=q_filter,
            limit=10000, 
            with_payload=True,
            with_vectors=False 
        )
        return [r.payload for r in records][:num_questions * 2]