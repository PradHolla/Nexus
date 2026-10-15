import concurrent.futures
import random
import uuid
import json
import time
from typing import List, Dict, Any, Optional, Union, Generator
from qdrant_client import QdrantClient
from qdrant_client.http import models
from flashrank import Ranker, RerankRequest

from src.ingestion.embedder import get_titan_embedding
from src.vector_store import (
    DEFAULT_COLLECTION_NAME,
    HYBRID_PREFETCH_LIMIT,
    HYBRID_RETRIEVAL_ENABLED,
    RERANK_TIMEOUT_SECONDS,
    SPARSE_MODEL_NAME,
    SPARSE_VECTOR_NAME,
    build_sparse_document,
    inspect_collection_schema,
)

def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

class QuizSampler:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str = DEFAULT_COLLECTION_NAME):
        self.client = qdrant_client
        self.collection_name = collection_name
        # Initialize the lightweight cross-encoder model
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        self.rerank_timeout_seconds = RERANK_TIMEOUT_SECONDS
        self._sparse_model_ready: Optional[bool] = None

    def _ensure_sparse_model(self) -> bool:
        if not HYBRID_RETRIEVAL_ENABLED:
            return False

        if self._sparse_model_ready is not None:
            return self._sparse_model_ready

        try:
            self.client.set_sparse_model(SPARSE_MODEL_NAME)
            self._sparse_model_ready = True
        except Exception:
            self._sparse_model_ready = False

        return self._sparse_model_ready

    def _get_collection_schema(self):
        return inspect_collection_schema(self.client.get_collection(self.collection_name))

    def _rerank_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        limit: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rerank_request = RerankRequest(query=query, passages=passages)
        started_at = time.perf_counter()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.ranker.rerank, rerank_request)
                reranked_results = future.result(timeout=self.rerank_timeout_seconds)
            rerank_mode = "reranked"
        except concurrent.futures.TimeoutError:
            reranked_results = passages
            rerank_mode = "timeout_fallback"
        except Exception as exc:
            reranked_results = passages
            rerank_mode = f"error_fallback:{exc}"

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return reranked_results[:limit], {"mode": rerank_mode, "latency_ms": elapsed_ms}

    def _points_to_passages(self, points: List[Any]) -> List[Dict[str, Any]]:
        passages = []
        for hit in points:
            payload = hit.payload or {}
            passages.append(
                {
                    "id": payload.get("chunk_id", str(getattr(hit, "id", uuid.uuid4()))),
                    "text": payload.get("text", ""),
                    "payload": payload,
                }
            )
        return passages

    def _dense_search(
        self,
        query_vector: List[float],
        query_filter: models.Filter,
        limit: int,
        dense_vector_name: str | None,
    ):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=dense_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

    def _hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        query_filter: models.Filter,
        limit: int,
        dense_vector_name: str,
    ):
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=dense_vector_name,
                    filter=query_filter,
                    limit=limit,
                ),
                models.Prefetch(
                    query=build_sparse_document(query_text),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

    def get_course_files(self, course_id: str) -> List[str]:
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id))]
                ),
                limit=10000,
                with_payload=["file_name"],
                with_vectors=False
            )
        except Exception:
            return []
        unique_files = {r.payload.get("file_name") for r in records if r.payload and "file_name" in r.payload}
        return sorted(list(unique_files))

    def get_quiz_chunks(
        self, 
        course_id: str, 
        num_questions: int, 
        file_filters: Optional[List[str]] = None,
        vector_queries: Optional[List[str]] = None,
        retrieval_mode: str = "hybrid",
        max_context_chunks: Optional[int] = None,
        include_retrieval_diagnostics: bool = False,
    ) -> Generator[Union[str, List[Dict[str, Any]]], None, None]:
        retrieval_mode = retrieval_mode.lower()
        
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
            yield _sse_event("log", {"message": f"Executing retrieval for agent queries: {vector_queries}"})
            all_results = []
            seen_ids = set()

            try:
                schema = self._get_collection_schema()
            except Exception as exc:
                yield _sse_event("log", {"message": f"Retrieval skipped because the Qdrant collection is unavailable: {exc}"})
                yield []
                return
            context_limit = max_context_chunks or (num_questions * 2)
            limit_per_query = max(2, context_limit // len(vector_queries))
            candidate_limit = max(HYBRID_PREFETCH_LIMIT, limit_per_query * 4)

            hybrid_requested = retrieval_mode != "dense"
            hybrid_available = (
                hybrid_requested
                and schema.supports_hybrid
                and schema.dense_vector_name is not None
                and self._ensure_sparse_model()
            )

            if hybrid_available:
                yield _sse_event(
                    "log",
                    {"message": f"Hybrid retrieval enabled with sparse model {SPARSE_MODEL_NAME}."},
                )
            else:
                fallback_reason = "dense retrieval forced by request" if retrieval_mode == "dense" else "hybrid unavailable"
                yield _sse_event(
                    "log",
                    {"message": f"Falling back to dense-only retrieval ({fallback_reason})."},
                )

            for query in vector_queries:
                query_vector = get_titan_embedding(query)
                if not query_vector:
                    continue

                retrieval_variant = "dense"
                try:
                    if hybrid_available and schema.dense_vector_name is not None:
                        search_response = self._hybrid_search(
                            query,
                            query_vector,
                            q_filter,
                            candidate_limit,
                            schema.dense_vector_name,
                        )
                        retrieval_variant = "hybrid_rrf"
                    else:
                        search_response = self._dense_search(
                            query_vector,
                            q_filter,
                            candidate_limit,
                            schema.dense_vector_name,
                        )
                except Exception as exc:
                    yield _sse_event(
                        "log",
                        {"message": f"Hybrid retrieval failed for '{query}'. Falling back to dense-only search. Error: {exc}"},
                    )
                    retrieval_variant = "dense_fallback"
                    search_response = self._dense_search(
                        query_vector,
                        q_filter,
                        candidate_limit,
                        schema.dense_vector_name,
                    )

                if not search_response.points:
                    continue

                passages = self._points_to_passages(search_response.points)
                top_reranked, rerank_meta = self._rerank_passages(query, passages, limit_per_query)

                if include_retrieval_diagnostics:
                    yield _sse_event(
                        "retrieval_diagnostics",
                        {
                            "query": query,
                            "mode": retrieval_variant,
                            "candidate_count": len(passages),
                            "selected_count": len(top_reranked),
                            "rerank_mode": rerank_meta["mode"],
                            "rerank_latency_ms": rerank_meta["latency_ms"],
                        },
                    )

                # Deduplicate chunks
                for hit in top_reranked:
                    chunk_id = hit["id"]
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_results.append(hit["payload"])
                        
            yield all_results
            return

        # Fallback: Stratified Random Sampling
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=q_filter,
                limit=10000, 
                with_payload=True,
                with_vectors=False 
            )
        except Exception:
            yield []
            return
        yield [r.payload for r in records][:num_questions * 2]
