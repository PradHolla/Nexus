"""Hybrid chunk retrieval for quiz slots. Sync; graph nodes call these through asyncio.to_thread."""
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import CHUNKS_COLLECTION
from src.embeddings import embed_query
from src.quiz.schemas import Slot
from src.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, build_sparse_document

PREFETCH_LIMIT = 30


def search_chunks(
    client: QdrantClient, course_id: str, query: str, files: list[str] | None, limit: int = 12
) -> list[dict]:
    """Dense + sparse prefetch fused with RRF over quiz-eligible chunks of the course; payloads in rank order."""
    must = [
        models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)),
        models.FieldCondition(key="quiz_eligible", match=models.MatchValue(value=True)),
    ]
    if files:
        must.append(models.FieldCondition(key="file_name", match=models.MatchAny(any=files)))
    query_filter = models.Filter(must=must)

    response = client.query_points(
        collection_name=CHUNKS_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=embed_query(query), using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT, filter=query_filter
            ),
            models.Prefetch(
                query=build_sparse_document(query), using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT, filter=query_filter
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [point.payload for point in response.points]


def assign_chunks(
    client: QdrantClient,
    course_id: str,
    slots: list[Slot],
    file_filters: list[str] | None,
    per_slot: int = 3,
) -> list[list[dict]]:
    """Top chunks per slot, in slot order. Each slot's first chunk (its anchor) is unique across slots so questions
    rest on distinct material; the supporting chunks may overlap."""
    allowed = set(file_filters or [])
    used_anchors: set[str] = set()
    assigned: list[list[dict]] = []
    for slot in slots:
        if not allowed:
            files = slot.target_files or None
        else:
            files = [f for f in slot.target_files if f in allowed] or list(file_filters)

        def pick(candidates: list[dict]) -> list[dict]:
            anchor = next((c for c in candidates if c["chunk_id"] not in used_anchors), None)
            if anchor is None:
                return []
            rest = [c for c in candidates if c["chunk_id"] != anchor["chunk_id"]]
            return [anchor, *rest[: per_slot - 1]]

        chunks = pick(search_chunks(client, course_id, slot.search_query, files))
        if not chunks and slot.target_files:
            fallback_files = list(file_filters) if allowed else None
            chunks = pick(search_chunks(client, course_id, slot.search_query, fallback_files))
        if chunks:
            used_anchors.add(chunks[0]["chunk_id"])
        assigned.append(chunks)
    return assigned
