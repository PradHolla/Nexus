"""Qdrant writes for the PDF ingestion pipeline: collection setup and idempotent per-file indexing."""
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION
from src.embeddings import EMBED_DIM, embed_documents
from src.ingestion.schemas import Chunk, DocumentAnalysis, ExtractedDocument
from src.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, build_sparse_document

UPSERT_BATCH_SIZE = 64

_CHUNK_INDEXES = {
    "course_id": models.PayloadSchemaType.KEYWORD,
    "file_name": models.PayloadSchemaType.KEYWORD,
    "page_type": models.PayloadSchemaType.KEYWORD,
    "quiz_eligible": models.PayloadSchemaType.BOOL,
    "lecture_number": models.PayloadSchemaType.INTEGER,
}
_DOCUMENT_INDEXES = {
    "course_id": models.PayloadSchemaType.KEYWORD,
    "file_name": models.PayloadSchemaType.KEYWORD,
}


def _dense_config() -> dict[str, models.VectorParams]:
    return {DENSE_VECTOR_NAME: models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)}


def _check_schema(client: QdrantClient, name: str, *, sparse: bool) -> None:
    params = client.get_collection(name).config.params
    vectors = params.vectors
    dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
    problem = None
    if dense is None or dense.size != EMBED_DIM:
        problem = f"its dense vector must be named {DENSE_VECTOR_NAME!r} with size {EMBED_DIM}"
    elif sparse and not (isinstance(params.sparse_vectors, dict) and SPARSE_VECTOR_NAME in params.sparse_vectors):
        problem = f"it must have a sparse vector named {SPARSE_VECTOR_NAME!r}"
    if problem:
        raise RuntimeError(
            f"Qdrant collection {name!r} has an incompatible schema: {problem}. "
            "Delete the collection and re-ingest."
        )


def _ensure_collection(
    client: QdrantClient, name: str, indexes: dict[str, models.PayloadSchemaType], *, sparse: bool
) -> None:
    if client.collection_exists(name):
        _check_schema(client, name, sparse=sparse)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=_dense_config(),
        sparse_vectors_config=(
            {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)} if sparse else None
        ),
    )
    for field, schema in indexes.items():
        client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)


def ensure_collections(client: QdrantClient) -> None:
    _ensure_collection(client, CHUNKS_COLLECTION, _CHUNK_INDEXES, sparse=True)
    _ensure_collection(client, DOCUMENTS_COLLECTION, _DOCUMENT_INDEXES, sparse=False)


def _file_filter(course_id: str, file_name: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)),
            models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name)),
        ]
    )


def index_document(
    client: QdrantClient,
    *,
    course_id: str,
    extracted: ExtractedDocument,
    analysis: DocumentAnalysis,
    chunks: list[Chunk],
) -> dict:
    """Replace everything stored for (course_id, file name) with these chunks. Same input gives the same point ids."""
    file_name = extracted.file_name

    # Embed first so a failure here leaves the previously indexed version untouched.
    dense_vectors = embed_documents([c.text for c in chunks])
    summary_text = f"{analysis.title}\n\n{analysis.summary}\n\nTopics: {', '.join(analysis.topics)}"
    document_vector = embed_documents([summary_text])[0]

    selector = models.FilterSelector(filter=_file_filter(course_id, file_name))
    client.delete(collection_name=CHUNKS_COLLECTION, points_selector=selector)
    client.delete(collection_name=DOCUMENTS_COLLECTION, points_selector=selector)

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: build_sparse_document(chunk.text),
                },
                payload={**chunk.model_dump(), "page_number": chunk.page_start},
            )
            for chunk, dense in zip(batch, dense_vectors[start : start + UPSERT_BATCH_SIZE])
        ]
        client.upsert(collection_name=CHUNKS_COLLECTION, points=points)

    quiz_eligible = sum(1 for c in chunks if c.quiz_eligible)
    client.upsert(
        collection_name=DOCUMENTS_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{course_id}:{file_name}")),
                vector={DENSE_VECTOR_NAME: document_vector},
                payload={
                    "course_id": course_id,
                    "file_name": file_name,
                    "lecture_number": extracted.lecture_number,
                    "title": analysis.title,
                    "summary": analysis.summary,
                    "topics": analysis.topics,
                    "page_count": extracted.page_count,
                    "chunk_count": len(chunks),
                    "quiz_eligible_chunk_count": quiz_eligible,
                    "file_hash": extracted.file_hash,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return {"chunks": len(chunks), "quiz_eligible": quiz_eligible}
