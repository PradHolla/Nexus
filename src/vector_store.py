import os
from dataclasses import dataclass
from typing import Any

from qdrant_client.http import models

from src.config import CHUNKS_COLLECTION


DEFAULT_COLLECTION_NAME = CHUNKS_COLLECTION
DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25")
SPARSE_BM25_AVG_LEN = float(os.getenv("QDRANT_SPARSE_BM25_AVG_LEN", "256"))
HYBRID_RETRIEVAL_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HYBRID_PREFETCH_LIMIT = int(os.getenv("QDRANT_HYBRID_PREFETCH_LIMIT", "30"))
RERANK_TIMEOUT_SECONDS = float(os.getenv("QDRANT_RERANK_TIMEOUT_SECONDS", "10"))


@dataclass(frozen=True)
class CollectionSchema:
    dense_vector_name: str | None
    is_named_dense: bool
    has_sparse_vector: bool

    @property
    def supports_hybrid(self) -> bool:
        return self.is_named_dense and self.has_sparse_vector and self.dense_vector_name is not None


def sparse_model_uses_bm25_options(model_name: str | None = None) -> bool:
    normalized = (model_name or SPARSE_MODEL_NAME).strip().lower()
    return normalized in {"qdrant/bm25", "qdrant/minicoil-v1"}


def build_sparse_document(text: str) -> models.Document:
    options = None
    if sparse_model_uses_bm25_options():
        options = {"avg_len": SPARSE_BM25_AVG_LEN}
    return models.Document(text=text, model=SPARSE_MODEL_NAME, options=options)


def inspect_collection_schema(collection_info: Any) -> CollectionSchema:
    params = getattr(getattr(collection_info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)

    if isinstance(vectors, dict):
        dense_vector_name = DENSE_VECTOR_NAME if DENSE_VECTOR_NAME in vectors else next(iter(vectors), None)
        is_named_dense = True
    else:
        dense_vector_name = None
        is_named_dense = False

    has_sparse_vector = isinstance(sparse_vectors, dict) and SPARSE_VECTOR_NAME in sparse_vectors

    return CollectionSchema(
        dense_vector_name=dense_vector_name,
        is_named_dense=is_named_dense,
        has_sparse_vector=has_sparse_vector,
    )
