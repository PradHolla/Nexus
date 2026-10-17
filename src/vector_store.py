import os

from qdrant_client.http import models


DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25")
SPARSE_BM25_AVG_LEN = float(os.getenv("QDRANT_SPARSE_BM25_AVG_LEN", "256"))


def sparse_model_uses_bm25_options(model_name: str | None = None) -> bool:
    normalized = (model_name or SPARSE_MODEL_NAME).strip().lower()
    return normalized in {"qdrant/bm25", "qdrant/minicoil-v1"}


def build_sparse_document(text: str) -> models.Document:
    options = None
    if sparse_model_uses_bm25_options():
        options = {"avg_len": SPARSE_BM25_AVG_LEN}
    return models.Document(text=text, model=SPARSE_MODEL_NAME, options=options)
