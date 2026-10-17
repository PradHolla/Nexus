"""Local dense embeddings (fastembed, ONNX). Sync API; async callers must use asyncio.to_thread."""
import os
import threading

from fastembed import TextEmbedding

from src.config import CACHE_DIR

EMBED_MODEL = os.getenv("EMBED_MODEL", "ibm-granite/granite-embedding-small-english-r2")
EMBED_BATCH_SIZE = 8  # small batches pad less, ~7x faster than the default 256 here

EMBED_DIM: int = next(
    m["dim"] for m in TextEmbedding.list_supported_models() if m["model"].lower() == EMBED_MODEL.lower()
)

_lock = threading.Lock()
_model: TextEmbedding | None = None


def _embed(texts: list[str]) -> list[list[float]]:
    global _model
    with _lock:
        if _model is None:
            _model = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE_DIR / "fastembed"))
        return [vec.tolist() for vec in _model.embed(texts, batch_size=EMBED_BATCH_SIZE)]


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _embed(texts)


def embed_query(text: str) -> list[float]:
    return _embed([text])[0]
