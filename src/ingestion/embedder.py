import os
import json
import uuid
import boto3
from botocore.config import Config
import concurrent.futures
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.job_status_store import append_job_log
from src.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    HYBRID_RETRIEVAL_ENABLED,
    SPARSE_VECTOR_NAME,
    SPARSE_MODEL_NAME,
    build_sparse_document,
    inspect_collection_schema,
)

# Configure the client to handle high concurrency
custom_config = Config(
    max_pool_connections=25, # Give it headroom above our 20 workers
    retries={'max_attempts': 3} # Pro-tip: Add automatic retries for API hiccups
)

bedrock_client = boto3.client(
    "bedrock-runtime", 
    config=custom_config
)

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


def _log_to_jobs(message: str, jobs: Optional[List[str]] = None) -> None:
    if jobs:
        for job_id in jobs:
            append_job_log(job_id, message)
    else:
        print(message)


def _enable_sparse_model(qdrant_client: QdrantClient, jobs: Optional[List[str]] = None) -> bool:
    if not HYBRID_RETRIEVAL_ENABLED:
        return False

    try:
        qdrant_client.set_sparse_model(SPARSE_MODEL_NAME)
        return True
    except Exception as exc:
        _log_to_jobs(
            f"Hybrid retrieval disabled for this ingestion run: sparse model '{SPARSE_MODEL_NAME}' is unavailable ({exc}).",
            jobs,
        )
        return False

def get_titan_embedding(text: str) -> List[float]:
    try:
        response = bedrock_client.invoke_model(
            body=json.dumps({"inputText": text}),
            modelId=EMBEDDING_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")
    except Exception as e:
        print(f"Embedding failed: {e}")
        return []


def _build_point_vector(
    chunk: Dict[str, Any],
    dense_vector: List[float],
    dense_vector_name: str | None,
    use_sparse_vector: bool,
) -> List[float] | Dict[str, Any]:
    if dense_vector_name is None:
        return dense_vector

    vector: Dict[str, Any] = {dense_vector_name: dense_vector}
    if use_sparse_vector:
        vector[SPARSE_VECTOR_NAME] = build_sparse_document(chunk["text"])
    return vector


def _embed_single_chunk(
    chunk: Dict[str, Any],
    jobs: Optional[List[str]] = None,
    dense_vector_name: str | None = DENSE_VECTOR_NAME,
    use_sparse_vector: bool = False,
) -> models.PointStruct | None:
    """Helper function for the thread pool to process a single chunk."""
    text = chunk.get("text", "")
    if not text:
        return None
        
    # --- BULLETPROOF SAFETY NET ---
    # We lowered this to 8,000 chars. This mathematically guarantees 
    # we will never exceed the 8,192 token limit of AWS Titan.
    if len(text) > 8000:
        msg = f"Warning: Chunk {chunk['chunk_id']} is massive ({len(text)} chars). Truncating to 8000 to fit Titan limits."
        _log_to_jobs(msg, jobs)
            
        text = text[:8000]
        # We must update the payload text so the LLM reads the exact same text that was embedded
        chunk["text"] = text 
    # -------------------------------------
        
    vector = get_titan_embedding(text)
    
    if vector:
        return models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])), 
            vector=_build_point_vector(chunk, vector, dense_vector_name, use_sparse_vector),
            payload=chunk
        )
    return None

def process_and_ingest_document(
    parsed_chunks: List[Dict[str, Any]], 
    qdrant_client: QdrantClient, 
    collection_name: str = DEFAULT_COLLECTION_NAME,
    jobs: Optional[List[str]] = None
):
    sparse_model_enabled = _enable_sparse_model(qdrant_client, jobs)

    try:
        collection_info = qdrant_client.get_collection(collection_name)
        schema = inspect_collection_schema(collection_info)
    except Exception:
        msg = f"Creating new Qdrant collection: {collection_name} with named dense vectors"
        if sparse_model_enabled:
            msg += f" and sparse vectors via {SPARSE_MODEL_NAME}"
        _log_to_jobs(msg, jobs)
            
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config=(
                {
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
                if sparse_model_enabled
                else None
            ),
            # --- OPTIMIZATION: INT8 Scalar Quantization ---
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
            # ----------------------------------------------
        )
        schema = inspect_collection_schema(qdrant_client.get_collection(collection_name))

    use_sparse_vector = sparse_model_enabled and schema.supports_hybrid
    if sparse_model_enabled and not schema.supports_hybrid:
        _log_to_jobs(
            "Collection schema is dense-only, so new chunks will be indexed without sparse vectors. Recreate the collection and re-ingest to enable hybrid retrieval.",
            jobs,
        )

    msg = f"Concurrently embedding {len(parsed_chunks)} chunks..."
    if use_sparse_vector:
        msg += " Hybrid sparse indexing is enabled."
    _log_to_jobs(msg, jobs)
    
    points = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                _embed_single_chunk,
                chunk,
                jobs,
                schema.dense_vector_name,
                use_sparse_vector,
            )
            for chunk in parsed_chunks
        ]
        for future in concurrent.futures.as_completed(futures):
            p = future.result()
            if p:
                points.append(p)

    if points:
        batch_size = 500 
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
        msg = f"Successfully bulk-upserted {len(points)} vectors to Qdrant."
        _log_to_jobs(msg, jobs)
    else:
        msg = "No valid vectors generated."
        _log_to_jobs(msg, jobs)
