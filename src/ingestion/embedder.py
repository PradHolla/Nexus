import os
import json
import uuid
import boto3
from botocore.config import Config
import concurrent.futures
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

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

def _embed_single_chunk(chunk: Dict[str, Any]) -> models.PointStruct | None:
    """Helper function for the thread pool to process a single chunk."""
    text = chunk.get("text", "")
    if not text:
        return None
        
    # --- BULLETPROOF SAFETY NET ---
    # We lowered this to 8,000 chars. This mathematically guarantees 
    # we will never exceed the 8,192 token limit of AWS Titan.
    if len(text) > 8000:
        print(f"Warning: Chunk {chunk['chunk_id']} is massive ({len(text)} chars). Truncating to 8000 to fit Titan limits.")
        text = text[:8000]
        # We must update the payload text so the LLM reads the exact same text that was embedded
        chunk["text"] = text 
    # -------------------------------------
        
    vector = get_titan_embedding(text)
    
    if vector:
        return models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])), 
            vector=vector,
            payload=chunk
        )
    return None

def process_and_ingest_document(
    parsed_chunks: List[Dict[str, Any]], 
    qdrant_client: QdrantClient, 
    collection_name: str = "Nexus_course_materials"
):
    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        print(f"Creating new Qdrant collection: {collection_name} with INT8 Quantization")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
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

    print(f"Concurrently embedding {len(parsed_chunks)} chunks...")
    
    points = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(_embed_single_chunk, parsed_chunks)
        points = [p for p in results if p is not None]

    if points:
        batch_size = 500 
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
        print(f"Successfully bulk-upserted {len(points)} vectors to Qdrant.")
    else:
        print("No valid vectors generated.")