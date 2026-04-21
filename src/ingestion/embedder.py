import os
import json
import uuid
import boto3
import concurrent.futures
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

bedrock_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
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
    collection_name: str = "scholera_course_materials"
):
    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        print(f"Creating new Qdrant collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )

    print(f"Concurrently embedding {len(parsed_chunks)} chunks...")
    
    points = []
    # OPTIMIZATION 1: Multithreading the network I/O
    # 20 workers is the sweet spot for AWS Bedrock rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(_embed_single_chunk, parsed_chunks)
        points = [p for p in results if p is not None]

    # OPTIMIZATION 4: Bulk Upserts
    if points:
        batch_size = 500 # Pushing 500 vectors per network call to Qdrant
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
        print(f"Successfully bulk-upserted {len(points)} vectors to Qdrant.")
    else:
        print("No valid vectors generated.")