"""Course outline from the documents collection (one entry per ingested lecture file)."""
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import DOCUMENTS_COLLECTION


def get_course_outline(client: QdrantClient, course_id: str, file_filters: list[str] | None) -> list[dict]:
    must = [models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id))]
    if file_filters:
        must.append(models.FieldCondition(key="file_name", match=models.MatchAny(any=file_filters)))

    outline: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=DOCUMENTS_COLLECTION,
            scroll_filter=models.Filter(must=must),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload
            outline.append(
                {
                    "file_name": payload["file_name"],
                    "lecture_number": payload.get("lecture_number"),
                    "title": payload.get("title", ""),
                    "summary": payload.get("summary", ""),
                    "topics": payload.get("topics", []),
                }
            )
        if offset is None:
            break

    outline.sort(key=lambda d: (d["lecture_number"] is None, d["lecture_number"] or 0, d["file_name"]))
    return outline


def list_course_files(client: QdrantClient, course_id: str) -> list[str]:
    return [d["file_name"] for d in get_course_outline(client, course_id, None)]
