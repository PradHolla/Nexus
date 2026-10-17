"""Runs ingestion graphs as background asyncio tasks and recovers unfinished jobs after a restart."""
import asyncio
import logging
import os
from pathlib import Path

from src.job_status_store import append_job_log, list_unfinished_jobs, upsert_job_status

logger = logging.getLogger(__name__)


class IngestionRunner:
    def __init__(self, graph, qdrant_client):
        self.graph = graph
        self.qdrant_client = qdrant_client
        self._semaphore = asyncio.Semaphore(int(os.getenv("INGEST_MAX_FILES", "2")))
        self._tasks: set[asyncio.Task] = set()

    def _config(self, job_id: str) -> dict:
        # The qdrant client is not serializable; LangGraph keeps non-primitive configurable values out of the
        # checkpoint metadata, so passing it through the run config is safe.
        return {"configurable": {"thread_id": job_id, "qdrant": self.qdrant_client}}

    def _start(self, job_id: str, state: dict | None) -> None:
        task = asyncio.create_task(self._run(job_id, state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def submit(self, job_id: str, course_id: str, file_name: str, file_path: str) -> None:
        state = {"job_id": job_id, "course_id": course_id, "file_name": file_name, "file_path": file_path}
        self._start(job_id, state)

    def resume(self, job_id: str) -> None:
        self._start(job_id, None)

    async def _run(self, job_id: str, state: dict | None) -> None:
        config = self._config(job_id)
        try:
            async with self._semaphore:
                await asyncio.to_thread(upsert_job_status, job_id, "processing")
                result = await self.graph.ainvoke(state, config=config, durability="sync")

            await asyncio.to_thread(upsert_job_status, job_id, "completed")
            await asyncio.to_thread(append_job_log, job_id, f"Completed. Stats: {result.get('stats')}")
            file_path = result.get("file_path")
            if file_path:
                Path(file_path).unlink(missing_ok=True)
            await self.graph.checkpointer.adelete_thread(job_id)
        except asyncio.CancelledError:
            raise  # shutdown: leave the job as "processing" so recover() picks it up on the next start
        except Exception as e:
            logger.exception("Ingestion job %s failed", job_id)
            message = f"{type(e).__name__}: {str(e)[:200]}"
            await asyncio.to_thread(upsert_job_status, job_id, f"failed: {message}")
            await asyncio.to_thread(append_job_log, job_id, f"Job failed: {type(e).__name__}: {e}")

    async def recover(self) -> list[str]:
        """Resume or restart every queued/processing job. Call once at startup."""
        handled: list[str] = []
        for job in await asyncio.to_thread(list_unfinished_jobs):
            job_id = job["job_id"]
            snapshot = await self.graph.aget_state(self._config(job_id))
            if snapshot.next:
                await asyncio.to_thread(append_job_log, job_id, "Resuming from checkpoint after restart.")
                self.resume(job_id)
            elif job["file_path"] and Path(job["file_path"]).exists():
                await self.graph.checkpointer.adelete_thread(job_id)  # drop any stale finished checkpoint
                await asyncio.to_thread(append_job_log, job_id, "Restarting after restart (no checkpoint).")
                self._start(
                    job_id,
                    {
                        "job_id": job_id,
                        "course_id": job["course_id"],
                        "file_name": job["file_name"],
                        "file_path": job["file_path"],
                    },
                )
            else:
                await asyncio.to_thread(upsert_job_status, job_id, "failed: interrupted (upload missing)")
            handled.append(job_id)
        return handled

    async def shutdown(self) -> None:
        """Cancel running jobs (they stay "processing" and are recovered on the next start)."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
