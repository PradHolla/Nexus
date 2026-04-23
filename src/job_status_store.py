import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

DB_PATH = Path(__file__).resolve().parents[1] / "jobs.db"


@contextmanager
def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS JobStatus (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL
            )
            """
        )


def upsert_job_status(job_id: str, status: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO JobStatus (job_id, status)
            VALUES (?, ?)
            ON CONFLICT(job_id) DO UPDATE SET status = excluded.status
            """,
            (job_id, status),
        )


def get_job_status(job_id: str) -> Optional[str]:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT status FROM JobStatus WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return row[0] if row else None