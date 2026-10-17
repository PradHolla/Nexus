import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional, Tuple

from src.config import JOBS_DB_PATH

DB_PATH = JOBS_DB_PATH


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
                status TEXT NOT NULL,
                logs TEXT DEFAULT ''
            )
            """
        )
        # Ensure 'logs' column exists for users with older databases
        try:
            conn.execute("ALTER TABLE JobStatus ADD COLUMN logs TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        for column in ("course_id", "file_name", "file_path", "created_at"):
            try:
                conn.execute(f"ALTER TABLE JobStatus ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass


def create_job(job_id: str, course_id: str, file_name: str, file_path: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO JobStatus (job_id, status, logs, course_id, file_name, file_path, created_at)
            VALUES (?, 'queued', '', ?, ?, ?, ?)
            """,
            (job_id, course_id, file_name, file_path, datetime.now(timezone.utc).isoformat()),
        )


def list_unfinished_jobs() -> list[dict]:
    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT job_id, status, course_id, file_name, file_path, created_at
            FROM JobStatus WHERE status IN ('queued', 'processing')
            ORDER BY created_at
            """
        ).fetchall()
        return [dict(row) for row in rows]


def upsert_job_status(job_id: str, status: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO JobStatus (job_id, status, logs)
            VALUES (?, ?, '')
            ON CONFLICT(job_id) DO UPDATE SET status = excluded.status
            """,
            (job_id, status),
        )

def append_job_log(job_id: str, message: str) -> None:
    """Appends a log message to the job's log history."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE JobStatus SET logs = logs || ? || CHAR(10) WHERE job_id = ?",
            (message, job_id),
        )

def get_job_status_and_logs(job_id: str) -> Optional[Tuple[str, str]]:
    """Returns both the status and the full log history for a job."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT status, logs FROM JobStatus WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None


def get_job_status(job_id: str) -> Optional[str]:
    res = get_job_status_and_logs(job_id)
    return res[0] if res else None