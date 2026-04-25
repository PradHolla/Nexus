import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Tuple

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