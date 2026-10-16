from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
VAR_DIR = Path(os.getenv("NEXUS_VAR_DIR", str(ROOT_DIR / "var"))).resolve()
JOBS_DB_PATH = VAR_DIR / "jobs.db"
CHECKPOINTS_DB_PATH = VAR_DIR / "checkpoints.db"
CACHE_DIR = VAR_DIR / "cache"
UPLOADS_DIR = VAR_DIR / "uploads"

CHUNKS_COLLECTION = os.getenv("QDRANT_CHUNKS_COLLECTION", "Nexus_course_materials")
DOCUMENTS_COLLECTION = os.getenv("QDRANT_DOCUMENTS_COLLECTION", "Nexus_course_documents")

LLM_MODEL = os.getenv("NEXUS_LLM_MODEL", "claude-haiku-5-5")


def ensure_var_dirs() -> None:
    """Create VAR_DIR, CACHE_DIR and UPLOADS_DIR (parents ok, exist ok)."""
    for directory in (VAR_DIR, CACHE_DIR, UPLOADS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
