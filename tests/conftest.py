import os
import tempfile
from pathlib import Path

# Must run before anything imports `src`: src.config reads these at import time. Tests get their own Qdrant
# collections and a fresh var dir (jobs/checkpoints dbs, uploads, vision cache), never the real ones.
os.environ["QDRANT_CHUNKS_COLLECTION"] = "test_nexus_chunks"
os.environ["QDRANT_DOCUMENTS_COLLECTION"] = "test_nexus_documents"
os.environ["NEXUS_VAR_DIR"] = tempfile.mkdtemp(prefix="nexus_test_var_")

# The embedding model is a ~180MB download; share its cache with the real var dir instead of re-downloading
# on every test run (it is not test data, unlike the vision cache).
_fastembed_cache = Path(__file__).resolve().parents[1] / "var" / "cache" / "fastembed"
_fastembed_cache.mkdir(parents=True, exist_ok=True)
(Path(os.environ["NEXUS_VAR_DIR"]) / "cache").mkdir()
(Path(os.environ["NEXUS_VAR_DIR"]) / "cache" / "fastembed").symlink_to(_fastembed_cache, target_is_directory=True)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
