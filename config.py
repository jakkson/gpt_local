import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Per-request timeout for chat/completion (seconds). 70B + large context needs more than 120s.
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", BASE_DIR / "chroma_db"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Hybrid retrieval (dense + BM25, reciprocal rank fusion)
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() in ("1", "true", "yes")
HYBRID_VECTOR_CANDIDATES = int(os.getenv("HYBRID_VECTOR_CANDIDATES", "20"))
HYBRID_BM25_CANDIDATES = int(os.getenv("HYBRID_BM25_CANDIDATES", "20"))
HYBRID_FUSION_TOP_K = int(os.getenv("HYBRID_FUSION_TOP_K", "8"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
# Rebuild BM25 only when chunk count changes by this much (avoids rebuild every ingest batch)
HYBRID_BM25_REBUILD_MIN_DELTA = int(os.getenv("HYBRID_BM25_REBUILD_MIN_DELTA", "2000"))

# Comma-separated substrings; files whose name contains any (case-insensitive) are skipped at scan time
_default_skip = "music software directory"
SKIP_FILENAME_SUBSTRINGS: tuple[str, ...] = tuple(
    s.strip().lower()
    for s in os.getenv("SKIP_FILENAME_SUBSTRINGS", _default_skip).split(",")
    if s.strip()
)

OUTLOOK_CLIENT_ID = os.getenv("OUTLOOK_CLIENT_ID", "")
OUTLOOK_TENANT_ID = os.getenv("OUTLOOK_TENANT_ID", "")
OUTLOOK_REDIRECT_URI = os.getenv("OUTLOOK_REDIRECT_URI", "http://localhost:8400")
TOKEN_CACHE_PATH = BASE_DIR / ".token_cache.json"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv",
    ".xlsx", ".xls", ".pptx", ".html", ".htm", ".eml",
    ".json", ".xml", ".rtf", ".log",
    # Image files (OCR)
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".bmp", ".gif", ".webp", ".heic",
}
