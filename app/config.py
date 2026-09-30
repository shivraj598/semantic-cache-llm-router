"""Shared Phase-0 config. Env-overridable for Grok / OpenAI / Ollama-public."""
import os
from dotenv import load_dotenv

load_dotenv()

CORPUS_VERSION: str = os.getenv("CORPUS_VERSION", "v1")

# Qdrant (embedded mode, no server needed)
QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_data")
RAG_COLLECTION: str = os.getenv("RAG_COLLECTION", "rag_chunks")

# Embeddings
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))

# Chunking: ~500 tokens with overlap (~4 chars/token heuristic)
CHUNK_CHARS: int = int(os.getenv("CHUNK_CHARS", "2000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "400"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

# LiteLLM — OpenAI-compatible (works for OpenAI, Grok/xAI, Ollama cloud).
# Examples:
#   OpenAI: MODEL_LARGE=gpt-4o  MODEL_SMALL=gpt-4o-mini  OPENAI_BASE_URL unset
#   Grok:   MODEL_LARGE=grok-4  MODEL_SMALL=grok-3-mini  OPENAI_BASE_URL=https://api.x.ai/v1
#   Ollama: MODEL_LARGE=ollama/llama3.1  OPENAI_BASE_URL=https://ollama.com/v1 (or local http://localhost:11434/v1)
MODEL_SMALL: str = os.getenv("MODEL_SMALL", "gpt-4o-mini")
MODEL_LARGE: str = os.getenv("MODEL_LARGE", "gpt-4o")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL") or None
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY") or None


def api_key_for(model: str) -> str | None:
    """Pick the right key per provider prefix.

    groq/ models use GROQ_API_KEY; everything else uses OPENAI_API_KEY,
    falling back to GROQ_API_KEY so the OpenAI-compatible Groq endpoint
    (OPENAI_BASE_URL=https://api.groq.com/openai/v1) works with one key.
    """
    if model.startswith("groq/"):
        return GROQ_API_KEY
    return OPENAI_API_KEY or GROQ_API_KEY


def extra_kwargs_for(model: str) -> dict:
    """Provider-specific LiteLLM kwargs (base URL only for generic OpenAI)."""
    if model.startswith("groq/"):
        return {}
    return {"api_base": OPENAI_BASE_URL} if OPENAI_BASE_URL else {}

# Price table USD per 1M tokens (override per provider via env).
PRICE_SMALL_IN_PER_1M: float = float(os.getenv("PRICE_SMALL_IN_PER_1M", "0.15"))
PRICE_SMALL_OUT_PER_1M: float = float(os.getenv("PRICE_SMALL_OUT_PER_1M", "0.60"))
PRICE_LARGE_IN_PER_1M: float = float(os.getenv("PRICE_LARGE_IN_PER_1M", "2.50"))
PRICE_LARGE_OUT_PER_1M: float = float(os.getenv("PRICE_LARGE_OUT_PER_1M", "10.00"))

# Postgres logging (Docker Compose default). Logging is best-effort.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/raglogs"
)

# Phase-1 semantic cache. Threshold tuned in Phase-3 sweep (0.85-0.97).
CACHE_COLLECTION: str = os.getenv("CACHE_COLLECTION", "semantic_cache")
CACHE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.93"))
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))

BASELINE_MODEL_ROLE: str = "large"  # Phase-0 baseline always routes to LARGE


def price_for(model: str, in_tokens: int, out_tokens: int) -> float:
    """Cost in USD from the price table."""
    if model == MODEL_SMALL:
        pin, pout = PRICE_SMALL_IN_PER_1M, PRICE_SMALL_OUT_PER_1M
    else:
        pin, pout = PRICE_LARGE_IN_PER_1M, PRICE_LARGE_OUT_PER_1M
    return in_tokens / 1_000_000 * pin + out_tokens / 1_000_000 * pout
