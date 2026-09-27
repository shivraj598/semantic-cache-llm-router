"""Fetch Wikipedia subset, chunk (~500 tokens + overlap), embed, store in Qdrant.

Usage (from repo root, venv active):
    python -m app.ingest [--limit 60] [--rebuild]
"""
from __future__ import annotations

import argparse
import re
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from app import config

# Single-domain subset (Python ecosystem). Missing titles are skipped.
TITLES = [
    "Python (programming language)", "CPython", "PyPy", "Cython",
    "Python Software Foundation", "Python Package Index", "pip (package manager)",
    "virtualenv", "Conda (package manager)", "Jupyter Notebook",
    "NumPy", "pandas (software)", "SciPy", "scikit-learn", "Matplotlib",
    "TensorFlow", "PyTorch", "Keras", "OpenCV", "Natural Language Toolkit",
    "FastAPI", "Flask (web framework)", "Django (web framework)", "SQLAlchemy",
    "Pydantic", "Uvicorn", "Starlette", "Requests (software)",
    "Hypertext Transfer Protocol", "Representational state transfer", "JSON",
    "WebSocket", "Docker (software)", "Kubernetes", "PostgreSQL", "SQLite",
    "Redis", "Vector database", "Embedding", "Transformer (deep learning architecture)",
    "Large language model", "Retrieval-augmented generation", "Prompt engineering",
    "Fine-tuning (deep learning)", "Global interpreter lock", "pytest",
    "Git", "GitHub", "Asyncio", "Decorator (computer science)",
    "Generator (computer programming)", "List (computer science)", "Hash table",
    "Regular expression", "Unit testing", "Continuous integration",
    "Application programming interface", "Microservices", "Virtual machine",
    "Cloud computing",
]

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "semantic-cache-llm-router/0.1 (phase-0 eval RAG; local dev)"}
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def fetch_article(title: str) -> str | None:
    try:
        r = httpx.get(API, headers=HEADERS, params={"action": "query", "prop": "extracts",
                                   "explaintext": True, "titles": title,
                                   "format": "json"}, timeout=30)
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        for page in pages.values():
            if "missing" in page:
                return None
            return page.get("extract", "")
    except Exception as e:
        print(f"skip {title}: {e}")
        return None
    return None


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    out, i, n = [], 0, len(text)
    while i < n:
        out.append(text[i:i + config.CHUNK_CHARS])
        if i + config.CHUNK_CHARS >= n:
            break
        i += config.CHUNK_CHARS - config.CHUNK_OVERLAP
    return [c for c in out if len(c.strip()) > 100]


def get_client() -> QdrantClient:
    client = QdrantClient(path=config.QDRANT_PATH)
    if not client.collection_exists(config.RAG_COLLECTION):
        client.create_collection(
            config.RAG_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
    return client


def main(limit: int = 60, rebuild: bool = False) -> None:
    client = get_client()
    if rebuild:
        client.delete_collection(config.RAG_COLLECTION)
        client.create_collection(
            config.RAG_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
    model = get_model()
    stored_docs, stored_chunks = 0, 0
    for title in TITLES[:limit]:
        text = fetch_article(title)
        if not text:
            print(f"skip (missing/empty): {title}")
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        vecs = model.encode(chunks, show_progress_bar=False).tolist()
        points = [{
            "id": uuid.uuid4().hex,
            "vector": v,
            "payload": {"doc_title": title, "text": c,
                        "corpus_version": config.CORPUS_VERSION},
        } for v, c in zip(vecs, chunks)]
        client.upsert(config.RAG_COLLECTION, points=points)
        stored_docs += 1
        stored_chunks += len(points)
        print(f"stored {title}: {len(points)} chunks")
    print(f"done: {stored_docs} docs, {stored_chunks} chunks, version={config.CORPUS_VERSION}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--rebuild", action="store_true")
    main(**vars(ap.parse_args()))
