"""Phase-1 semantic cache: normalized query embeddings in Qdrant.

Hit rule: nearest past query >= CACHE_THRESHOLD AND corpus_version matches
AND within CACHE_TTL_SECONDS -> return cached answer with no LLM call.
Never caches errors, empty answers, or user-specific answers.
"""
from __future__ import annotations

import logging
import re
import time
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config

log = logging.getLogger(__name__)

_USER_SPECIFIC_RE = re.compile(
    r"\b(my|mine|my account|my email|my password|user_id|account id)\b",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?!.,;:]+$", "", q).strip()
    return q


def _is_cacheable(answer: str, sources: list | None = None) -> bool:
    if not answer or not answer.strip():
        return False
    a = answer.strip()
    if len(a) < 10:
        return False
    if a.startswith("ERROR:"):
        return False
    if _USER_SPECIFIC_RE.search(a):
        return False
    return True


def ensure_cache_collection(client) -> None:
    if not client.collection_exists(config.CACHE_COLLECTION):
        client.create_collection(
            config.CACHE_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )


def lookup(query: str, embedding: list[float] | None = None) -> dict | None:
    """Return cached entry or None. Never raises (miss on any error)."""
    try:
        from app.rag import get_client, get_model

        client = get_client()
        ensure_cache_collection(client)
        qnorm = normalize_query(query)
        if not qnorm:
            return None
        qv = embedding if embedding is not None else get_model().encode([qnorm]).tolist()[0]
        res = client.query_points(config.CACHE_COLLECTION, query=qv, limit=1)
        if not res.points:
            return None
        top = res.points[0]
        if float(top.score) < config.CACHE_THRESHOLD:
            return None
        p = top.payload or {}
        if p.get("corpus_version", "") != config.CORPUS_VERSION:
            return None
        created = float(p.get("created_at", 0) or 0)
        if created and (time.time() - created) > config.CACHE_TTL_SECONDS:
            return None
        answer = (p.get("answer") or "").strip()
        if not _is_cacheable(answer):
            return None
        return {
            "answer": answer,
            "sources": p.get("sources", []),
            "model": p.get("model", ""),
            "query_norm": p.get("query_norm", qnorm),
            "score": float(top.score),
        }
    except Exception as e:
        log.warning("cache lookup miss (error): %s", e)
        return None


def store(query: str, answer: str, sources: list, model: str,
          embedding: list[float] | None = None) -> bool:
    """Store a Q/A pair. Returns False if skipped. Never raises."""
    try:
        if not _is_cacheable(answer, sources):
            return False
        from app.rag import get_client, get_model

        client = get_client()
        ensure_cache_collection(client)
        qnorm = normalize_query(query)
        if not qnorm:
            return False
        qv = embedding if embedding is not None else get_model().encode([qnorm]).tolist()[0]
        client.upsert(
            config.CACHE_COLLECTION,
            points=[PointStruct(
                id=uuid.uuid4().hex,
                vector=qv,
                payload={
                    "query": query[:2000],
                    "query_norm": qnorm,
                    "answer": answer[:8000],
                    "sources": sources,
                    "model": model,
                    "corpus_version": config.CORPUS_VERSION,
                    "created_at": time.time(),
                },
            )],
        )
        return True
    except Exception as e:
        log.warning("cache store skipped: %s", e)
        return False
