"""Plain Phase-0 RAG: embed query -> Qdrant top-K -> LARGE model -> log."""
from __future__ import annotations

import time

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app import config
from app.db import init_db, log_request
from app.llm import generate

_model = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=config.QDRANT_PATH)
    return _client


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    qv = get_model().encode([query]).tolist()[0]
    hits = get_client().query_points(
        config.RAG_COLLECTION, query=qv, limit=top_k).points
    return [{"doc_title": h.payload.get("doc_title", ""), "text": h.payload.get("text", ""),
             "score": float(h.score)} for h in hits]


def answer(query: str) -> dict:
    """Baseline answer: always LARGE model, log everything for before/after."""
    t0 = time.perf_counter()
    hits = retrieve(query)
    ctx = [h["text"] for h in hits]
    gen = generate(ctx, query, model=config.MODEL_LARGE)
    latency_ms = (time.perf_counter() - t0) * 1000
    cost = config.price_for(config.MODEL_LARGE, gen["input_tokens"], gen["output_tokens"])
    result = {
        "answer": gen["answer"],
        "sources": [{"doc_title": h["doc_title"], "score": h["score"]} for h in hits],
        "model": config.MODEL_LARGE,
        "route": "baseline",
        "cache_hit": False,
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "answer_source": gen["source"],
    }
    init_db()
    log_request(query=query, model=result["model"], route="baseline",
                cache_hit=False, input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"], cost_usd=cost,
                latency_ms=latency_ms, answer=result["answer"][:4000],
                sources=result["sources"])
    return result
