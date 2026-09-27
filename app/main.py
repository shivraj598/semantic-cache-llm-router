"""FastAPI wrapper for the baseline RAG pipeline."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.db import init_db
from app.rag import answer

app = FastAPI(title="semantic-cache-llm-router (phase-0 baseline)")


class Ask(BaseModel):
    query: str


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/answer")
def answer_route(body: Ask) -> dict:
    return answer(body.query)
