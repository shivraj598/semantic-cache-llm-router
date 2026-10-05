"""FastAPI wrapper for the RAG pipeline."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.db import init_db
from app.rag import answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="semantic-cache-llm-router")


class Ask(BaseModel):
    query: str
    use_cache: bool = True


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/answer")
def answer_route(body: Ask) -> dict:
    return answer(body.query, use_cache=body.use_cache)
