"""Best-effort Postgres request logging. Never crashes the request path."""
from __future__ import annotations

import json
import logging

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
)
from sqlalchemy.exc import SQLAlchemyError

from app import config

log = logging.getLogger(__name__)
metadata = MetaData()

requests_log = Table(
    "requests_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), server_default=func.now()),
    Column("query", Text, nullable=False),
    Column("model", String(256), default=""),
    Column("route", String(64), default="baseline"),
    Column("cache_hit", Boolean, default=False),
    Column("input_tokens", Integer, default=0),
    Column("output_tokens", Integer, default=0),
    Column("cost_usd", Float, default=0.0),
    Column("latency_ms", Float, default=0.0),
    Column("retrieval_hit", Boolean, nullable=True),
    Column("quality", Float, nullable=True),
    Column("answer", Text, default=""),
    Column("sources", Text, default="[]"),  # JSON list
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
    return _engine


def init_db() -> bool:
    try:
        metadata.create_all(get_engine())
        return True
    except SQLAlchemyError as e:
        log.warning("Postgres unavailable, logging disabled: %s", e)
        return False


def log_request(**row) -> None:
    """Insert one log row; swallow errors so RAG never fails on logging."""
    try:
        src = row.get("sources", [])
        if not isinstance(src, str):
            row["sources"] = json.dumps(src)
        with get_engine().begin() as conn:
            conn.execute(requests_log.insert().values(**row))
    except SQLAlchemyError as e:
        log.warning("log_request skipped (db down): %s", e)
