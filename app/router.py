"""Phase-2 cost-aware router: rule-based simple/complex classification.

Signals (per README): query length, entities, trigger words
(compare/why/explain/...), retrieval score spread. Deterministic and
offline-capable; validated against hand labels in Phase-3.
"""
from __future__ import annotations

import re

COMPLEX_WORDS = frozenset({
    "compare", "contrast", "versus", "vs", "why", "explain", "analyse",
    "analyze", "evaluate", "difference", "differences", "differ", "differs", "tradeoff",
    "trade-off", "pros", "cons", "design", "architect", "debug",
    "optimize", "optimise", "migrate", "war", "vs.",
})

WEAK_PATTERNS = (
    "i don't know", "i do not know", "don't know", "do not know",
    "insufficient context", "insufficient information", "cannot answer",
    "can't answer", "no information", "not enough information",
    "unable to answer",
)

CITATION_RE = re.compile(r"\[\d+\]")
_WORD_RE = re.compile(r"[a-z0-9']+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]+\b|[a-z]+_[a-z_]+|\(\)")


def _words(query: str) -> list[str]:
    return _WORD_RE.findall(query.lower())


def classify(query: str, hits: list[dict] | None = None) -> dict:
    """Return {route, complexity, reasons}. route is 'simple' or 'complex'."""
    hits = hits or []
    words = _words(query)
    reasons: list[str] = []
    complexity = 0

    if COMPLEX_WORDS & set(words):
        found = sorted(COMPLEX_WORDS & set(words))
        complexity += 2
        reasons.append(f"trigger words: {', '.join(found)}")

    # Lazy import so router never creates an import cycle with config consumers.
    from app import config

    if len(words) > config.ROUTER_SIMPLE_MAX_WORDS:
        complexity += 1
        reasons.append(f"long query ({len(words)} words)")

    if query.count("?") > 1 or ";" in query or " and " in query.lower():
        complexity += 1
        reasons.append("multi-part question")

    entities = set(_ENTITY_RE.findall(query))
    if len(entities) >= 3:
        complexity += 1
        reasons.append(f"many entities ({len(entities)})")

    scores = [h.get("score", 0.0) for h in hits if isinstance(h.get("score"), (int, float))]
    if scores:
        top, bottom = max(scores), min(scores)
        if top < config.ROUTER_LOW_SCORE:
            complexity += 1
            reasons.append(f"low top retrieval score ({top:.2f})")
        elif (top - bottom) < 0.08 and top < 0.70:
            complexity += 1
            reasons.append(f"flat retrieval spread ({top:.2f}-{bottom:.2f})")

    route = "complex" if complexity >= 2 else "simple"
    return {"route": route, "complexity": complexity, "reasons": reasons}


def is_weak_answer(answer: str, hits: list[dict] | None = None) -> bool:
    """True when a SMALL-model answer should escalate to LARGE."""
    if not answer or not answer.strip():
        return True
    low = answer.lower()
    if any(p in low for p in WEAK_PATTERNS):
        return True
    if not CITATION_RE.search(answer):
        return True
    from app import config

    scores = [h.get("score", 0.0) for h in (hits or [])
              if isinstance(h.get("score"), (int, float))]
    if scores and max(scores) < config.ROUTER_LOW_SCORE:
        return True
    return False
