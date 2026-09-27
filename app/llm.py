"""LiteLLM wrapper over any OpenAI-compatible provider.

Falls back to extractive answer when no key / call fails so Phase-0
eval and threshold work still run offline.
"""
from __future__ import annotations

import logging

from app import config

log = logging.getLogger(__name__)

SYSTEM = (
    "You answer questions using ONLY the provided context passages. "
    "Cite sources like [1], [2]. If the context is insufficient, say you don't know."
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def generate(context_blocks: list[str], query: str, model: str) -> dict:
    prompt = (
        "Context:\n"
        + "\n\n".join(f"[{i+1}] {b[:1500]}" for i, b in enumerate(context_blocks))
        + f"\n\nQuestion: {query}\nAnswer with citations:"
    )
    in_tokens = _estimate_tokens(SYSTEM + prompt)
    try:
        import litellm

        kwargs: dict = {"model": model, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]}
        if config.OPENAI_BASE_URL:
            kwargs["api_base"] = config.OPENAI_BASE_URL
        if config.OPENAI_API_KEY:
            kwargs["api_key"] = config.OPENAI_API_KEY
        resp = litellm.completion(**kwargs)
        text = resp["choices"][0]["message"]["content"] or ""
        try:
            usage = resp.get("usage", {}) or {}
            out_tokens = int(usage.get("completion_tokens") or _estimate_tokens(text))
            in_tokens = int(usage.get("prompt_tokens") or in_tokens)
        except Exception:
            out_tokens = _estimate_tokens(text)
        return {"answer": text.strip(), "input_tokens": in_tokens,
                "output_tokens": out_tokens, "source": "llm"}
    except Exception as e:
        log.warning("LLM call failed, extractive fallback: %s", e)
        top = context_blocks[0] if context_blocks else "I don't know."
        text = f"Based on the retrieved context: {top[:800]} [1]"
        return {"answer": text, "input_tokens": in_tokens,
                "output_tokens": _estimate_tokens(text), "source": "extractive_fallback"}
