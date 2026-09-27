"""Run the 50-Q eval through the baseline RAG, score, write CSV.

Scores:
  retrieval_hit: 1 if expected source_doc appears in top-K sources else 0
  quality: LLM-judge (0..1) when API key works, else keyword-overlap fallback

Usage:
    python -m eval.run [--out eval/results/baseline.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def keyword_score(reference: str, answer: str) -> float:
    stop = {"what", "is", "the", "a", "an", "for", "with", "and", "are",
            "was", "were", "does", "how", "why", "which", "that", "this"}
    tok = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop}
    r, a = tok(reference), tok(answer)
    if not r:
        return 0.0
    return round(len(r & a) / len(r), 3)


def llm_judge(question: str, reference: str, answer: str) -> float | None:
    try:
        import litellm
        from app import config
        if not config.api_key_for(config.MODEL_SMALL):
            return None
        kwargs: dict = {
            "model": config.MODEL_SMALL,
            "messages": [{
                "role": "user",
                "content": ("Rate answer correctness 0..1 (number only).\n"
                            f"Q: {question}\nReference: {reference}\nAnswer: {answer}")}],
        }
        kwargs.update(config.extra_kwargs_for(config.MODEL_SMALL))
        key = config.api_key_for(config.MODEL_SMALL)
        if key:
            kwargs["api_key"] = key
        out = litellm.completion(**kwargs)["choices"][0]["message"]["content"].strip()
        m = re.search(r"0?\.\d+|\b[01]\b", out)
        return max(0.0, min(1.0, float(m.group()))) if m else None
    except Exception:
        return None


def main(out: str = "eval/results/baseline.csv") -> None:
    from app.rag import answer as rag_answer

    qs = load_questions(ROOT / "eval" / "questions.jsonl")
    rows = []
    for q in qs:
        try:
            res = rag_answer(q["question"])
        except Exception as e:
            res = {"answer": f"ERROR: {e}", "sources": [], "model": "error",
                   "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                   "latency_ms": 0.0}
        titles = [s.get("doc_title", "") for s in res.get("sources", [])]
        hit = 1 if q["source_doc"] in titles else 0
        score = llm_judge(q["question"], q["reference"], res.get("answer", ""))
        judge = "llm" if score is not None else "keyword"
        if score is None:
            score = keyword_score(q["reference"], res.get("answer", ""))
        rows.append({"id": q["id"], "question": q["question"],
                     "source_doc": q["source_doc"], "retrieval_hit": hit,
                     "quality": score, "judge": judge,
                     "model": res.get("model", ""), "cost_usd": round(res.get("cost_usd", 0.0), 6),
                     "latency_ms": round(res.get("latency_ms", 0.0), 1),
                     "input_tokens": res.get("input_tokens", 0),
                     "output_tokens": res.get("output_tokens", 0)})
        print(f"{q['id']} hit={hit} q={score} ({judge}) ms={rows[-1]['latency_ms']}")
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"\nwrote {outp}: n={n} "
          f"retrieval_hit_rate={sum(r['retrieval_hit'] for r in rows)/n:.3f} "
          f"avg_quality={sum(r['quality'] for r in rows)/n:.3f} "
          f"total_cost=${sum(r['cost_usd'] for r in rows):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/baseline.csv")
    main(**vars(ap.parse_args()))
