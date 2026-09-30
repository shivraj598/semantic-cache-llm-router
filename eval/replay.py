"""Phase 4 realistic-traffic replay: pipeline (cache + router) vs always-large.

Traffic: each eval question fires once, then its paraphrase fires right after
(what the semantic cache exists for). Compares three arms:

  pipeline      the real thing: cache-first + routed small/large + escalation
  nocache       same router, cache disabled (isolates the cache contribution)
  always-large  every query answered by MODEL_LARGE (the Phase-0 baseline arm)

Always-large runs with CACHE OFF so the baseline arm reflects "no caching"
(the deployment configuration it models), not a to-hit-only policy.

Judge = MODEL_SMALL scoring 0..1 vs reference; identical answers are
short-circuited to 1.0 without a call (cache-hit answers repeat the exact
stored text, so they are never judged twice).

Results go to eval/results/replay.csv plus per-arm aggregate rows prefixed
with `__agg__` in the same file, so the dashboard and RESULTS.md recompute
every headline number from the CSV alone.

Usage:
    python -m eval.replay [--out eval/results/replay.csv] [--no-llm-judge]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG_PREFIX = "__agg__"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_paraphrases(path: Path) -> dict[str, str]:
    return {p["id"]: p["paraphrase"] for p in load_questions(path)}


def llm_judge(question: str, reference: str, answer: str,
              retries: int = 3) -> float | None:
    import re

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
    for attempt in range(retries):
        try:
            out = litellm.completion(**kwargs)["choices"][0]["message"]["content"].strip()
            m = re.search(r"0?\.\d+|\b[01]\b", out)
            return max(0.0, min(1.0, float(m.group()))) if m else None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def keyword_score(reference: str, answer: str) -> float:
    import re
    stop = {"what", "is", "the", "a", "an", "for", "with", "and", "are",
            "was", "were", "does", "how", "why", "which", "that", "this"}
    tok = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop}
    r, a = tok(reference), tok(answer)
    if not r:
        return 0.0
    return round(len(r & a) / len(r), 3)


class Judge:
    """LLM judge with a per-arm exact-answer cache (identical answer = 1.0)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.cache: dict[str, float] = {}

    def score(self, q: dict, answer_text: str) -> tuple[float, str]:
        if not answer_text or not self.enabled:
            return 0.0, "keyword"
        if answer_text in self.cache:
            return self.cache[answer_text], "llm_cached"
        s = llm_judge(q["question"], q["reference"], answer_text)
        if s is None:
            return keyword_score(q["reference"], answer_text), "keyword"
        self.cache[answer_text] = s
        return s, "llm"


def run_arm(arm: str, qs: list[dict], paras: dict[str, str],
            rows: list[dict]) -> None:
    from app import config
    from app.rag import answer as rag_answer

    judge = Judge(enabled=True)
    use_cache = arm == "pipeline"  # nocache + always-large run cache-off
    prev_force = config.FORCE_MODEL
    if arm == "always-large":
        config.FORCE_MODEL = "large"
    try:
        for q in qs:
            for variant in ("orig", "para"):
                text = q["question"] if variant == "orig" else paras.get(q["id"], "")
                if not text:
                    continue
                t0 = time.perf_counter()
                res = rag_answer(text, use_cache=use_cache)
                latency_ms = (time.perf_counter() - t0) * 1000
                quality, judge_kind = judge.score(q, res.get("answer", ""))
                rows.append({
                    "arm": arm, "qid": q["id"], "variant": variant,
                    "query": text[:200],
                    "route": res.get("route", "?"), "model": res.get("model", ""),
                    "escalated": res.get("escalated", False),
                    "cache_hit": res.get("cache_hit", False),
                    "cache_score": round(res.get("cache_score", 0.0), 4),
                    "retrieval_hit": int(q["source_doc"] in
                                         [s.get("doc_title", "") for s in res.get("sources", [])]),
                    "quality": round(quality, 3), "judge": judge_kind,
                    "cost_usd": round(res.get("cost_usd", 0.0), 6),
                    "latency_ms": round(res.get("latency_ms", 0.0), 1),
                })
                r = rows[-1]
                print(f"{arm:>12} {q['id']} {variant:<4} route={r['route']:<15} "
                      f"hit={r['retrieval_hit']} q={r['quality']} "
                      f"${r['cost_usd']:.4f} {r['latency_ms']:.0f}ms", flush=True)
    finally:
        config.FORCE_MODEL = prev_force


def summarize(rows: list[dict]) -> list[dict]:
    """Per-arm aggregate rows appended to the same CSV."""
    out: list[dict] = []
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        n = len(sub)
        lats = sorted(r["latency_ms"] for r in sub)

        def pct(p: float) -> float:
            i = max(0, min(len(lats) - 1, round(p / 100 * (len(lats) - 1))))
            return lats[i]

        out.append({
            "arm": f"{AGG_PREFIX}{arm}", "qid": "", "variant": "agg", "query": "",
            "route": "agg", "model": "agg", "escalated": "",
            "cache_hit": "",
            "cache_score": "",
            "retrieval_hit": round(sum(r["retrieval_hit"] for r in sub) / n, 3),
            "quality": round(sum(r["quality"] for r in sub) / n, 3),
            "judge": f"llm={sum(1 for r in sub if r['judge'].startswith('llm'))}/{n}",
            "cost_usd": round(sum(r["cost_usd"] for r in sub), 4),
            "latency_ms": round(sum(r["latency_ms"] for r in sub) / n, 1),
            "n": n,
            "cache_hit_rate": round(sum(1 for r in sub if r["cache_hit"]) / n, 3),
            "p50_ms": pct(50), "p95_ms": pct(95),
            "cost_per_req_usd": round(sum(r["cost_usd"] for r in sub) / n, 6),
            "routes_small": sum(1 for r in sub if r["route"] == "small"),
            "routes_large": sum(1 for r in sub if r["route"] == "large"),
            "routes_escalated": sum(1 for r in sub if r["route"] == "small_escalated"),
            "routes_cache": sum(1 for r in sub if r["route"] == "cache"),
        })
    return out


def write_csv(path: Path, rows: list[dict], agg: list[dict]) -> None:
    fields = ["arm", "qid", "variant", "query", "route", "model", "escalated",
              "cache_hit", "cache_score", "retrieval_hit", "quality", "judge",
              "cost_usd", "latency_ms", "n", "cache_hit_rate", "p50_ms", "p95_ms",
              "cost_per_req_usd", "routes_small", "routes_large",
              "routes_escalated", "routes_cache"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        w.writerows(agg)


def main(out: str = "eval/results/replay.csv", no_llm_judge: bool = False) -> None:
    qs = load_questions(ROOT / "eval" / "questions.jsonl")
    paras = load_paraphrases(ROOT / "eval" / "paraphrases.jsonl")
    rows: list[dict] = []
    for arm in ("pipeline", "nocache", "always-large"):
        run_arm(arm, qs, paras, rows)
    agg = summarize(rows)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_csv(outp, rows, agg)

    print("\n=== summary (paraphrase-augmented replay, 100 requests/arm) ===")
    base = next((a for a in agg if a["arm"] == f"{AGG_PREFIX}always-large"), None)
    for a in agg:
        name = a["arm"].replace(AGG_PREFIX, "")
        extra = ""
        if base and a is not base and base["cost_usd"]:
            extra = (f"  vs always-large: cost {(1 - a['cost_usd'] / base['cost_usd']) * 100:+.1f}%"
                     f" | p95 {(a['p95_ms'] - base['p95_ms']) / max(base['p95_ms'], 1e-9) * 100:+.1f}%"
                     f" | quality {a['quality'] - base['quality']:+.3f}")
        print(f"{name:>13}: n={a['n']} cache_hit_rate={a['cache_hit_rate']:.2f} "
              f"quality={a['quality']:.3f} cost=${a['cost_usd']:.4f} "
              f"p50={a['p50_ms']:.0f}ms p95={a['p95_ms']:.0f}ms{extra}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/replay.csv")
    ap.add_argument("--no-llm-judge", action="store_true")
    main(**vars(ap.parse_args()))
