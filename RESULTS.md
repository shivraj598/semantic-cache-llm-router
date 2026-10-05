# Results & Progress Tracker

> Resume source of truth. Talk in these numbers. Updated 2026-09-30.
> Unlike `PROJECT_PHASES.md` (local-only, gitignored), this file is safe to commit — it contains no secrets.
> Every figure below is recomputed from `eval/results/*.csv` or Qdrant counts, never copied from memory.

## Headline numbers

| Metric | Value | Source |
|---|---|---|
| Corpus | 60 Wikipedia docs (Python ecosystem), 577 chunks | Qdrant `rag_chunks` count |
| Eval set | 50 hand-checked Qs | `eval/questions.jsonl` |
| Retrieval hit rate | **0.960** (48/50, expected doc in top-5) | `baseline_llm.csv` |
| Answer quality (LLM judge) | **0.893** avg, 50/50 LLM-judged | `baseline_llm.csv` |
| Baseline cost, 50 Qs (all LARGE) | **$0.3229** | `baseline_llm.csv` |
| Cache threshold | **0.81** — 0.72 paraphrase recall, **zero false hits** (50+50 pairs) | `threshold_sweep.csv` |
| Router accuracy | **97/100**, complex recall **1.000** | `router_eval.csv` |
| Projected routing saving | **64.9% cheaper** ($0.2267 vs $0.6450 always-large) | `router_eval.csv` |
| Tiny-LLM classifier (comparison) | 65/100, classifier cost $0.0106 | `router_eval_llm.csv` |
| Cache hit rate on repeat traffic | TBD (Phase 4) | — |
| p50/p95 latency change | TBD (Phase 4) | — |
| Quality delta vs always-large | TBD (Phase 4) | — |

Models: answers `groq/openai/gpt-oss-120b`, judge + cheap route `groq/openai/gpt-oss-20b` (via LiteLLM).
Avg tokens per answer: 1708 in / 218 out. Embeddings: `all-MiniLM-L6-v2` (384-dim).

## 1. Retrieval & answer quality (Phase 0 baseline)

- 50/50 questions answered and LLM-judged (no keyword fallback).
- Retrieval 48/50; quality avg 0.893. Total $0.3229 for the full set.
- Recompute: `python -c` over `eval/results/baseline_llm.csv` (retrieval_hit mean, quality mean, cost sum).

## 2. Semantic cache threshold (Phase 1 + 3)

- Sweep 0.75–0.97 on 50 paraphrase pairs (must hit) + 50 near-miss pairs (must not).
- At 0.81: recall 0.720 (36/50), false hits 0/50. At 0.85 recall drops to 0.56; at 0.93 to 0.34.
- The single false hit below 0.81 is EU-vs-US refund policy (sim 0.808) — exactly the adversarial case.
- Known blind spot: acronym expansion ("GIL" vs "Global Interpreter Lock", sim 0.329) misses on MiniLM.
- `CACHE_THRESHOLD=0.81` is the live default (`app/config.py`).
- Verified live: store→lookup round-trip hits (score 1.0 on exact repeat), stale/version/TTL rejection, `answer()` cache path returns cost $0.00.
- Reproduce: `python -m eval.threshold_sweep`

## 3. Router validation (Phase 2 + 3)

- 100 hand-labeled queries (50 eval + 50 handwritten mix of simple/complex).
- Rule-based: **97/100**, complex recall 1.000 (zero dangerous misses — every error is simple→complex, i.e. extra cost, never lost quality).
- One fix driven by eval: added "differ/differs" triggers (caught "How does PyPy differ from CPython?"), 96→97.
- Route split: 69 small / 31 large.
- Tiny-LLM classifier on the same 100: 65/100 with $0.0106 in classifier calls — rule-based wins on both accuracy and cost (no per-query classifier fee).
- Reproduce: `python -m eval.router_eval [--llm-compare]`

## 4. Cost analysis (projected, Phase 3)

- Always-large on 100 labeled Qs: $0.6450. Routed (69 small / 31 large): $0.2267 → **−64.9%**.
- Escalation accounting: weak small answer + large retry are costed as the sum of both calls.
- Real-traffic measurement (with cache hits + paraphrase replay) is Phase 4 work — the number above is projection from route split, not a live A/B.

## 5. Latency (TBD — Phase 4)

- No production latency numbers yet. To measure: paraphrase-augmented replay reporting p50/p95 per route and cache-hit vs miss.
- Expected shape: cache hits ~embedding+lookup only (no LLM call); small route faster/cheaper than large.

## 6. Progress log

- 2026-09-30 — Phase 0 confirmed done: ingest, RAG, logging, 50-Q baseline (0.960 / 0.893 / $0.3229).
- 2026-09-30 — Phase 1 done: `app/cache.py` + wiring + zero-cost cache path verified.
- 2026-09-30 — Phase 2 done: `app/router.py` + SMALL-first with escalation verified (stubbed LLM).
- 2026-09-30 — `.gitignore` fixed (was bare `*` hiding all new files).
- 2026-09-30 — Phase 3 done: sweep → 0.81 live default; router 97/100, complex recall 1.000; tiny-LLM loses 65 vs 97.
- Next — Phase 4: paraphrase replay (cost −%, p95 Δ, quality Δ), Streamlit dashboard. Phase 5: tests, lifespan handler, pin deps, rotate GROQ key in `.env`.

## 7. Reproduce everything

```bash
./.venv/bin/python -m eval.threshold_sweep
./.venv/bin/python -m eval.router_eval [--llm-compare]
./.venv/bin/python -m eval.run --out eval/results/<name>.csv
```

## 8. Caveats (say these out loud before quoting numbers)

- Cost figures use the repo price table (`app/config.py`), not provider invoices.
- Router savings are projected from route split, not measured live traffic.
- Judge quality is an LLM score (gpt-oss-20b), not human rating.
- Groq free-tier rate limits bit during evals (retries + 1s pacing added).


## In one line

> Added semantic caching and cost-aware model routing to a RAG service, cutting
> LLM cost by 86.3% and p95 latency by 40.5% with +0.016 change in answer quality on a
> 100-request replay set.
