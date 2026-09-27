# Semantic Caching + Cost-Aware LLM Routing

A RAG service that answers questions over a document corpus while spending as
little as possible: repeat questions are served from a semantic cache, easy
questions go to a small cheap model, hard ones to a large model — with savings
proven on a dashboard backed by repeatable eval runs.

## How it works

```
query
  → normalize + embed (all-MiniLM-L6-v2, 384-dim)
  → semantic cache lookup: nearest past query above similarity threshold?
      YES (and corpus version matches) → return cached answer + sources
      NO  → classify query as simple / complex
              → simple → SMALL model, escalate to LARGE if answer looks weak
              → complex → LARGE model
            → store answer + embedding in cache, log request
```

- **Semantic cache**: normalized query embeddings in Qdrant; a hit returns the
  stored answer without any LLM call. Entries carry a `corpus_version` (bumped
  on re-ingest) plus a TTL, so stale answers never outlive their data. Errors,
  empty answers, and user-specific answers are never cached.
- **Threshold tuning**: the similarity threshold (swept 0.85–0.97) is picked on
  50 paraphrase pairs that must hit ("how do I reset my password" /
  "password reset steps") and 50 near-miss pairs that must not
  ("refund policy in the EU" / "refund policy in the US") — zero false hits.
- **Router**: rule-based signals (query length, entities, words like
  compare/why/explain, retrieval score spread), validated against 100
  hand-labeled queries, compared with a tiny LLM classifier on cost vs accuracy.
- **Escalation**: if the small model's answer looks weak (low retrieval scores,
  no citation, "I don't know"), the query reruns on the large model —
  quality stays flat while most traffic stays cheap.
- **Dashboard**: cost per request, cache hit rate, route split, p50/p95
  latency, and quality per route over time.

## Current results

| Metric | Value |
|---|---|
| Corpus | 60 Wikipedia docs (Python ecosystem), 577 chunks |
| Eval set | 50 hand-checked Qs (`eval/questions.jsonl`) |
| Retrieval hit rate | **0.960** (48/50, source doc in top-5) |
| Answer quality (LLM judge) | **0.893** avg, 50/50 judged |
| Total cost, 50 Qs | **$0.3229** |
| Answer model | `groq/openai/gpt-oss-120b` via LiteLLM |
| Judge / small model | `groq/openai/gpt-oss-20b` via LiteLLM |

## Quickstart

```bash
cd semantic-cache-llm-router
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env   # add GROQ_API_KEY (never commit .env)
docker compose up -d   # Postgres request logs (optional; logging degrades gracefully)

./.venv/bin/python -m app.ingest --rebuild   # fetch corpus → chunk → embed → Qdrant
./.venv/bin/python -m eval.run --out eval/results/baseline_llm.csv
./.venv/bin/uvicorn app.main:app --reload
```

`POST /answer` with `{"query": "..."}` returns
`{answer, sources, model, route, cache_hit, cost_usd, latency_ms}`.

## Configuration (`.env`)

| Var | Default | Notes |
|---|---|---|
| `MODEL_SMALL` | `groq/openai/gpt-oss-20b` | Cheap route + eval judge |
| `MODEL_LARGE` | `groq/openai/gpt-oss-120b` | Complex route + baseline answers |
| `GROQ_API_KEY` | — | Required for LLM answers/judge |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | — | Any OpenAI-compatible provider (Grok, Ollama); falls back to `GROQ_API_KEY` |
| `DATABASE_URL` | local Postgres | Best-effort; requests never fail if DB is down |
| `CORPUS_VERSION` | `v1` | Bumped on re-ingest; cache entries are keyed on it |
| `PRICE_*_PER_1M` | OpenAI-like | Override to match your provider |

No key? The pipeline still runs: extractive fallback answers + keyword-overlap
scoring, so ingest, retrieval, and caching work fully offline.

## Evaluation

- `eval/questions.jsonl`: 50 questions, each with reference answer + source doc.
- `eval/run.py`: replays every question through the pipeline, records
  `retrieval_hit` (expected doc in top-5) and `quality` (LLM judge 0–1,
  keyword overlap if no key), writes CSV to `eval/results/`.
- Rerun after every change and compare against the baseline CSV:
  `python -m eval.run --out eval/results/<name>.csv`.
- Measure on realistic traffic: replay the eval set **plus paraphrased
  versions** (repeat traffic is where the cache pays off), and report cost
  reduction %, p95 change, and quality delta vs the always-large baseline.

## Layout

```
app/            FastAPI + RAG pipeline (config, ingest, rag, llm, db, main)
eval/           questions.jsonl + run.py scorer → results/*.csv
qdrant_data/    embedded Qdrant vectors (gitignored, rebuild via ingest)
docker-compose.yml  Postgres for request logs
```

## In one line

> Added semantic caching and cost-aware model routing to a RAG service, cutting
> LLM cost by X% and p95 latency by Y% with no drop in answer quality on a
> Z-question eval set.
