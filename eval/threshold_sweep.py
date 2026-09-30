"""Sweep the semantic-cache similarity threshold (0.85-0.97).

Must-hit:  eval/paraphrases.jsonl (paraphrase of a cached query -> hit).
Must-not: eval/near_miss.jsonl (different meaning despite overlap -> miss).

Picks the highest threshold with zero false hits on near-miss pairs and
reports paraphrase recall there. Run offline (local embeddings only):

    python -m eval.threshold_sweep [--out eval/results/threshold_sweep.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [round(0.75 + 0.01 * i, 2) for i in range(23)]  # 0.75..0.97


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(out: str = "eval/results/threshold_sweep.csv") -> None:
    from app.cache import normalize_query
    from app.rag import get_model

    paras = load(ROOT / "eval" / "paraphrases.jsonl")
    nears = load(ROOT / "eval" / "near_miss.jsonl")
    texts = [normalize_query(p["original"]) for p in paras]
    texts += [normalize_query(p["paraphrase"]) for p in paras]
    texts += [normalize_query(n["query_a"]) for n in nears]
    texts += [normalize_query(n["query_b"]) for n in nears]
    vecs = get_model().encode(texts, show_progress_bar=False)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    n = len(paras)
    para_sims = [cosine(vecs[i], vecs[n + i]) for i in range(n)]
    m = len(nears)
    near_sims = [cosine(vecs[2 * n + i], vecs[2 * n + m + i]) for i in range(m)]

    rows = []
    for t in THRESHOLDS:
        recall = sum(s >= t for s in para_sims) / n
        fpr = sum(s >= t for s in near_sims) / m
        rows.append({"threshold": t, "paraphrase_recall": round(recall, 3),
                     "near_miss_fpr": round(fpr, 3),
                     "missed_paraphrases": sum(s < t for s in para_sims),
                     "false_hits": sum(s >= t for s in near_sims)})
    print(f"{'thr':>6} {'recall':>7} {'fpr':>6} {'missed':>7} {'false_hit':>9}")
    for r in rows:
        print(f"{r['threshold']:>6.2f} {r['paraphrase_recall']:>7.3f} "
              f"{r['near_miss_fpr']:>6.3f} {r['missed_paraphrases']:>7d} {r['false_hits']:>9d}")

    zero_fp = [r for r in rows if r["false_hits"] == 0]
    if not zero_fp:
        best = min(rows, key=lambda r: (r["false_hits"], -r["paraphrase_recall"]))
        print(f"\nWARNING: no threshold gives zero false hits; "
              f"closest: {best['threshold']:.2f} "
              f"(recall={best['paraphrase_recall']:.3f}, false={best['false_hits']})")
    else:
        best = max(zero_fp, key=lambda r: r["paraphrase_recall"])
        print(f"\nRecommended CACHE_THRESHOLD={best['threshold']:.2f} "
              f"(recall={best['paraphrase_recall']:.3f}, zero false hits on {m} near-miss pairs)")
        worst = sorted(((s, paras[i]["id"]) for i, s in enumerate(para_sims)
                        if s < best["threshold"]), reverse=True)[:5]
        for s, pid in worst:
            print(f"  missed {pid}: sim={s:.3f}")

    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/threshold_sweep.csv")
    main(**vars(ap.parse_args()))
