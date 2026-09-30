"""Validate the rule-based router on 100 hand-labeled queries.

Reports accuracy, confusion, route split, and projected cost vs an
always-large baseline. Optional tiny-LLM comparison (costs real API calls):

    python -m eval.router_eval [--out eval/results/router_eval.csv]
    python -m eval.router_eval --llm-compare [--out eval/results/router_eval_llm.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_labels(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def llm_classify(query: str) -> tuple[str | None, int, int]:
    """Tiny-LLM classifier via MODEL_SMALL. Returns (label, in_tok, out_tok)."""
    import litellm

    from app import config

    kwargs: dict = {
        "model": config.MODEL_SMALL,
        "messages": [{
            "role": "user",
            "content": ("Classify this search query as exactly one word, "
                        "either simple or complex. Complex means it needs "
                        "comparison, explanation, or multi-step reasoning.\n"
                        f"Query: {query}")}],
    }
    kwargs.update(config.extra_kwargs_for(config.MODEL_SMALL))
    key = config.api_key_for(config.MODEL_SMALL)
    if key:
        kwargs["api_key"] = key
    out = litellm.completion(**kwargs)
    text = out["choices"][0]["message"]["content"].strip().lower()
    usage = out.get("usage", {}) or {}
    m = re.search(r"\b(simple|complex)\b", text)
    return ((m.group(1) if m else None),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0))


def avg_baseline_tokens() -> tuple[int, int]:
    """Mean input/output tokens from the last baseline run (fallback: estimates)."""
    for name in ("baseline_llm.csv", "baseline.csv"):
        p = ROOT / "eval" / "results" / name
        if p.exists():
            with p.open() as f:
                rows = list(csv.DictReader(f))
            if rows:
                it = sum(int(r.get("input_tokens", 0)) for r in rows) / len(rows)
                ot = sum(int(r.get("output_tokens", 0)) for r in rows) / len(rows)
                return int(it), int(ot)
    return 1650, 300


def main(out: str = "eval/results/router_eval.csv", llm_compare: bool = False) -> None:
    from app import config, router
    from app.rag import retrieve

    labels = load_labels(ROOT / "eval" / "router_labels.jsonl")
    avg_in, avg_out = avg_baseline_tokens()
    rows = []
    llm_ok, llm_in, llm_out = 0, 0, 0
    for item in labels:
        hits = retrieve(item["query"])
        dec = router.classify(item["query"], hits)
        pred = dec["route"]
        row = {"id": item["id"], "query": item["query"], "label": item["label"],
               "predicted": pred, "correct": int(pred == item["label"]),
               "complexity": dec["complexity"], "reasons": "; ".join(dec["reasons"])}
        if llm_compare:
            try:
                lab, it, ot = llm_classify(item["query"])
                llm_in += it
                llm_out += ot
                row["llm_predicted"] = lab or "parse_fail"
                row["llm_correct"] = int(lab == item["label"])
                llm_ok += row["llm_correct"]
            except Exception as e:
                row["llm_predicted"] = f"ERROR: {e}"
                row["llm_correct"] = 0
        rows.append(row)
        mark = "ok" if row["correct"] else "MISS"
        print(f"{item['id']} [{mark}] label={item['label']:<7} pred={pred:<7} {item['query'][:60]}")

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    tp = sum(1 for r in rows if r["label"] == "complex" and r["predicted"] == "complex")
    fp = sum(1 for r in rows if r["label"] == "simple" and r["predicted"] == "complex")
    fn = sum(1 for r in rows if r["label"] == "complex" and r["predicted"] == "simple")
    n_complex = sum(1 for r in rows if r["label"] == "complex")
    n_simple = n - n_complex
    n_pred_small = sum(1 for r in rows if r["predicted"] == "simple")

    base = n * config.price_for(config.MODEL_LARGE, avg_in, avg_out)
    routed = (n_pred_small * config.price_for(config.MODEL_SMALL, avg_in, avg_out)
              + (n - n_pred_small) * config.price_for(config.MODEL_LARGE, avg_in, avg_out))
    print(f"\nrule-based: acc={acc:.3f} ({sum(r['correct'] for r in rows)}/{n}) "
          f"complex recall={tp/max(1,n_complex):.3f} simple precision={(n_simple-fp)/max(1,n_simple):.3f}")
    print(f"route split: {n_pred_small} small / {n-n_pred_small} large "
          f"-> projected cost ${routed:.4f} vs always-large ${base:.4f} "
          f"({(1-routed/base)*100:.1f}% cheaper, avg {avg_in}/{avg_out} tok)")
    if fn:
        print("complex-as-simple misses (escalation is the safety net):")
        for r in rows:
            if r["label"] == "complex" and r["predicted"] == "simple":
                print(f"  {r['id']}: {r['query'][:70]}")
    if llm_compare:
        llm_cost = config.price_for(config.MODEL_SMALL, llm_in, llm_out)
        print(f"tiny-LLM: acc={llm_ok/n:.3f} ({llm_ok}/{n}) classifier cost=${llm_cost:.4f}")

    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/router_eval.csv")
    ap.add_argument("--llm-compare", action="store_true")
    main(**vars(ap.parse_args()))
