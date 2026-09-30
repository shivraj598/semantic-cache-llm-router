"""Phase-4 ops dashboard: cost, cache hit rate, route split, latency, quality.

Sources, in priority order (first one that has data wins):
  1. eval/results/replay.csv   — Phase-4 paraphrase replay (per-arm aggregates)
  2. eval/results/*.csv        — Phase-0/3 eval outputs (baseline, router, sweep)
  3. requests_log table        — live traffic logged by the API (Postgres via
                                 DATABASE_URL; best-effort, section hides on error)

Every number is recomputed from those files/tables at render time — nothing is
hard-coded. Run:

    ./.venv/bin/streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
REPLAY_CSV = RESULTS / "replay.csv"
AGG_PREFIX = "__agg__"

st.set_page_config(page_title="RAG cost & cache dashboard",
                   page_icon=":moneybag:", layout="wide")

# ---------------------------------------------------------------- data loading


@st.cache_data(ttl=30)
def load_replay() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not REPLAY_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(REPLAY_CSV)
    agg = df[df["arm"].astype(str).str.startswith(AGG_PREFIX)].copy()
    rows = df[~df.index.isin(agg.index)].copy()
    if not agg.empty:
        agg["arm"] = agg["arm"].str.replace(AGG_PREFIX, "", regex=False)
    return rows, agg


@st.cache_data(ttl=30)
def load_result_csv(name: str) -> pd.DataFrame:
    p = RESULTS / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=30)
def load_db_log(limit: int = 5000) -> pd.DataFrame:
    try:
        from sqlalchemy import create_engine, text

        from app import config
        eng = create_engine(config.DATABASE_URL, pool_pre_ping=True)
        with eng.connect() as conn:
            return pd.read_sql(
                text("SELECT ts, query, model, route, cache_hit, input_tokens, "
                     "output_tokens, cost_usd, latency_ms, retrieval_hit, quality "
                     f"FROM requests_log ORDER BY id DESC LIMIT {limit}"), conn)
    except Exception:
        return pd.DataFrame()


def money(x: float) -> str:
    return f"${x:.4f}"


# ------------------------------------------------------------------- header

st.title("Semantic cache + cost-aware routing — ops dashboard")

replay_rows, replay_agg = load_replay()
db = load_db_log()
has_replay = not replay_agg.empty
has_db = not db.empty

if not has_replay and not has_db:
    st.warning(
        "No data yet. Run the replay (`python -m eval.replay`) and/or start "
        "the API so `requests_log` fills up, then refresh.")
    st.stop()

# ------------------------------------------------------- replay headline (KPIs)

st.header("Phase-4 replay — pipeline vs always-large", divider="gray")
st.caption("Paraphrase-augmented traffic: every eval question fires once, then "
           "its paraphrase immediately after — the repeat traffic the semantic "
           "cache exists for. All numbers recomputed from eval/results/replay.csv.")

if has_replay:
    def agg(name: str) -> pd.Series | None:
        s = replay_agg[replay_agg["arm"] == name]
        return s.iloc[0] if len(s) else None

    a_pipe, a_nocache, a_base = agg("pipeline"), agg("nocache"), agg("always-large")

    if a_pipe is not None and a_base is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        cost_save = 1 - a_pipe["cost_usd"] / a_base["cost_usd"] if a_base["cost_usd"] else 0.0
        p95_save = 1 - a_pipe["p95_ms"] / a_base["p95_ms"] if a_base["p95_ms"] else 0.0
        dq = a_pipe["quality"] - a_base["quality"]
        c1.metric("Cost saving vs always-large", f"{cost_save * 100:.1f}%",
                  f"{money(a_base['cost_usd'])} → {money(a_pipe['cost_usd'])}",
                  delta_color="off")
        c2.metric("Cache hit rate", f"{a_pipe['cache_hit_rate'] * 100:.0f}%")
        c3.metric("p95 latency change", f"{(p95_save) * 100:+.1f}%",
                  f"{a_base['p95_ms']:.0f} ms → {a_pipe['p95_ms']:.0f} ms",
                  delta_color="inverse")
        c4.metric("Quality delta", f"{dq:+.3f}",
                  "LLM judge, 0–1 scale", delta_color="off")
        c5.metric("Requests", f"{int(a_pipe['n'])}/arm",
                  f"50 Qs + 50 paraphrases")
        st.divider()

        # ---------------------------------------------- arm comparison table
        st.subheader("Arms compared")
        tbl = replay_agg.set_index("arm")[
            ["n", "cache_hit_rate", "quality", "cost_usd", "cost_per_req_usd",
             "p50_ms", "p95_ms", "retrieval_hit",
             "routes_cache", "routes_small", "routes_large", "routes_escalated"]]
        st.dataframe(tbl.style.format({
            "cache_hit_rate": "{:.0%}", "quality": "{:.3f}",
            "cost_usd": money, "cost_per_req_usd": money,
            "p50_ms": "{:.0f}", "p95_ms": "{:.0f}",
        }), use_container_width=True)

        # ------------------------------------------------------ route split
        st.subheader("Route split (pipeline arm)")
        rc = a_pipe[["routes_cache", "routes_small", "routes_large",
                     "routes_escalated"]].astype(int)
        if rc.sum() > 0:
            chart_df = pd.DataFrame({
                "route": ["cache", "small", "large", "escalated"],
                "requests": rc.tolist(),
            })
            st.bar_chart(chart_df.set_index("route"), height=280)

        # --------------------------------------------- cost & quality by arm
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Total cost by arm")
            st.bar_chart(replay_agg.set_index("arm")["cost_usd"], height=280)
        with cc2:
            st.subheader("Quality by arm")
            st.bar_chart(replay_agg.set_index("arm")["quality"], height=280)
    else:
        st.info("replay.csv found but missing expected aggregate rows — rerun "
                "`python -m eval.replay`.")

# ------------------------------------------------- per-request latency + cache

if has_replay and not replay_rows.empty:
    st.header("Per-request view", divider="gray")
    arm = st.selectbox("Arm", sorted(replay_rows["arm"].unique()))
    view = replay_rows[replay_rows["arm"] == arm].copy()
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg latency", f"{view['latency_ms'].mean():.0f} ms")
    k2.metric("Cache hits", f"{int(view['cache_hit'].sum())}/{len(view)}")
    k3.metric("Total cost", money(view["cost_usd"].sum()))
    l1, l2 = st.columns(2)
    with l1:
        st.subheader("Latency by request")
        st.line_chart(view.set_index("qid")[["latency_ms"]], height=250)
    with l2:
        st.subheader("Cost by request")
        st.bar_chart(view.set_index("qid")[["cost_usd"]], height=250)
    with st.expander("Raw requests"):
        st.dataframe(view, use_container_width=True)

# ------------------------------------------------------------------ live logs

if has_db:
    st.header("Live traffic (requests_log)", divider="gray")
    st.caption(f"Last {len(db)} requests from Postgres `requests_log` "
               f"({config.DATABASE_URL.split('@')[-1] if False else 'DATABASE_URL'}).")
    db["ts"] = pd.to_datetime(db["ts"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cache hit rate", f"{db['cache_hit'].mean() * 100:.0f}%")
    k2.metric("Avg cost / request", money(db["cost_usd"].mean()))
    k3.metric("Total spend", money(db["cost_usd"].sum()))
    k4.metric("p95 latency", f"{db['latency_ms'].quantile(0.95):.0f} ms")

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Cost over time")
        ts = db.set_index("ts").sort_index()
        st.line_chart(ts["cost_usd"].rolling(20, min_periods=1).mean(), height=250)
    with t2:
        st.subheader("Route split")
        st.bar_chart(db["route"].value_counts(), height=250)
    with st.expander("Recent requests"):
        st.dataframe(db, use_container_width=True)
elif has_replay:
    st.info("Live traffic section hidden — Postgres (`docker compose up -d`) "
            "is not reachable, so `requests_log` is empty. Start it and hit "
            "`POST /answer` a few times to feed this section.")

# ------------------------------------------------- phase 0-3 eval artifacts

st.header("Eval artifacts", divider="gray")
ev1, ev2, ev3 = st.columns(3)
with ev1:
    st.subheader("Baseline (50 Qs)")
    b = load_result_csv("baseline_llm.csv")
    if not b.empty:
        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieval hit", f"{b['retrieval_hit'].mean():.3f}")
        m2.metric("Quality", f"{b['quality'].mean():.3f}")
        m3.metric("Cost", money(b["cost_usd"].sum()))
    else:
        st.caption("baseline_llm.csv not found")
with ev2:
    st.subheader("Router (100 labeled)")
    r = load_result_csv("router_eval.csv")
    if not r.empty:
        acc = r["correct"].mean()
        split = r["predicted"].value_counts()
        st.metric("Accuracy", f"{acc:.2f}")
        st.bar_chart(split, height=220)
    else:
        st.caption("router_eval.csv not found")
with ev3:
    st.subheader("Cache threshold sweep")
    t = load_result_csv("threshold_sweep.csv")
    if not t.empty:
        best = t[t["false_hits"] == 0].sort_values(
            "paraphrase_recall", ascending=False).head(1)
        if not best.empty:
            st.metric("Recommended threshold", f"{best['threshold'].iloc[0]:.2f}",
                      f"recall {best['paraphrase_recall'].iloc[0]:.2f}, 0 false hits")
        line = t.set_index("threshold")[["paraphrase_recall", "near_miss_fpr"]]
        st.line_chart(line, height=220)
    else:
        st.caption("threshold_sweep.csv not found")

st.divider()
st.caption("Recompute: `python -m eval.replay` (Phase 4) · "
           "`python -m eval.run` (baseline) · `python -m eval.router_eval` · "
           "`python -m eval.threshold_sweep`. Dashboard reads files at render "
           "time; refresh after new runs.")
