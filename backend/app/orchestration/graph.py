"""
Research orchestration graph
============================
A LangGraph state machine over the agent set, with a real revision loop and a
human approval gate.

WHY A GRAPH FRAMEWORK NOW (it was correctly rejected before)
------------------------------------------------------------
The original build used a plain sequential function and documented, rightly,
that a straight-line DAG gains nothing from a graph framework. That reasoning
was sound for what it described. It stopped being true once the Critic became
load-bearing: the Critic can now REJECT a run, the planner can revise the
plan, and the graph re-runs from the data stage with the revision applied.
That is a cycle with a conditional edge and bounded iteration -- exactly the
control flow a hand-rolled pipeline expresses badly.

THE APPROVAL GATE IS NOT DECORATION
The graph terminates at `awaiting_approval`. It does not mark a strategy
usable, and nothing downstream treats a result as decision-grade without a
human Approval row. An LLM chooses what to research; it never decides that a
strategy is worth capital.

FAILURE HANDLING
Every node runs through `_node`, which records timing, catches exceptions, and
classifies the outcome as ok / degraded / failed. A degraded agent (say, EDGAR
unreachable) does not abort the run -- it records why it degraded and the
downstream agents see the reduced inputs. A failure in a critical node
(market data) aborts; a failure in an enrichment node (sentiment) does not.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.agents import backtest as backtest_agent
from app.agents import critic as critic_agent
from app.agents import market_data as market_data_agent
from app.agents import ml_research as ml_agent
from app.agents import planner as planner_agent
from app.agents import portfolio as portfolio_agent
from app.agents import quant_research as quant_agent
from app.agents import report as report_agent
from app.agents import risk as risk_agent
from app.agents import sentiment as sentiment_agent

MAX_REVISIONS = 1  # bounded; an unbounded critic loop is a way to burn money


def _merge_steps(left: list, right: list) -> list:
    return (left or []) + (right or [])


class ResearchState(TypedDict, total=False):
    # inputs
    run_id: str
    question: str
    overrides: dict
    prefer_llm_planner: bool

    # planning
    plan: dict
    resolved_factors: list
    planner_backend: str

    # data
    market_data: dict
    fundamentals: dict
    sentiment: dict
    macro: Any
    documents: dict
    provenance: dict

    # research
    panel: Any
    selected_factors: list
    factor_diagnostics: dict
    factor_ic: dict
    fama_macbeth: dict
    ml_result: dict
    backtest_comparison: dict
    risk_result: dict
    portfolio_result: dict
    critic_result: dict

    # control
    steps: Annotated[list, _merge_steps]
    revision_count: int
    revisions_applied: list
    status: str
    error: str
    report_markdown: str
    llm_usage: dict


# ---------------------------------------------------------------------------
# Node wrapper
# ---------------------------------------------------------------------------
def _node(name: str, fn, state: ResearchState, critical: bool = False) -> dict:
    """Run one agent, recording timing, status and any degradation."""
    start = time.time()
    try:
        result = fn(state)
        seconds = time.time() - start
        step = {
            "agent": name,
            "status": result.pop("_status", "ok"),
            "seconds": round(seconds, 3),
            "summary": result.pop("_summary", None),
            "degraded_reason": result.pop("_degraded_reason", None),
        }
        return {**result, "steps": [step]}
    except Exception as e:
        seconds = time.time() - start
        step = {
            "agent": name,
            "status": "failed",
            "seconds": round(seconds, 3),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
        }
        if critical:
            return {"steps": [step], "status": "failed", "error": f"{name}: {type(e).__name__}: {e}"}
        return {"steps": [step]}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def plan_node(state: ResearchState) -> dict:
    def _run(s):
        if s.get("revision_count", 0) > 0 and s.get("plan"):
            return {"plan": s["plan"], "resolved_factors": s.get("resolved_factors", []),
                    "_summary": {"revised": True, "revision": s["revision_count"]}}
        out = planner_agent.run_planner(
            s["question"], s.get("overrides"), prefer_llm=s.get("prefer_llm_planner", True)
        )
        return {
            "plan": out["plan"],
            "resolved_factors": out["resolved_factors"],
            "planner_backend": out["backend"],
            "llm_usage": out.get("llm_usage") or {},
            "_summary": {
                "backend": out["backend"],
                "universe_size": len(out["plan"]["universe"]),
                "factor_families": out["plan"]["factor_families"],
                "fallback_reason": out.get("fallback_reason"),
            },
            "_status": "degraded" if out.get("fallback_reason") else "ok",
            "_degraded_reason": out.get("fallback_reason"),
        }

    return _node("research_planner", _run, state, critical=True)


def market_data_node(state: ResearchState) -> dict:
    def _run(s):
        plan = s["plan"]
        md = market_data_agent.load_universe(plan["universe"], n_days=plan["n_days"])
        if md["status"] != "ok":
            raise RuntimeError("Market data agent returned no usable assets.")
        return {
            "market_data": md,
            "provenance": {**(s.get("provenance") or {}), "prices": md["provenance"]},
            "_summary": {
                "n_assets": len(md["frames"]),
                "is_synthetic": md["is_synthetic"],
                "provider": md["provenance"]["provider"],
                "warnings": md["validation"]["n_warnings"],
                "dropped": list(md["validation"]["dropped_tickers"]),
            },
            "_status": "degraded" if md["is_synthetic"] else "ok",
            "_degraded_reason": md["provenance"].get("fallback_reason"),
        }

    return _node("market_data_agent", _run, state, critical=True)


def macro_node(state: ResearchState) -> dict:
    def _run(s):
        from app.data.providers.macro import fetch_macro

        df, prov = fetch_macro()
        return {
            "macro": df,
            "provenance": {**(s.get("provenance") or {}), "macro": prov},
            "_summary": {"n_series": len(prov["series"]), "rows": prov["n_rows"]},
        }

    return _node("macro_data_agent", _run, state)


def fundamentals_node(state: ResearchState) -> dict:
    def _run(s):
        from app.data.providers.edgar import build_pit_fundamentals

        out, errors = {}, {}
        for t in s["plan"]["universe"]:
            try:
                df, _prov = build_pit_fundamentals(t)
                if not df.empty:
                    out[t] = df
            except Exception as e:
                errors[t] = f"{type(e).__name__}: {str(e)[:90]}"
        return {
            "fundamentals": out,
            "provenance": {
                **(s.get("provenance") or {}),
                "fundamentals": {
                    "provider": "sec_edgar_xbrl",
                    "point_in_time": True,
                    "n_tickers": len(out),
                    "errors": errors,
                },
            },
            "_summary": {"n_tickers_with_fundamentals": len(out), "n_errors": len(errors)},
            "_status": "degraded" if errors else "ok",
            "_degraded_reason": f"{len(errors)} tickers had no usable XBRL facts" if errors else None,
        }

    return _node("fundamental_data_agent", _run, state)


def sentiment_node(state: ResearchState) -> dict:
    def _run(s):
        res = sentiment_agent.run_sentiment_agent(s["plan"]["universe"])
        return {
            "sentiment": res,
            "provenance": {**(s.get("provenance") or {}), "news": res["news_provenance"]},
            "_summary": {
                "backend": res["backend"],
                "n_tickers": len(res["per_ticker"]),
                "coverage_days": res["news_provenance"].get("coverage_days"),
            },
            "_status": "degraded" if res["news_provenance"].get("is_synthetic") else "ok",
            "_degraded_reason": "News source is a bundled sample, not live coverage."
            if res["news_provenance"].get("is_synthetic")
            else None,
        }

    return _node("news_sentiment_agent", _run, state)


def documents_node(state: ResearchState) -> dict:
    def _run(s):
        plan = s["plan"]
        if not plan.get("ingest_filings"):
            return {"documents": {"status": "skipped", "reason": "Plan did not request filing text."},
                    "_status": "skipped"}

        from app.agents.fundamental_rag import ingest_edgar_filings, run_fundamental_agent
        from app.core.db import SessionLocal

        db = SessionLocal()
        try:
            ingested = [ingest_edgar_filings(db, t, limit=1) for t in plan.get("filing_tickers", [])[:3]]
            answer = None
            if plan.get("document_question"):
                answer = run_fundamental_agent(
                    db, plan["document_question"], tickers=plan.get("filing_tickers")
                )
            return {
                "documents": {"status": "ok", "ingested": ingested, "answer": answer},
                "_summary": {
                    "n_documents": sum(i.get("n_documents", 0) for i in ingested),
                    "verification": (answer or {}).get("verification", {}).get("verdict"),
                },
            }
        finally:
            db.close()

    return _node("document_rag_agent", _run, state)


def factors_node(state: ResearchState) -> dict:
    def _run(s):
        plan = s["plan"]
        md = s["market_data"]
        sent = s.get("sentiment") or {}
        panel = quant_agent.build_factor_panel(
            md["frames"],
            fundamentals=s.get("fundamentals") or {},
            sentiment_series=sent.get("series") or {},
            macro_df=s.get("macro"),
            label_horizon=plan["label_horizon"],
        )
        # Measure coverage on the FULL panel but only against the factors the
        # plan asked for. Slicing the panel first made every unrequested factor
        # report 0% coverage, which reads as a measured finding about a column
        # that was never a candidate.
        requested = s.get("resolved_factors") or quant_agent.ALL_FACTORS
        selected, diagnostics = quant_agent.select_usable_factors(panel, factors=requested)

        cross_sectional = [f for f in selected if f not in quant_agent.MACRO_FACTORS]
        ic = quant_agent.compute_factor_ic(panel, cross_sectional)
        fm = quant_agent.fama_macbeth(panel, cross_sectional)

        return {
            "panel": panel,
            "selected_factors": selected,
            "factor_diagnostics": diagnostics,
            "factor_ic": ic,
            "fama_macbeth": fm,
            "_summary": {
                "panel_rows": int(len(panel)),
                "n_selected": len(selected),
                "n_rejected": len(diagnostics["rejected"]),
                "fama_macbeth_mode": fm.get("mode"),
            },
        }

    return _node("quant_research_agent", _run, state, critical=True)


def ml_node(state: ResearchState) -> dict:
    def _run(s):
        plan = s["plan"]
        res = ml_agent.run_ml_research_agent(
            s["panel"],
            s["selected_factors"],
            n_folds=plan["n_folds"],
            include_deep_learning=plan.get("include_deep_learning", True),
        )
        return {
            "ml_result": res,
            "_summary": {
                "status": res.get("status"),
                "best_model": res.get("best_model"),
                "verdict": res.get("model_verdict"),
                "models": res.get("models_compared"),
            },
        }

    return _node("ml_research_agent", _run, state, critical=True)


def backtest_node(state: ResearchState) -> dict:
    def _run(s):
        ml = s.get("ml_result") or {}
        rm = backtest_agent.build_returns_matrix(s["market_data"]["frames"])
        oos = ml.get("oos_predictions")
        comparison = backtest_agent.run_strategy_comparison(
            s["panel"], rm, oos if oos is not None else pd.DataFrame(), s["selected_factors"]
        )
        return {
            "backtest_comparison": comparison,
            "_summary": {
                "n_strategies": comparison.get("n_strategies_tested"),
                "best": comparison.get("best_strategy"),
                "best_sharpe": comparison.get("best_sharpe"),
                "beats_baseline": comparison.get("model_beats_baseline"),
            },
        }

    return _node("backtesting_agent", _run, state, critical=True)


def risk_node(state: ResearchState) -> dict:
    def _run(s):
        comparison = s.get("backtest_comparison") or {}
        best = comparison.get("best_strategy")
        bt = comparison.get("strategies", {}).get(best, {})
        bench = None
        bdf = s["market_data"].get("benchmark")
        if bdf is not None and not bdf.empty:
            bench = bdf.set_index("date")["close"].astype(float).pct_change().dropna()

        panel = s["panel"]
        dv = {}
        if "liquidity_log_dollar_vol" in panel.columns:
            import numpy as np

            for t, grp in panel.groupby("ticker"):
                vals = grp["liquidity_log_dollar_vol"].dropna()
                if len(vals):
                    dv[t] = float(np.exp(vals.iloc[-1]))

        res = risk_agent.run_risk_agent(
            bt, bench, panel=panel, factor_cols=s["selected_factors"], dollar_volume=dv
        )
        return {
            "risk_result": res,
            "_summary": {
                "var_95": res.get("value_at_risk_daily", {}).get("cornish_fisher_95"),
                "beta": (res.get("benchmark_relative") or {}).get("beta"),
                "factor_r2": (res.get("factor_risk") or {}).get("r_squared"),
            },
        }

    return _node("risk_agent", _run, state)


def portfolio_node(state: ResearchState) -> dict:
    def _run(s):
        rm = backtest_agent.build_returns_matrix(s["market_data"]["frames"])
        res = portfolio_agent.run_portfolio_agent(rm, max_weight=s["plan"]["max_weight"])
        return {
            "portfolio_result": res,
            "_summary": {
                "best_method": res.get("best_method"),
                "oos_sharpe": res.get("best_oos_sharpe"),
                "in_sample_optimism": res.get("in_sample_optimism"),
            },
        }

    return _node("portfolio_construction_agent", _run, state)


def critic_node(state: ResearchState) -> dict:
    def _run(s):
        res = critic_agent.run_critic_agent(
            s.get("panel"),
            s.get("selected_factors", []),
            s.get("ml_result") or {},
            s.get("backtest_comparison") or {},
            s.get("market_data"),
        )
        return {
            "critic_result": res,
            "_summary": {
                "verdict": res["overall_verdict"],
                "errors": res["n_errors"],
                "warnings": res["n_warnings"],
                "revisions": res["recommended_revisions"],
            },
        }

    return _node("critic_agent", _run, state, critical=True)


def revise_node(state: ResearchState) -> dict:
    def _run(s):
        critic = s.get("critic_result") or {}
        revisions = critic.get("recommended_revisions", [])
        new_plan, applied = planner_agent.revise_plan(s["plan"], revisions)
        return {
            "plan": new_plan,
            "resolved_factors": planner_agent.ResearchPlan(**new_plan).resolved_factors(),
            "revision_count": s.get("revision_count", 0) + 1,
            "revisions_applied": (s.get("revisions_applied") or []) + applied,
            "_summary": {"requested": revisions, "applied": applied},
        }

    return _node("plan_revision", _run, state)


def report_node(state: ResearchState) -> dict:
    def _run(s):
        md = report_agent.build_report(s)
        return {
            "report_markdown": md,
            "status": "awaiting_approval",
            "_summary": {"length_chars": len(md)},
        }

    return _node("research_report_agent", _run, state, critical=True)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_critic(state: ResearchState) -> str:
    """The Critic's verdict genuinely changes what happens next.

    A REJECT with actionable revisions and budget remaining sends the run back
    through planning with the revision applied. Otherwise it proceeds to the
    report and stops at the approval gate.
    """
    critic = state.get("critic_result") or {}
    if state.get("status") == "failed":
        return "report"
    if (
        critic.get("recommended_action") == "revise_and_rerun"
        and critic.get("recommended_revisions")
        and state.get("revision_count", 0) < MAX_REVISIONS
    ):
        return "revise"
    return "report"


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("plan", plan_node)
    g.add_node("market_data", market_data_node)
    g.add_node("macro", macro_node)
    g.add_node("fundamentals", fundamentals_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("documents", documents_node)
    g.add_node("factors", factors_node)
    g.add_node("ml", ml_node)
    g.add_node("backtest", backtest_node)
    g.add_node("risk", risk_node)
    g.add_node("portfolio", portfolio_node)
    g.add_node("critic", critic_node)
    g.add_node("revise", revise_node)
    g.add_node("report", report_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "market_data")
    # Data enrichment runs after prices, which every one of them depends on.
    g.add_edge("market_data", "macro")
    g.add_edge("macro", "fundamentals")
    g.add_edge("fundamentals", "sentiment")
    g.add_edge("sentiment", "documents")
    g.add_edge("documents", "factors")
    g.add_edge("factors", "ml")
    g.add_edge("ml", "backtest")
    g.add_edge("backtest", "risk")
    g.add_edge("risk", "portfolio")
    g.add_edge("portfolio", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"revise": "revise", "report": "report"})
    # The revision loop re-enters at market_data: a revised plan can change the
    # universe or history length, so the data must be refetched.
    g.add_edge("revise", "market_data")
    g.add_edge("report", END)

    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_research(
    question: str,
    run_id: str,
    overrides: dict | None = None,
    prefer_llm_planner: bool = True,
    recursion_limit: int = 60,
) -> dict:
    """Execute the graph. Returns the final state (non-serializable keys dropped)."""
    started = datetime.now(timezone.utc)
    t0 = time.time()

    initial: ResearchState = {
        "run_id": run_id,
        "question": question,
        "overrides": overrides or {},
        "prefer_llm_planner": prefer_llm_planner,
        "steps": [],
        "revision_count": 0,
        "revisions_applied": [],
        "status": "running",
    }

    final = get_graph().invoke(initial, {"recursion_limit": recursion_limit})

    elapsed = time.time() - t0
    if final.get("status") not in ("failed", "awaiting_approval"):
        final["status"] = "awaiting_approval"

    final["started_at"] = started.isoformat()
    final["finished_at"] = datetime.now(timezone.utc).isoformat()
    final["total_seconds"] = round(elapsed, 2)
    return final


def serializable_state(state: dict) -> dict:
    """Strip DataFrames and other non-JSON objects before persistence.

    The panel is large and reproducible from the plan; storing the derived
    summaries instead keeps run records queryable rather than enormous.
    """
    drop = {"panel", "macro", "fundamentals", "market_data", "overrides"}
    out = {}
    for k, v in state.items():
        if k in drop:
            continue
        if isinstance(v, pd.DataFrame):
            continue
        if k == "ml_result" and isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk != "oos_predictions"}
        if k == "sentiment" and isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk != "series"}
        out[k] = v

    md = state.get("market_data") or {}
    if md:
        out["data_summary"] = {
            "tickers": md.get("tickers"),
            "is_synthetic": md.get("is_synthetic"),
            "provenance": md.get("provenance"),
            "validation": md.get("validation"),
            "alignment": md.get("alignment"),
            "quality_reports": md.get("quality_reports"),
        }
        frames = md.get("frames") or {}
        out["price_series"] = {
            t: {
                "dates": f["date"].dt.strftime("%Y-%m-%d").tolist()[::5],
                "close": f["close"].round(3).tolist()[::5],
            }
            for t, f in frames.items()
        }
    return out
