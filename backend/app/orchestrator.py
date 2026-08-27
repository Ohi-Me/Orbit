"""
Orchestrator
==============
Runs the fixed agent pipeline described in the project design doc:

    Research Planner
      -> Market Data Agent
      -> Fundamental Analysis Agent (RAG) + News & Sentiment Agent
      -> Quant Research Agent (factor panel)
      -> ML Research Agent (walk-forward)
      -> Backtesting Agent
      -> Risk Agent
      -> Critic / Validation Agent
      -> Portfolio Construction Agent
      -> Research Report Agent

FRAMEWORK CHOICE: implemented as a plain, explicit Python function pipeline
rather than LangGraph/CrewAI/AutoGen. Rationale (see reference.md for the
full write-up): the agent dependency graph here is a straight-line DAG with
no branching, retries-with-different-tools, or dynamic re-planning --
exactly the case where a graph orchestration framework adds dependency
weight and indirection without changing behavior. Each function below IS
an "agent" in the sense the project brief means (a bounded unit with a
single responsibility and a typed input/output contract); swapping this
for LangGraph means wrapping each function as a node and this file becomes
the graph edges -- a mechanical, low-risk migration if/when branching
logic (e.g. "if Critic rejects, request a different factor set and
re-run") is added.
"""

from __future__ import annotations

import time

import numpy as np

from app.agents import backtest as backtest_agent
from app.agents import critic as critic_agent
from app.agents import market_data as market_data_agent
from app.agents import ml_research as ml_agent
from app.agents import portfolio as portfolio_agent
from app.agents import quant_research as quant_agent
from app.agents import report as report_agent
from app.agents import risk as risk_agent
from app.agents import sentiment as sentiment_agent


def run_research_pipeline(
    tickers: list[str],
    n_days: int = 756,
    seed: int = 42,
    n_folds: int = 4,
    max_weight: float = 0.4,
    fundamental_query: str | None = None,
) -> dict:
    t0 = time.time()
    log = []

    def _step(name, fn, *args, **kwargs):
        s = time.time()
        result = fn(*args, **kwargs)
        log.append({"agent": name, "seconds": round(time.time() - s, 3)})
        return result

    clean_frames, bench_df, quality_reports = _step(
        "market_data_agent", market_data_agent.load_universe, tickers, n_days, seed
    )

    sentiment = _step("news_sentiment_agent", sentiment_agent.run_sentiment_agent, tickers)

    fundamental = None
    if fundamental_query:
        from app.agents import fundamental_rag

        fundamental = _step(
            "fundamental_analysis_agent", fundamental_rag.run_fundamental_agent, fundamental_query
        )

    sentiment_scores = {t: s["score"] for t, s in sentiment.items()}
    # Synthetic earnings surprise, deterministic per ticker (real pipeline would
    # come from the Fundamental Analysis Agent's extracted actual vs. estimate).
    rng = np.random.default_rng(seed + 999)
    earnings_surprise = {t: float(rng.normal(0, 0.02)) for t in tickers}

    panel = _step(
        "quant_research_agent",
        quant_agent.build_factor_panel,
        clean_frames,
        sentiment_scores,
        earnings_surprise,
    )

    ml_result = _step("ml_research_agent", ml_agent.run_ml_research_agent, panel, n_folds)
    backtest_result = _step("backtesting_agent", backtest_agent.run_backtest, panel)

    bench_returns = None
    if backtest_result.get("status") == "ok":
        bench_close = bench_df.set_index("date")["close"]
        bench_returns_series = bench_close.pct_change().dropna()
        # crude alignment: resample benchmark to the same number of periods as the backtest
        n_periods = backtest_result["n_rebalances"]
        step = max(1, len(bench_returns_series) // max(n_periods, 1))
        bench_returns = bench_returns_series.iloc[::step].values[:n_periods]

    risk_result = _step("risk_agent", risk_agent.run_risk_agent, backtest_result, bench_returns)

    critic_result = _step(
        "critic_agent",
        critic_agent.run_critic_agent,
        list(panel.columns),
        ml_result,
        backtest_result,
    )

    # Portfolio agent needs an aligned daily returns matrix across tickers.
    returns_frames = []
    for t in tickers:
        r = clean_frames[t].set_index("date")["close"].pct_change().rename(t)
        returns_frames.append(r)
    import pandas as pd

    returns_matrix_df = pd.concat(returns_frames, axis=1).dropna()
    portfolio_result = _step(
        "portfolio_construction_agent",
        portfolio_agent.run_portfolio_agent,
        tickers,
        returns_matrix_df.values,
        max_weight,
    )

    report_md = _step(
        "research_report_agent",
        report_agent.build_report,
        tickers,
        quality_reports,
        sentiment,
        ml_result,
        backtest_result,
        risk_result,
        portfolio_result,
        critic_result,
    )

    return {
        "tickers": tickers,
        "params": {"n_days": n_days, "seed": seed, "n_folds": n_folds, "max_weight": max_weight},
        "quality_reports": quality_reports,
        "sentiment": sentiment,
        "fundamental": fundamental,
        "ml_result": ml_result,
        "backtest_result": backtest_result,
        "risk_result": risk_result,
        "portfolio_result": portfolio_result,
        "critic_result": critic_result,
        "report_markdown": report_md,
        "price_series": {
            t: {
                "dates": clean_frames[t]["date"].dt.strftime("%Y-%m-%d").tolist(),
                "close": clean_frames[t]["close"].round(3).tolist(),
            }
            for t in tickers
        },
        "pipeline_log": log,
        "total_seconds": round(time.time() - t0, 3),
    }
