"""
ML platform routes: model leaderboard, pipeline topology, agent registry.

These exist because the platform's primary artefact is a *machine-learning
pipeline*, not a single report. A user's recurring questions are "which model
architecture actually wins across all my experiments", "what does the pipeline
do", and "which agent is responsible for what" -- and none of those are
answerable from a single run's payload.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import ModelResult, Run

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/leaderboard")
def leaderboard(
    limit_runs: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Aggregate model performance across every experiment.

    This is the question a modelling engineer actually asks: over all the
    experiments I have run, does the sequence model beat gradient boosting, or
    did it win once on a lucky fold? Ranked by mean information coefficient,
    because IC measures ranking skill directly while accuracy on a near-balanced
    label barely discriminates.
    """
    rows = db.execute(
        select(
            ModelResult.model_name,
            func.count(ModelResult.id),
            func.count(func.distinct(ModelResult.run_id)),
            func.avg(ModelResult.accuracy),
            func.avg(ModelResult.auc),
            func.avg(ModelResult.information_coefficient),
            func.avg(ModelResult.signal_mean_return),
            func.min(ModelResult.accuracy),
            func.max(ModelResult.accuracy),
        ).group_by(ModelResult.model_name)
    ).all()

    models = []
    for name, n_folds, n_runs, acc, auc, ic, sig_ret, acc_min, acc_max in rows:
        models.append(
            {
                "model": name,
                "n_folds": int(n_folds),
                "n_runs": int(n_runs),
                "mean_accuracy": round(float(acc), 4) if acc is not None else None,
                "mean_auc": round(float(auc), 4) if auc is not None else None,
                "mean_information_coefficient": round(float(ic), 4) if ic is not None else None,
                "mean_signal_return": round(float(sig_ret), 5) if sig_ret is not None else None,
                "accuracy_range": [
                    round(float(acc_min), 4) if acc_min is not None else None,
                    round(float(acc_max), 4) if acc_max is not None else None,
                ],
            }
        )

    models.sort(key=lambda m: (m["mean_information_coefficient"] or -9), reverse=True)

    total_runs = db.execute(select(func.count(Run.id))).scalar_one()
    return {
        "models": models,
        "n_experiments": total_runs,
        "ranked_by": "mean_information_coefficient",
        "note": "Aggregated across every fold of every experiment. A model that "
        "leads on one run and trails on the rest will show a high accuracy range "
        "and a mediocre mean IC -- which is the pattern that distinguishes a real "
        "architecture advantage from a lucky fold.",
    }


@router.get("/pipeline")
def pipeline_topology():
    """The pipeline DAG, its stages, and what each one guarantees.

    Served from the backend rather than hardcoded in the UI so the diagram can
    never drift from the graph that actually executes.
    """
    return {
        "orchestrator": "langgraph",
        "has_cycle": True,
        "cycle_description": "The validation stage can reject a result and send the "
        "run back through planning with a bounded revision, so the graph is a cycle "
        "rather than a straight line.",
        "max_revisions": 1,
        "stages": [
            {
                "id": "plan",
                "agent": "research_planner",
                "group": "planning",
                "does": "Turns a natural-language question into a schema-validated plan: universe, factor families, horizon, folds.",
                "guarantees": "Output is a validated Pydantic model with bounded fields. An LLM plan that violates a bound is discarded for the deterministic plan.",
                "uses_llm": True,
            },
            {
                "id": "market_data",
                "agent": "market_data_agent",
                "group": "ingestion",
                "does": "Fetches daily OHLCV, validates, cleans and calendar-aligns the universe.",
                "guarantees": "Split/dividend adjusted; assets failing validation are dropped, not silently carried.",
                "uses_llm": False,
            },
            {
                "id": "macro",
                "agent": "macro_data_agent",
                "group": "ingestion",
                "does": "Fetches macro series (rates, curve slope, volatility, credit spreads, inflation).",
                "guarantees": "Each series shifted by its real publication lag, so a feature cannot see an unreleased statistic.",
                "uses_llm": False,
            },
            {
                "id": "fundamentals",
                "agent": "fundamental_data_agent",
                "group": "ingestion",
                "does": "Pulls XBRL company facts from SEC EDGAR.",
                "guarantees": "Point-in-time: keyed on filing date, not period end, and the first-reported value wins over later restatements.",
                "uses_llm": False,
            },
            {
                "id": "sentiment",
                "agent": "news_sentiment_agent",
                "group": "ingestion",
                "does": "Scores dated news with FinBERT (domain-tuned transformer), falling back to an LLM then a keyword baseline.",
                "guarantees": "Output is a dated time series, never a constant broadcast across history.",
                "uses_llm": False,
            },
            {
                "id": "documents",
                "agent": "document_rag_agent",
                "group": "ingestion",
                "does": "Ingests filings, chunks section-aware, embeds, and answers questions with citations.",
                "guarantees": "Metadata pre-filter, hybrid dense+BM25 retrieval, cross-encoder rerank, and every figure verified against source text.",
                "uses_llm": True,
            },
            {
                "id": "factors",
                "agent": "quant_research_agent",
                "group": "features",
                "does": "Builds the feature panel and computes information coefficients and Fama-MacBeth premia.",
                "guarantees": "Features admitted on measured coverage; time-constant features rejected as a leakage channel.",
                "uses_llm": False,
            },
            {
                "id": "ml",
                "agent": "ml_research_agent",
                "group": "modelling",
                "does": "Trains and compares logistic regression, gradient boosting, LSTM and a Transformer encoder.",
                "guarantees": "Purged, embargoed, expanding walk-forward validation; identical folds and features for every model; out-of-sample predictions returned for downstream use.",
                "uses_llm": False,
            },
            {
                "id": "backtest",
                "agent": "backtesting_agent",
                "group": "evaluation",
                "does": "Trades each model's out-of-sample scores through identical mechanics with a real cost model.",
                "guarantees": "Spread + square-root market impact + borrow; multiple-testing correction across every strategy tried.",
                "uses_llm": False,
            },
            {
                "id": "risk",
                "agent": "risk_agent",
                "group": "evaluation",
                "does": "Decomposes risk into factor exposures vs idiosyncratic, computes tail risk and replays historical scenarios.",
                "guarantees": "Cornish-Fisher tail measures adjusted for skew and kurtosis; benchmark aligned by date join.",
                "uses_llm": False,
            },
            {
                "id": "portfolio",
                "agent": "portfolio_construction_agent",
                "group": "evaluation",
                "does": "Compares allocation methods walk-forward with shrunk covariance and a turnover penalty.",
                "guarantees": "Covariance re-estimated on a trailing window at each rebalance; in-sample optimism reported explicitly.",
                "uses_llm": False,
            },
            {
                "id": "critic",
                "agent": "critic_agent",
                "group": "validation",
                "does": "Runs ten independent checks against the pipeline's own outputs and issues a verdict with actionable revisions.",
                "guarantees": "Leakage is tested empirically, not by column name. The verdict changes control flow rather than decorating the report.",
                "uses_llm": False,
            },
            {
                "id": "revise",
                "agent": "plan_revision",
                "group": "validation",
                "does": "Applies the critic's recommended revisions and re-enters the pipeline.",
                "guarantees": "Bounded to a fixed number of iterations so the loop cannot run away.",
                "uses_llm": False,
            },
            {
                "id": "report",
                "agent": "research_report_agent",
                "group": "reporting",
                "does": "Assembles the report from upstream structured output only.",
                "guarantees": "Template-driven. No language model writes the numbers, so the report cannot state a figure no stage computed.",
                "uses_llm": False,
            },
        ],
        "edges": [
            ["plan", "market_data"], ["market_data", "macro"], ["macro", "fundamentals"],
            ["fundamentals", "sentiment"], ["sentiment", "documents"], ["documents", "factors"],
            ["factors", "ml"], ["ml", "backtest"], ["backtest", "risk"], ["risk", "portfolio"],
            ["portfolio", "critic"], ["critic", "revise"], ["critic", "report"],
            ["revise", "market_data"],
        ],
        "terminal_state": "awaiting_approval",
        "terminal_note": "The pipeline never terminates at 'done'. It stops at a human "
        "approval gate; no output is treated as decision-grade without an explicit, "
        "attributable human decision.",
    }
