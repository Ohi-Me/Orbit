"""Operational routes: approval queue, agent health, drift, cost."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import Approval, Run, RunStep, User
from app.core.security import get_current_user

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/approvals")
def approval_queue(
    status: str = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    """Runs waiting on a human decision.

    This is the screen that makes the human gate real rather than nominal: if
    nobody can see what is waiting, nothing gets reviewed.
    """
    stmt = (
        select(Approval, Run)
        .join(Run, Approval.run_id == Run.id)
        .where(Approval.status == status)
        .order_by(desc(Approval.created_at))
        .limit(limit)
    )
    rows = db.execute(stmt).all()
    return {
        "status_filter": status,
        "n_pending": db.execute(
            select(func.count(Approval.id)).where(Approval.status == "pending")
        ).scalar_one(),
        "items": [
            {
                "approval_id": a.id,
                "run_id": r.id,
                "question": r.question,
                "critic_verdict": a.critic_verdict,
                "critic_summary": a.critic_summary,
                "sharpe": r.sharpe,
                "cagr": r.cagr,
                "max_drawdown": r.max_drawdown,
                "best_model": r.best_model,
                "created_at": a.created_at,
                "decided_by": a.decided_by,
                "decided_at": a.decided_at,
            }
            for a, r in rows
        ],
    }


@router.get("/agents")
def agent_health(
    days: int = Query(default=7, ge=1, le=90),
    db: Session = Depends(get_db),
):
    """Per-agent reliability and latency across recent runs.

    Answers the operational questions a research platform actually gets asked:
    which agent is slow, which is failing, and which is quietly degrading.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        select(
            RunStep.agent,
            func.count(RunStep.id),
            func.avg(RunStep.seconds),
            func.max(RunStep.seconds),
            func.sum(RunStep.llm_input_tokens),
            func.sum(RunStep.llm_output_tokens),
            func.sum(RunStep.llm_calls),
        )
        .where(RunStep.created_at >= since)
        .group_by(RunStep.agent)
    ).all()

    status_rows = db.execute(
        select(RunStep.agent, RunStep.status, func.count(RunStep.id))
        .where(RunStep.created_at >= since)
        .group_by(RunStep.agent, RunStep.status)
    ).all()
    status_map: dict[str, dict] = {}
    for agent, st, count in status_rows:
        status_map.setdefault(agent, {})[st] = count

    agents = []
    for agent, n, avg_s, max_s, tin, tout, calls in rows:
        statuses = status_map.get(agent, {})
        total = sum(statuses.values()) or 1
        agents.append(
            {
                "agent": agent,
                "n_executions": n,
                "avg_seconds": round(float(avg_s or 0), 3),
                "max_seconds": round(float(max_s or 0), 3),
                "status_counts": statuses,
                "failure_rate": round(statuses.get("failed", 0) / total, 4),
                "degraded_rate": round(statuses.get("degraded", 0) / total, 4),
                "llm_calls": int(calls or 0),
                "llm_input_tokens": int(tin or 0),
                "llm_output_tokens": int(tout or 0),
            }
        )

    agents.sort(key=lambda a: -a["avg_seconds"])
    return {
        "window_days": days,
        "agents": agents,
        "slowest_agent": agents[0]["agent"] if agents else None,
        "most_failure_prone": max(agents, key=lambda a: a["failure_rate"])["agent"] if agents else None,
    }


@router.get("/runs/summary")
def runs_summary(
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Platform-level run statistics."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    runs = db.execute(select(Run).where(Run.created_at >= since)).scalars().all()

    by_status: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    durations, sharpes = [], []
    for r in runs:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        if r.critic_verdict:
            by_verdict[r.critic_verdict] = by_verdict.get(r.critic_verdict, 0) + 1
        if r.total_seconds:
            durations.append(r.total_seconds)
        if r.sharpe is not None:
            sharpes.append(r.sharpe)

    durations.sort()

    def _pctile(vals, q):
        if not vals:
            return None
        idx = min(int(len(vals) * q), len(vals) - 1)
        return round(vals[idx], 2)

    return {
        "window_days": days,
        "n_runs": len(runs),
        "by_status": by_status,
        "by_critic_verdict": by_verdict,
        "duration_seconds": {
            "median": _pctile(durations, 0.5),
            "p90": _pctile(durations, 0.9),
            "max": round(durations[-1], 2) if durations else None,
        },
        "n_with_sharpe": len(sharpes),
        "approval_rate": round(by_status.get("approved", 0) / len(runs), 3) if runs else None,
        "note": "A high rejection rate is a healthy sign that the Critic is doing "
        "something. A 100% approval rate usually means the gate is being rubber-stamped.",
    }


@router.post("/drift")
def compute_drift(
    tickers: list[str],
    n_days: int = Query(default=1008, ge=252, le=3024),
    recent_fraction: float = Query(default=0.25, ge=0.05, le=0.5),
):
    """Factor-distribution drift on a freshly built panel.

    Deliberately recomputes the panel rather than reading a stored one: drift
    monitoring is about TODAY's data against the reference window, so reading
    a snapshot taken at training time would answer the wrong question.
    """
    from app.agents.market_data import load_universe
    from app.agents.quant_research import build_factor_panel, select_usable_factors
    from app.core.observability import detect_feature_drift
    from app.data.providers.edgar import build_pit_fundamentals
    from app.data.providers.macro import fetch_macro

    md = load_universe([t.upper() for t in tickers], n_days=n_days)
    if md["status"] != "ok":
        return {"status": "no_data", "detail": md}

    funds = {}
    for t in md["tickers"]:
        try:
            df, _ = build_pit_fundamentals(t)
            if not df.empty:
                funds[t] = df
        except Exception:
            continue

    try:
        macro_df, _ = fetch_macro()
    except Exception:
        macro_df = None

    panel = build_factor_panel(md["frames"], fundamentals=funds, macro_df=macro_df)
    selected, _ = select_usable_factors(panel)
    drift = detect_feature_drift(panel, selected, recent_fraction=recent_fraction)
    return {
        "tickers": md["tickers"],
        "is_synthetic": md["is_synthetic"],
        "n_factors": len(selected),
        "drift": drift,
    }
