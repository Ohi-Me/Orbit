"""Research run routes -- submit, poll, inspect, compare, approve."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import Approval, DataSnapshot, ModelResult, Run, RunStep, User
from app.core.security import get_current_user, owns_or_404, require_user
from app.schemas import ApprovalDecision, ResearchRunRequest
from app.services import run_service

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _summary(run: Run) -> dict:
    prov = (run.data_provenance or {}).get("prices", {})
    return {
        "id": run.id,
        "question": run.question,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "total_seconds": run.total_seconds,
        "sharpe": run.sharpe,
        "cagr": run.cagr,
        "max_drawdown": run.max_drawdown,
        "critic_verdict": run.critic_verdict,
        "n_critic_flags": run.n_critic_flags,
        "best_model": run.best_model,
        "is_synthetic": prov.get("is_synthetic"),
        "universe_size": len((run.plan or {}).get("universe", [])),
        "error": run.error,
    }


@router.post("", status_code=202)
def submit(
    req: ResearchRunRequest,
    user: User | None = Depends(get_current_user),
):
    """Queue a research run. Returns immediately with a run id to poll.

    202 rather than 200 deliberately: the work has been accepted, not done. A
    full run takes minutes, and holding the request open for it was the
    original design's most consequential bug.
    """
    run_id = run_service.create_run(req.question, req.overrides, user.id if user else None)
    run_service.submit_run(run_id, prefer_llm_planner=req.use_llm_planner)
    return {
        "run_id": run_id,
        "status": "queued",
        "poll": f"/api/runs/{run_id}",
        "note": "Poll this run until status is 'awaiting_approval' or 'failed'.",
    }


@router.get("")
def list_runs(
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = None,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    stmt = select(Run).order_by(desc(Run.created_at))
    if user is not None:
        stmt = stmt.where(Run.user_id == user.id)
    else:
        stmt = stmt.where(Run.user_id.is_(None))
    if status:
        stmt = stmt.where(Run.status == status)

    runs = db.execute(stmt.limit(limit).offset(offset)).scalars().all()
    return {"runs": [_summary(r) for r in runs], "limit": limit, "offset": offset}


@router.get("/queue")
def queue():
    return run_service.queue_status()


@router.get("/{run_id}")
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    run = owns_or_404(db.get(Run, run_id), user)
    approval = db.execute(
        select(Approval).where(Approval.run_id == run_id).order_by(desc(Approval.created_at))
    ).scalars().first()

    return {
        **_summary(run),
        "plan": run.plan,
        "capabilities": run.capabilities,
        "data_provenance": run.data_provenance,
        "result": run.result,
        "approval": {
            "id": approval.id,
            "status": approval.status,
            "decided_by": approval.decided_by,
            "decided_at": approval.decided_at,
            "feedback": approval.feedback,
            "critic_summary": approval.critic_summary,
        }
        if approval
        else None,
    }


@router.get("/{run_id}/steps")
def get_steps(
    run_id: str,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    """The agent activity trace -- what ran, how long, what degraded."""
    owns_or_404(db.get(Run, run_id), user)
    steps = db.execute(
        select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.sequence)
    ).scalars().all()
    return {
        "run_id": run_id,
        "steps": [
            {
                "sequence": s.sequence,
                "agent": s.agent,
                "status": s.status,
                "seconds": s.seconds,
                "summary": s.summary,
                "error": s.error,
                "degraded_reason": s.degraded_reason,
                "llm_calls": s.llm_calls,
                "llm_input_tokens": s.llm_input_tokens,
                "llm_output_tokens": s.llm_output_tokens,
            }
            for s in steps
        ],
        "total_seconds": round(sum(s.seconds for s in steps), 2),
        "n_failed": sum(1 for s in steps if s.status == "failed"),
        "n_degraded": sum(1 for s in steps if s.status == "degraded"),
    }


@router.get("/{run_id}/models")
def get_model_results(
    run_id: str,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    """Per-fold model metrics as rows -- the model comparison screen's source."""
    owns_or_404(db.get(Run, run_id), user)
    rows = db.execute(
        select(ModelResult).where(ModelResult.run_id == run_id).order_by(
            ModelResult.model_name, ModelResult.fold
        )
    ).scalars().all()
    return {
        "run_id": run_id,
        "folds": [
            {
                "model": r.model_name,
                "fold": r.fold,
                "train": [r.train_start, r.train_end],
                "test": [r.test_start, r.test_end],
                "n_train": r.n_train,
                "n_test": r.n_test,
                "accuracy": r.accuracy,
                "auc": r.auc,
                "information_coefficient": r.information_coefficient,
                "signal_mean_return": r.signal_mean_return,
                "signal_t_stat": r.signal_t_stat,
                "signal_p_value": r.signal_p_value,
            }
            for r in rows
        ],
    }


@router.get("/{run_id}/lineage")
def get_lineage(
    run_id: str,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    """What data this run consumed -- provider, window, synthetic-or-not."""
    run = owns_or_404(db.get(Run, run_id), user)
    snaps = db.execute(select(DataSnapshot).where(DataSnapshot.run_id == run_id)).scalars().all()
    return {
        "run_id": run_id,
        "capabilities_at_execution": run.capabilities,
        "snapshots": [
            {
                "dataset": s.dataset,
                "provider": s.provider,
                "is_synthetic": s.is_synthetic,
                "universe": s.universe,
                "start_date": s.start_date,
                "end_date": s.end_date,
                "n_rows": s.n_rows,
                "validation": s.validation,
                "fetched_at": s.fetched_at,
            }
            for s in snaps
        ],
    }


@router.get("/{run_id}/report", response_class=PlainTextResponse)
def get_report(
    run_id: str,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    run = owns_or_404(db.get(Run, run_id), user)
    if not run.report_markdown:
        raise HTTPException(status_code=404, detail="No report yet; the run may still be executing.")
    return PlainTextResponse(run.report_markdown, media_type="text/markdown")


@router.post("/{run_id}/approve")
def decide(
    run_id: str,
    decision: ApprovalDecision,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Record a human decision on a run.

    Requires an authenticated user even when AUTH_REQUIRED is off: an approval
    that cannot be attributed to a person is not a control, and the whole
    point of the gate is that a named human accepted the conclusions.
    """
    run = owns_or_404(db.get(Run, run_id), user)
    if run.status not in ("awaiting_approval", "approved", "rejected"):
        raise HTTPException(
            status_code=409,
            detail=f"Run is '{run.status}'. Only a completed run awaiting approval can be decided.",
        )

    approval = db.execute(
        select(Approval).where(Approval.run_id == run_id).order_by(desc(Approval.created_at))
    ).scalars().first()
    if approval is None:
        raise HTTPException(status_code=404, detail="No approval record for this run.")

    approval.status = decision.decision
    approval.decided_by = user.email
    approval.decided_at = datetime.now(timezone.utc)
    approval.feedback = decision.feedback
    run.status = decision.decision
    db.commit()

    return {
        "run_id": run_id,
        "status": run.status,
        "decided_by": approval.decided_by,
        "decided_at": approval.decided_at,
        "feedback": approval.feedback,
    }


@router.get("/{run_id}/drift")
def get_drift(
    run_id: str,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    """Factor-distribution drift between the reference and recent windows."""
    run = owns_or_404(db.get(Run, run_id), user)
    result = run.result or {}
    drift = result.get("drift")
    if drift:
        return drift
    return {
        "status": "not_computed",
        "note": "Drift is computed from the factor panel, which is not persisted with "
        "the run record. Re-run with drift enabled, or compute it against a live panel "
        "via /api/monitoring/drift.",
    }


@router.delete("/{run_id}", status_code=204)
def delete_run(
    run_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    run = owns_or_404(db.get(Run, run_id), user)
    db.delete(run)
    db.commit()
