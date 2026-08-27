"""
Run service -- executes research runs off the request thread and persists them.

WHY NOT CELERY/ARQ+REDIS
A full research run takes minutes. Running it inside the request, as the
original build did, means the HTTP call blocks for the whole run, any client
timeout destroys the result (it was only ever in a process dict), and a
server restart wipes every run that ever completed.

The fix does not require a broker. Runs execute on a bounded thread pool and
every state transition is written to Postgres/SQLite, so the queue IS the
`runs` table: status moves queued -> running -> awaiting_approval|failed, and
the client polls. That survives restarts (a run left `running` by a crash is
detectable and re-queueable), needs no fourth service, and is honest about
its one limitation -- it is single-node.

Redis + a real broker becomes worth its weight when runs must survive process
death mid-execution or fan out across machines. That is a documented upgrade
path, not a missing piece: `submit_run` is the only function that would change.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.config import effective_capabilities, get_settings
from app.core.db import session_scope
from app.core.models import Approval, DataSnapshot, ModelResult, Run, RunStep
from app.core.observability import bind_run, get_logger, track_research_run
from app.orchestration.graph import run_research, serializable_state

log = get_logger("run_service")

_settings = get_settings()
_executor = ThreadPoolExecutor(
    max_workers=max(1, _settings.max_concurrent_runs), thread_name_prefix="research"
)
_active: dict[str, threading.Event] = {}


def create_run(question: str, overrides: dict | None, user_id: str | None) -> str:
    """Persist a queued run and return its id."""
    with session_scope() as db:
        run = Run(
            user_id=user_id,
            question=question,
            params=overrides or {},
            status="queued",
            capabilities=effective_capabilities(),
        )
        db.add(run)
        db.flush()
        return run.id


def submit_run(run_id: str, prefer_llm_planner: bool = True) -> None:
    """Schedule execution on the worker pool."""
    _active[run_id] = threading.Event()
    _executor.submit(_execute, run_id, prefer_llm_planner)


def _execute(run_id: str, prefer_llm_planner: bool) -> None:
    with bind_run(run_id):
        log.info("run_started", run_id=run_id)
        with session_scope() as db:
            run = db.get(Run, run_id)
            if run is None:
                return
            run.status = "running"
            run.started_at = datetime.now(timezone.utc)
            question = run.question
            overrides = dict(run.params or {})

        try:
            state = run_research(
                question, run_id, overrides=overrides, prefer_llm_planner=prefer_llm_planner
            )
            _persist_success(run_id, state)
            try:
                track_research_run(run_id, state)
            except Exception as e:  # tracking must never fail a run
                log.warning("tracking_failed", error=str(e))
            log.info("run_finished", run_id=run_id, status=state.get("status"))
        except Exception as e:
            log.error("run_failed", run_id=run_id, error=str(e))
            _persist_failure(run_id, e)
        finally:
            evt = _active.pop(run_id, None)
            if evt:
                evt.set()


def _persist_success(run_id: str, state: dict) -> None:
    payload = serializable_state(state)
    comparison = state.get("backtest_comparison") or {}
    critic = state.get("critic_result") or {}
    ml = state.get("ml_result") or {}

    best_name = comparison.get("best_strategy")
    best = comparison.get("strategies", {}).get(best_name, {}) if best_name else {}
    metrics = best.get("metrics", {}) if best.get("status") == "ok" else {}

    with session_scope() as db:
        run = db.get(Run, run_id)
        if run is None:
            return

        run.status = state.get("status", "awaiting_approval")
        run.plan = state.get("plan")
        run.result = payload
        run.report_markdown = state.get("report_markdown")
        run.data_provenance = state.get("provenance") or {}
        run.finished_at = datetime.now(timezone.utc)
        run.total_seconds = state.get("total_seconds")
        run.sharpe = metrics.get("sharpe_ratio")
        run.cagr = metrics.get("cagr")
        run.max_drawdown = metrics.get("max_drawdown")
        run.critic_verdict = critic.get("overall_verdict")
        run.n_critic_flags = critic.get("n_checks_failed")
        run.best_model = ml.get("best_model")

        for i, step in enumerate(state.get("steps") or []):
            db.add(
                RunStep(
                    run_id=run_id,
                    sequence=i,
                    agent=step.get("agent", "unknown"),
                    status=step.get("status", "ok"),
                    seconds=step.get("seconds", 0.0),
                    summary=step.get("summary"),
                    error=step.get("error"),
                    degraded_reason=step.get("degraded_reason"),
                )
            )

        for model_block in ml.get("per_model_folds", []) or []:
            model_name = model_block.get("model")
            for fold in model_block.get("folds", []):
                if fold.get("accuracy") is None:
                    continue
                train = fold.get("train_period", ["", ""])
                test = fold.get("test_period", ["", ""])
                db.add(
                    ModelResult(
                        run_id=run_id,
                        model_name=model_name,
                        fold=fold.get("fold", 0),
                        train_start=train[0], train_end=train[1],
                        test_start=test[0], test_end=test[1],
                        n_train=fold.get("n_train", 0),
                        n_test=fold.get("n_test", 0),
                        accuracy=fold.get("accuracy"),
                        auc=fold.get("auc"),
                        information_coefficient=fold.get("information_coefficient"),
                        signal_mean_return=fold.get("signal_mean_return"),
                        signal_t_stat=fold.get("signal_t_stat"),
                        signal_p_value=fold.get("signal_p_value"),
                    )
                )

        prov = state.get("provenance") or {}
        md = state.get("market_data") or {}
        if prov.get("prices"):
            p = prov["prices"]
            db.add(
                DataSnapshot(
                    run_id=run_id, dataset="prices", provider=p.get("provider", "unknown"),
                    is_synthetic=bool(p.get("is_synthetic")), universe=md.get("tickers"),
                    start_date=p.get("start_date"), end_date=p.get("end_date"),
                    n_rows=p.get("n_rows", 0), validation=md.get("validation"),
                )
            )
        for key in ("macro", "fundamentals", "news"):
            if prov.get(key):
                p = prov[key]
                # Each provider counts a different unit, so fall back through the
                # ones it actually reports rather than defaulting to 0 and
                # implying nothing was fetched.
                n_rows = (
                    p.get("n_rows")
                    or p.get("n_items")
                    or p.get("n_facts")
                    or p.get("n_tickers")
                    or 0
                )
                db.add(
                    DataSnapshot(
                        run_id=run_id, dataset=key, provider=p.get("provider", "unknown"),
                        is_synthetic=bool(p.get("is_synthetic")), n_rows=int(n_rows),
                        start_date=p.get("start_date"), end_date=p.get("end_date"),
                    )
                )

        # Every completed run opens a pending approval. Nothing is decision-grade
        # until a named human resolves this row.
        db.add(
            Approval(
                run_id=run_id,
                status="pending",
                critic_verdict=critic.get("overall_verdict"),
                critic_summary={
                    "n_errors": critic.get("n_errors"),
                    "n_warnings": critic.get("n_warnings"),
                    "recommended_action": critic.get("recommended_action"),
                    "recommended_revisions": critic.get("recommended_revisions"),
                },
            )
        )


def _persist_failure(run_id: str, exc: Exception) -> None:
    with session_scope() as db:
        run = db.get(Run, run_id)
        if run is None:
            return
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.finished_at = datetime.now(timezone.utc)
        db.add(
            RunStep(
                run_id=run_id, sequence=999, agent="orchestrator", status="failed",
                error=f"{type(exc).__name__}: {exc}",
                summary={"traceback": traceback.format_exc()[-2000:]},
            )
        )


def queue_status() -> dict:
    """What the worker pool is doing right now."""
    with session_scope() as db:
        queued = db.execute(select(Run).where(Run.status == "queued")).scalars().all()
        running = db.execute(select(Run).where(Run.status == "running")).scalars().all()
        return {
            "max_workers": _settings.max_concurrent_runs,
            "n_queued": len(queued),
            "n_running": len(running),
            "active_in_process": list(_active.keys()),
            "running_ids": [r.id for r in running],
            "note": "Runs execute on an in-process thread pool and persist every state "
            "transition, so the runs table is the queue. Single-node by design; a run "
            "left 'running' after a restart is stale and can be re-submitted.",
        }


def reap_stale_runs() -> dict:
    """Mark runs abandoned by a crash. Called on startup.

    Without this, a process killed mid-run leaves rows stuck in 'running'
    forever and the UI shows a spinner that will never resolve.
    """
    with session_scope() as db:
        stale = db.execute(select(Run).where(Run.status.in_(["running", "queued"]))).scalars().all()
        for run in stale:
            if run.id in _active:
                continue
            run.status = "failed"
            run.error = (
                "Run was interrupted by a server restart. Runs execute in-process, so "
                "they do not survive a shutdown. Resubmit to retry."
            )
            run.finished_at = datetime.now(timezone.utc)
        return {"reaped": len(stale), "ids": [r.id for r in stale]}
