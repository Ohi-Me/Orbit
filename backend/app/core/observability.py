"""
Observability: structured logging, run metrics, and experiment tracking.

WHAT THIS IS FOR
A research platform that cannot answer "why is this run slow", "which agent
is failing", "what did the LLM cost", and "is today's data drifting from what
the model was fitted on" is not operable. Each of those is a concrete
question, and each has a concrete mechanism below rather than a dashboard
aspiration.

  * STRUCTURED LOGS (structlog) -- JSON in production, human-readable in dev,
    with run_id bound to every line so a run's history is one grep.
  * RUN METRICS -- per-agent timings, failure counts, and token cost are
    persisted on RunStep rows, so the agent-activity screen reads real data.
  * MLFLOW -- experiment tracking around the walk-forward loop. Optional and
    probed: with mlflow absent, the tracker becomes a no-op that records
    nothing and claims nothing.
  * DRIFT -- population stability index between the training window's factor
    distribution and the most recent window's. Cheap, standard, and it
    answers the question that matters: is the model being asked about a world
    it was not fitted on.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

from app.core.config import get_settings

_configured = False


def configure_logging(json_logs: bool | None = None) -> None:
    global _configured
    if _configured:
        return

    import structlog

    if json_logs is None:
        json_logs = os.environ.get("LOG_FORMAT", "console").lower() == "json"

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "quant"):
    configure_logging()
    import structlog

    return structlog.get_logger(name)


@contextmanager
def bind_run(run_id: str, **extra):
    """Bind run_id to every log line emitted inside the block."""
    configure_logging()
    import structlog

    structlog.contextvars.bind_contextvars(run_id=run_id, **extra)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars("run_id", *extra.keys())


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------
class ExperimentTracker:
    """Thin MLflow wrapper that degrades to a no-op when MLflow is absent.

    Deliberately thin: the agents should not import mlflow directly, so that
    removing experiment tracking is a config change rather than a refactor.
    """

    def __init__(self, run_name: str, tags: dict | None = None):
        self.run_name = run_name
        self.tags = tags or {}
        self.enabled = False
        self._run = None
        self._mlflow = None

        settings = get_settings()
        if not settings.enable_mlflow:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            self._mlflow = mlflow
            self.enabled = True
        except Exception:
            self.enabled = False

    def __enter__(self):
        if self.enabled:
            try:
                self._run = self._mlflow.start_run(run_name=self.run_name)
                if self.tags:
                    self._mlflow.set_tags({k: str(v)[:250] for k, v in self.tags.items()})
            except Exception:
                self.enabled = False
        return self

    def __exit__(self, *exc):
        if self.enabled and self._run is not None:
            try:
                self._mlflow.end_run()
            except Exception:
                pass
        return False

    def log_params(self, params: dict):
        if not self.enabled:
            return
        try:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        except Exception:
            pass

    def log_metrics(self, metrics: dict, step: int | None = None):
        if not self.enabled:
            return
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        }
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception:
            pass

    def log_dict(self, obj: dict, filename: str):
        if not self.enabled:
            return
        try:
            self._mlflow.log_dict(obj, filename)
        except Exception:
            pass

    def log_text(self, text: str, filename: str):
        if not self.enabled:
            return
        try:
            self._mlflow.log_text(text, filename)
        except Exception:
            pass


def track_research_run(run_id: str, state: dict) -> dict:
    """Log a completed run's parameters, metrics and artefacts to MLflow."""
    plan = state.get("plan") or {}
    ml = state.get("ml_result") or {}
    comparison = state.get("backtest_comparison") or {}
    critic = state.get("critic_result") or {}

    tags = {
        "run_id": run_id,
        "planner_backend": state.get("planner_backend", "n/a"),
        "critic_verdict": critic.get("overall_verdict", "n/a"),
        "synthetic_data": str((state.get("market_data") or {}).get("is_synthetic", "unknown")),
    }

    with ExperimentTracker(run_name=f"research-{run_id[:8]}", tags=tags) as t:
        if not t.enabled:
            return {"tracked": False, "reason": "MLflow unavailable or disabled."}

        t.log_params(
            {
                "question": state.get("question", "")[:250],
                "universe_size": len(plan.get("universe", [])),
                "n_days": plan.get("n_days"),
                "label_horizon": plan.get("label_horizon"),
                "n_folds": plan.get("n_folds"),
                "factor_families": ",".join(plan.get("factor_families", [])),
                "n_factors_selected": len(state.get("selected_factors", [])),
                "include_deep_learning": plan.get("include_deep_learning"),
                "revision_count": state.get("revision_count", 0),
            }
        )

        for s in ml.get("summary", []):
            if s.get("status") != "ok":
                continue
            model = s["model"]
            t.log_metrics(
                {
                    f"{model}/accuracy": s.get("mean_accuracy"),
                    f"{model}/accuracy_lift": s.get("mean_accuracy_lift"),
                    f"{model}/auc": s.get("mean_auc"),
                    f"{model}/ic": s.get("mean_information_coefficient"),
                    f"{model}/train_seconds": s.get("train_seconds"),
                }
            )

        for name, r in comparison.get("strategies", {}).items():
            if r.get("status") != "ok":
                continue
            m = r["metrics"]
            sig = r.get("significance", {})
            t.log_metrics(
                {
                    f"bt/{name}/sharpe": m.get("sharpe_ratio"),
                    f"bt/{name}/sharpe_gross": m.get("sharpe_gross_of_costs"),
                    f"bt/{name}/cagr": m.get("cagr"),
                    f"bt/{name}/max_drawdown": m.get("max_drawdown"),
                    f"bt/{name}/hac_p_value": sig.get("p_value_hac"),
                    f"bt/{name}/deflated_sharpe": (sig.get("deflated_sharpe") or {}).get("deflated_sharpe"),
                }
            )

        t.log_metrics(
            {
                "critic/n_errors": critic.get("n_errors"),
                "critic/n_warnings": critic.get("n_warnings"),
                "run/total_seconds": state.get("total_seconds"),
            }
        )

        t.log_dict({"plan": plan, "selected_factors": state.get("selected_factors")}, "plan.json")
        t.log_dict(critic, "critic.json")
        if state.get("report_markdown"):
            t.log_text(state["report_markdown"], "report.md")

        return {"tracked": True, "experiment": get_settings().mlflow_experiment}


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float | None:
    """PSI between a reference and a current distribution.

    Convention: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 material shift.
    Bin edges come from the REFERENCE distribution's quantiles, which is what
    makes the comparison meaningful -- rebinning on the current data would hide
    exactly the shift being measured.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 50 or len(actual) < 20:
        return None

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-6, None)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def detect_feature_drift(
    panel: pd.DataFrame, factor_cols: list[str], split_date=None, recent_fraction: float = 0.25
) -> dict:
    """Compare the recent factor distribution against the earlier reference.

    This is the monitoring signal that matters for a factor model: the model is
    fitted on one regime's factor distribution and then asked about another.
    PSI per factor answers whether that has happened, before the P&L does.
    """
    if panel is None or panel.empty or not factor_cols:
        return {"status": "no_data"}

    dates = pd.DatetimeIndex(panel["date"]).sort_values().unique()
    if len(dates) < 120:
        return {"status": "insufficient_history", "n_dates": len(dates)}

    if split_date is None:
        split_date = dates[int(len(dates) * (1 - recent_fraction))]

    reference = panel[panel["date"] < split_date]
    recent = panel[panel["date"] >= split_date]

    results, drifted = {}, []
    for f in factor_cols:
        if f not in panel.columns:
            continue
        psi = population_stability_index(
            reference[f].dropna().to_numpy(), recent[f].dropna().to_numpy()
        )
        if psi is None:
            results[f] = {"psi": None, "status": "insufficient_data"}
            continue
        level = "material" if psi > 0.25 else "moderate" if psi > 0.10 else "stable"
        results[f] = {"psi": round(psi, 4), "level": level}
        if level == "material":
            drifted.append(f)

    return {
        "status": "ok",
        "split_date": str(pd.Timestamp(split_date).date()),
        "n_reference_rows": int(len(reference)),
        "n_recent_rows": int(len(recent)),
        "per_factor": results,
        "drifted_factors": drifted,
        "n_drifted": len(drifted),
        "interpretation": "PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 material. "
        "Bin edges are taken from the reference window's quantiles. A materially "
        "drifted factor means the model is being asked about a distribution it was "
        "not fitted on -- retrain before trusting live predictions.",
    }
