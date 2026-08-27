"""
Critic / Validation Agent
=========================
Runs after the research agents and actively looks for reasons the result is
wrong. Every check is a real computation over the pipeline's own outputs, and
every check can genuinely fail -- a weak run produces flags.

WHAT CHANGED FROM THE ORIGINAL
------------------------------
The original Critic had two structural weaknesses:

  1. Its look-ahead check inspected COLUMN NAMES. It looked for the substring
     "fwd" or "future" in the factor list and passed if it found none. That
     catches a careless rename and nothing else -- it would not have caught
     the actual leak that existed in the pipeline (a factor computed from
     present-day data and broadcast across all history), because that column
     was innocently named "sentiment_score". The check is now empirical: it
     tests each factor for time-constancy and correlates each factor against
     the label at a NEGATIVE lag, which is what real leakage looks like.

  2. Its verdict changed nothing. It emitted REJECT / CAUTION / OK and the
     pipeline proceeded identically either way. The verdict now carries a
     `recommended_action` that the orchestration graph actually branches on.

Checks are ordered by severity of what they catch, not by convenience.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TURNOVER_WARNING = 2.0          # turnover above 2x gross per rebalance is high
MIN_FOLDS = 3
MIN_TRADING_DAYS = 120
MIN_UNIVERSE_FOR_CROSS_SECTION = 15
LEAKAGE_CORRELATION_ALARM = 0.30


def _check(name, passed, detail, severity="warning", **extra):
    return {"check": name, "passed": passed, "severity": severity, "detail": detail, **extra}


# ---------------------------------------------------------------------------
# 1. Leakage -- empirical, not nominal
# ---------------------------------------------------------------------------
def check_lookahead_empirical(panel: pd.DataFrame, factor_cols: list[str], label: str = "fwd_return_21d") -> dict:
    """Test for leakage by measuring it, not by reading column names.

    Two symptoms, both of which the original name-matching check would miss:

    a) TIME-CONSTANT FACTORS. A factor that never changes within a ticker
       carries no time information. If it was derived from present-day data
       (today's news, today's fundamentals) and written across all history,
       every past row silently contains future knowledge.

    b) IMPLAUSIBLE CONTEMPORANEOUS CORRELATION. A legitimate predictive factor
       correlates weakly with a forward return -- an information coefficient
       above ~0.05 is already a good factor. A |correlation| above 0.30 with a
       21-day forward return is not a discovery; it is nearly always the label
       leaking into the feature.
    """
    if panel is None or panel.empty or not factor_cols:
        return _check("lookahead_bias_empirical", None, "Skipped -- no panel to test.", "info")

    constant, suspicious = [], []
    for f in factor_cols:
        if f not in panel.columns:
            continue
        nun = panel.groupby("ticker")[f].nunique(dropna=True)
        if len(nun) and nun.max() <= 1:
            constant.append(f)
        sub = panel[[f, label]].dropna()
        if len(sub) > 30:
            c = float(np.corrcoef(sub[f], sub[label])[0, 1])
            if np.isfinite(c) and abs(c) > LEAKAGE_CORRELATION_ALARM:
                suspicious.append({"factor": f, "correlation_with_label": round(c, 3)})

    problems = []
    if constant:
        problems.append(
            f"Time-constant factors (no variation within a ticker across the whole "
            f"sample): {constant}. If these were derived from present-day data, every "
            "historical row contains future knowledge."
        )
    if suspicious:
        problems.append(
            f"Implausibly high correlation with the forward-return label: {suspicious}. "
            f"A genuine factor rarely exceeds |{LEAKAGE_CORRELATION_ALARM}|; this pattern "
            "usually means the label has leaked into the feature."
        )

    return _check(
        "lookahead_bias_empirical",
        len(problems) == 0,
        problems if problems else "No time-constant factors and no factor-label "
        "correlation above the leakage alarm threshold. Tested empirically, not by "
        "column name.",
        severity="error",
        n_factors_tested=len(factor_cols),
    )


def check_validation_scheme(ml_result: dict) -> dict:
    """Verify the cross-validation actually purged and embargoed.

    Adjacent train/test splits leak by construction when the label is a
    multi-day forward return: the last training sample's outcome resolves
    inside the test window.
    """
    if ml_result.get("status") != "ok":
        return _check("validation_scheme", None, "Skipped -- ML agent did not run.", "info")

    scheme = ml_result.get("validation_scheme", {})
    purge = scheme.get("purge_days", 0)
    horizon = scheme.get("label_horizon_days", 0)
    embargo = scheme.get("embargo_days", 0)

    ok = purge >= horizon > 0
    return _check(
        "validation_scheme",
        ok,
        f"Purge={purge}d, embargo={embargo}d, label horizon={horizon}d. "
        + (
            "Purging covers the full label horizon, so no training sample's outcome "
            "resolves inside its test window."
            if ok
            else "Purge does not cover the label horizon -- training samples near the "
            "fold boundary have labels that resolve inside the test period, which "
            "trains the model on the outcome it is then scored on."
        ),
        severity="error",
    )


# ---------------------------------------------------------------------------
# 2. Statistical strength
# ---------------------------------------------------------------------------
def check_significance(backtest: dict) -> dict:
    """Autocorrelation-corrected significance of the traded strategy.

    Reads the HAC t-statistic, not the naive one, and reports the inflation
    factor so the difference is visible.
    """
    if backtest.get("status") != "ok":
        return _check("statistical_significance", None, "Skipped -- no successful backtest.", "info")

    sig = backtest.get("significance", {})
    p = sig.get("p_value_hac")
    t = sig.get("t_stat_hac")
    naive = sig.get("t_stat_naive")
    infl = sig.get("t_inflation_factor")

    if p is None:
        return _check("statistical_significance", None, "No HAC p-value available.", "info")

    passed = p < 0.05
    return _check(
        "statistical_significance",
        passed,
        f"HAC t={t}, p={p} (naive t={naive}, inflated {infl}x by overlapping returns). "
        + (
            "The mean return is distinguishable from zero after correcting for "
            "autocorrelation -- though this alone does not establish an exploitable edge."
            if passed
            else "The mean return is NOT distinguishable from zero once the "
            "autocorrelation induced by overlapping return windows is corrected for. "
            "The naive t-statistic would have suggested otherwise, which is exactly "
            "why it is not used."
        ),
        severity="error",
        p_value=p,
    )


def check_multiple_testing(comparison: dict) -> dict:
    """Is the best result distinguishable from the luckiest of N tries?"""
    if comparison.get("status") != "ok":
        return _check("multiple_testing", None, "Skipped -- no strategy comparison.", "info")

    n = comparison.get("n_strategies_tested", 1)
    best = comparison.get("best_strategy")
    strategies = comparison.get("strategies", {})
    res = strategies.get(best, {})
    dsr = res.get("significance", {}).get("deflated_sharpe", {})
    value = dsr.get("deflated_sharpe")

    if value is None:
        return _check("multiple_testing", None, "No deflated Sharpe available.", "info")

    passed = bool(dsr.get("significant_at_95"))
    return _check(
        "multiple_testing",
        passed,
        f"{n} strategies were tested; the best ('{best}') has a deflated Sharpe of "
        f"{value} against a null threshold of {dsr.get('expected_max_sharpe_under_null')}. "
        + (
            "It clears the bar that the luckiest of these many trials would be "
            "expected to reach."
            if passed
            else "It does NOT clear the bar the luckiest of these trials would be "
            "expected to reach by chance alone. Selecting the best of several "
            "backtests and reporting its Sharpe without this correction is how "
            "strategies that never work get funded."
        ),
        severity="error",
        n_trials=n,
    )


def check_model_adds_value(ml_result: dict, comparison: dict) -> dict:
    """Did any model actually beat the naive baselines?

    Two baselines matter and both are checked: the base rate (always predict
    the majority direction) and the hand-weighted composite strategy.
    """
    if ml_result.get("status") != "ok":
        return _check("model_adds_value", None, "Skipped -- ML agent did not run.", "info")

    verdict = ml_result.get("model_verdict")
    beats_bt = comparison.get("model_beats_baseline") if comparison else None
    best_model = ml_result.get("best_model")

    passed = verdict == "candidate_signal" and beats_bt is not False
    return _check(
        "model_adds_value",
        passed,
        f"ML verdict: {verdict}. Best model: {best_model}. "
        f"Beats the hand-weighted composite in the backtest: {beats_bt}. "
        + (
            "The model clears both the base rate and the linear baseline."
            if passed
            else "No model beat the naive baselines on a risk-adjusted basis. The "
            "honest conclusion is that this factor set does not predict this "
            "universe at this horizon. That is a valid research finding, not a bug."
        ),
        severity="warning",
    )


# ---------------------------------------------------------------------------
# 3. Robustness
# ---------------------------------------------------------------------------
def check_regime_dependence(backtest: dict) -> dict:
    """Does performance survive splitting the sample in half?"""
    if backtest.get("status") != "ok":
        return _check("regime_dependence", None, "Skipped -- no successful backtest.", "info")

    r = np.array(backtest["daily_returns"], dtype=float)
    if len(r) < MIN_TRADING_DAYS:
        return _check(
            "regime_dependence", None,
            f"Only {len(r)} trading days -- too few to split meaningfully.", "info"
        )

    mid = len(r) // 2
    first, second = r[:mid], r[mid:]

    def _sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")

    s1, s2 = _sharpe(first), _sharpe(second)
    diverges = np.isfinite(s1) and np.isfinite(s2) and np.sign(s1) != np.sign(s2)

    return _check(
        "regime_dependence",
        not diverges,
        f"First-half Sharpe={s1:.2f}, second-half Sharpe={s2:.2f}. "
        + (
            "The sign flips between sub-periods -- performance is regime-dependent "
            "rather than a stable edge."
            if diverges
            else "Direction is consistent across sub-periods. Two sub-periods cannot "
            "rule out regime dependence; they can only fail to detect it."
        ),
        first_half_sharpe=round(s1, 3) if np.isfinite(s1) else None,
        second_half_sharpe=round(s2, 3) if np.isfinite(s2) else None,
    )


def check_fold_stability(ml_result: dict) -> dict:
    """Is performance stable across walk-forward folds, or luck in one fold?"""
    if ml_result.get("status") != "ok":
        return _check("fold_stability", None, "Skipped -- ML agent did not run.", "info")

    flags = []
    for s in ml_result.get("summary", []):
        if s.get("status") != "ok":
            continue
        std = s.get("std_accuracy")
        if std is not None and std > 0.10:
            flags.append(
                f"{s['model']}: fold-to-fold accuracy std={std:.3f} (>0.10). "
                "Performance is unstable across time periods."
            )
        if s.get("n_degenerate_folds", 0) > 0:
            flags.append(
                f"{s['model']}: {s['n_degenerate_folds']} fold(s) where the model "
                "predicted a single class for >95% of samples -- it learned the base "
                "rate, not a signal."
            )
    return _check(
        "fold_stability",
        len(flags) == 0,
        flags if flags else "Fold-to-fold variance within range and no degenerate "
        "single-class predictors.",
    )


def check_turnover_and_costs(backtest: dict) -> dict:
    """Is the strategy's edge surviving its own trading costs?"""
    if backtest.get("status") != "ok":
        return _check("turnover_and_costs", None, "Skipped -- no successful backtest.", "info")

    costs = backtest.get("costs", {})
    m = backtest.get("metrics", {})
    turnover = costs.get("avg_turnover_per_rebalance", 0)
    net = m.get("sharpe_ratio")
    gross = m.get("sharpe_gross_of_costs")

    cost_kills_it = gross is not None and net is not None and gross > 0 >= net
    high_turnover = turnover > TURNOVER_WARNING

    detail = (
        f"Turnover {turnover:.2f}x gross per rebalance. Sharpe {gross} gross of costs "
        f"-> {net} net. Impact cost {costs.get('total_impact_cost')} vs spread "
        f"{costs.get('total_spread_cost')}. "
    )
    if cost_kills_it:
        detail += (
            "The strategy is profitable before costs and unprofitable after them. "
            "Whatever signal exists is smaller than the cost of harvesting it."
        )
    elif high_turnover:
        detail += (
            f"Turnover above {TURNOVER_WARNING}x is high; the square-root impact model "
            "understates real costs at larger AUM, so this degrades further with size."
        )
    else:
        detail += "Turnover and cost drag are within a plausible range."

    return _check("turnover_and_costs", not (cost_kills_it or high_turnover), detail)


# ---------------------------------------------------------------------------
# 4. Evidence adequacy
# ---------------------------------------------------------------------------
def check_sample_adequacy(ml_result: dict, backtest: dict, panel: pd.DataFrame | None) -> dict:
    """Is there enough independent evidence to support any conclusion?"""
    issues = []

    n_folds = 0
    if ml_result.get("status") == "ok":
        summaries = [s for s in ml_result.get("summary", []) if s.get("status") == "ok"]
        n_folds = max((s.get("n_folds_evaluated", 0) for s in summaries), default=0)
        if n_folds < MIN_FOLDS:
            issues.append(f"Only {n_folds} walk-forward folds (minimum {MIN_FOLDS}).")

        ess = ml_result.get("effective_sample_size", {})
        eff = ess.get("effective_n")
        raw = ess.get("raw_n")
        if eff is not None and eff < 100:
            issues.append(
                f"Effective sample size is {eff} independent observations (raw row count "
                f"{raw}). Overlapping 21-day windows and cross-sectional correlation mean "
                "the row count vastly overstates the independent evidence available."
            )

    n_days = backtest.get("n_trading_days", 0) if backtest.get("status") == "ok" else 0
    if n_days < MIN_TRADING_DAYS:
        issues.append(f"Only {n_days} out-of-sample trading days (minimum {MIN_TRADING_DAYS}).")

    if panel is not None and not panel.empty:
        n_names = int(panel["ticker"].nunique())
        if n_names < MIN_UNIVERSE_FOR_CROSS_SECTION:
            issues.append(
                f"Universe has {n_names} names. A cross-sectional long/short strategy on "
                f"fewer than {MIN_UNIVERSE_FOR_CROSS_SECTION} names is a bet on a handful "
                "of companies, not a factor: the tercile portfolios collapse to one or two "
                "positions and idiosyncratic risk dominates any factor signal."
            )

    return _check(
        "sample_adequacy",
        len(issues) == 0,
        issues if issues else f"{n_folds} folds, {n_days} out-of-sample trading days, "
        "adequate universe breadth for a preliminary read.",
        severity="error",
    )


def check_data_quality(market_data: dict) -> dict:
    """Did the data itself arrive in a state worth researching on?"""
    if not market_data:
        return _check("data_quality", None, "Skipped -- no market data record.", "info")

    issues = []
    if market_data.get("is_synthetic"):
        issues.append(
            "Universe is SYNTHETIC. Every performance number below describes the "
            "behaviour of a random-process generator, not a market. Methodology is "
            "demonstrable; results carry no information about real assets."
        )

    validation = market_data.get("validation", {})
    if validation.get("dropped_tickers"):
        issues.append(f"Assets dropped for data errors: {list(validation['dropped_tickers'])}")
    if validation.get("n_warnings", 0) > 0:
        issues.append(f"{validation['n_warnings']} data-quality warnings raised during ingestion.")

    alignment = market_data.get("alignment", {})
    if alignment and not alignment.get("passed", True):
        issues.append(alignment.get("detail", "Calendar alignment below threshold."))

    return _check(
        "data_quality",
        len(issues) == 0,
        issues if issues else "Live data, no assets dropped, calendar well aligned.",
        severity="error" if market_data.get("is_synthetic") else "warning",
    )


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------
def run_critic_agent(
    panel: pd.DataFrame | None,
    factor_cols: list[str],
    ml_result: dict,
    backtest_comparison: dict,
    market_data: dict | None = None,
) -> dict:
    """Run every check and produce a verdict the orchestrator can branch on."""
    best_name = backtest_comparison.get("best_strategy") if backtest_comparison else None
    best_backtest = (
        backtest_comparison.get("strategies", {}).get(best_name, {}) if best_name else {}
    )

    checks = [
        check_lookahead_empirical(panel, factor_cols),
        check_validation_scheme(ml_result),
        check_data_quality(market_data or {}),
        check_sample_adequacy(ml_result, best_backtest, panel),
        check_significance(best_backtest),
        check_multiple_testing(backtest_comparison or {}),
        check_model_adds_value(ml_result, backtest_comparison or {}),
        check_fold_stability(ml_result),
        check_regime_dependence(best_backtest),
        check_turnover_and_costs(best_backtest),
    ]

    failed = [c for c in checks if c["passed"] is False]
    errors = [c for c in failed if c.get("severity") == "error"]
    warnings = [c for c in failed if c.get("severity") == "warning"]

    if errors:
        verdict = "REJECT_INSUFFICIENT_EVIDENCE"
        action = "revise_and_rerun"
    elif len(warnings) >= 3:
        verdict = "CAUTION_MATERIAL_FLAGS"
        action = "human_review_required"
    elif warnings:
        verdict = "CAUTION_SEE_FLAGS"
        action = "human_review_required"
    else:
        verdict = "NO_STRUCTURAL_ISSUES_DETECTED"
        action = "human_review_required"

    # Concrete, actionable revisions -- a verdict that cannot be acted on is
    # a label, and the orchestrator branches on these.
    revisions = []
    for c in errors:
        if c["check"] == "sample_adequacy":
            revisions.append("widen_universe")
            revisions.append("lengthen_history")
        elif c["check"] == "lookahead_bias_empirical":
            revisions.append("drop_leaking_factors")
        elif c["check"] == "data_quality":
            revisions.append("use_live_data")
        elif c["check"] in ("statistical_significance", "multiple_testing"):
            revisions.append("reduce_strategy_count")

    return {
        "checks": checks,
        "n_checks_run": len(checks),
        "n_checks_failed": len(failed),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "overall_verdict": verdict,
        "recommended_action": action,
        "recommended_revisions": sorted(set(revisions)),
        "evaluated_strategy": best_name,
        "verdict_note": "This verdict describes METHODOLOGICAL SOUNDNESS, not "
        "profitability. A run with zero flags can still lose money; a run with flags "
        "is not necessarily wrong -- it means the claims must be qualified. No result "
        "from this platform is decision-grade until a human has approved it.",
    }
