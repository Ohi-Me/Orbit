"""
Backtesting Agent
=================
Trades the out-of-sample predictions of a walk-forward-validated model and
reports institutional-style performance after realistic costs.

THE FIX THIS FILE EXISTS FOR
----------------------------
The original build validated logistic regression and XGBoost, then
backtested a completely separate hand-weighted blend of raw factor z-scores
whose weights were hardcoded constants no model had ever fit. The ML table
and the equity curve described different objects, so the platform could not
answer its own research question -- "does the model add risk-adjusted
value" -- because the thing being tested was not the thing being validated.

`run_backtest` now takes `oos_predictions` from the ML Research Agent and
trades those scores directly. Only test-fold predictions exist in that
frame, so every position is formed from a model that had never seen the data
it is trading on. The old hand-tuned formula survives as
`composite_score` -- but as an explicitly labelled BASELINE strategy to beat,
which is what it should always have been.

WHAT CHANGED IN THE MECHANICS
  * Daily P&L. The original compounded one approximate return per rebalance
    (~36 observations over three years). Weights are now held between
    rebalances and marked to market daily, giving ~750 observations, a real
    drawdown path, and enough data for the statistics to mean anything.
  * A real cost model. A flat 10 bps regardless of trade size was flagged in
    review as understating reality. Costs are now spread + square-root market
    impact scaled by participation in daily volume + borrow on the short leg.
  * Dollar-neutral, gross-normalized weights, so the reported return is not
    silently levered 2x by the long/short construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.stats import (
    deflated_sharpe_ratio,
    newey_west_tstat,
    sharpe_of,
    stationary_bootstrap_ci,
)

TRADING_DAYS = 252
REBALANCE_EVERY = 21  # ~monthly

# --- cost model parameters -------------------------------------------------
# Half-spread actually paid on a marketable order in a liquid large cap.
SPREAD_BPS = 3.0
# Coefficient on the square-root impact law: impact = k * sigma * sqrt(participation).
# k ~ 0.5-1.0 is the range reported in the market-microstructure literature;
# 1.0 is the conservative end and is what we use.
IMPACT_COEFFICIENT = 1.0
# Annualized cost to borrow a general-collateral name for shorting.
BORROW_COST_ANNUAL = 0.005
# Fraction of a day's dollar volume the strategy is assumed to represent.
ASSUMED_AUM = 10_000_000.0


def composite_score(panel: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
    """The original hand-weighted factor blend, kept as an explicit BASELINE.

    These weights were never fitted to anything -- they are a reasonable prior
    a human might write down. That makes them a fair benchmark for whether the
    models add value, and a dishonest thing to present as a model's output.
    """
    weights = {
        "momentum_12_1": 0.30,
        "momentum_3m": 0.15,
        "earnings_yield": 0.20,
        "book_to_price": 0.10,
        "return_on_equity": 0.15,
        "operating_margin": 0.10,
        "mean_reversion_z": -0.10,
        "volatility_20d": -0.10,
        "leverage_ratio": -0.05,
    }
    usable = {k: v for k, v in weights.items() if k in factor_cols and k in panel.columns}
    if not usable:
        return pd.Series(np.nan, index=panel.index)

    grouped = panel.groupby("date")[list(usable)]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    z = ((panel[list(usable)] - mean) / std).clip(-3, 3)
    return sum(z[c] * w for c, w in usable.items())


# ---------------------------------------------------------------------------
# Position construction
# ---------------------------------------------------------------------------
def _form_weights(scores: pd.Series, top_frac: float = 0.3, max_weight: float = 0.25) -> dict:
    """Dollar-neutral long/short weights from a cross-sectional score.

    Gross exposure is normalized to 1.0 so reported returns are not implicitly
    levered. With a small universe the tercile can collapse to one name per
    side; that is reported rather than hidden, because a 1-long/1-short book
    is a bet on two companies, not a factor strategy.
    """
    scores = scores.dropna()
    n = len(scores)
    if n < 2:
        return {}

    k = max(1, int(round(n * top_frac)))
    ranked = scores.sort_values(ascending=False)
    longs = ranked.index[:k].tolist()
    shorts = ranked.index[-k:].tolist()
    overlap = set(longs) & set(shorts)
    if overlap:  # universe too small to separate the legs
        return {}

    w = {}
    for t in longs:
        w[t] = 0.5 / k
    for t in shorts:
        w[t] = -0.5 / k

    # Cap and renormalize gross to 1.0.
    w = {t: float(np.clip(v, -max_weight, max_weight)) for t, v in w.items()}
    gross = sum(abs(v) for v in w.values())
    if gross <= 0:
        return {}
    return {t: v / gross for t, v in w.items()}


def _transaction_cost(
    prev_w: dict, new_w: dict, volatility: dict, dollar_volume: dict, aum: float = ASSUMED_AUM
) -> tuple[float, dict]:
    """Cost of moving from prev_w to new_w, as a fraction of portfolio value.

    Three components, each a real effect rather than a single flat number:
      spread  -- half-spread paid per unit traded
      impact  -- k * sigma * sqrt(participation), the square-root law; this is
                 what makes cost scale with trade SIZE and name liquidity
                 instead of being a constant
      (borrow is charged separately, daily, on the short leg)
    """
    tickers = set(prev_w) | set(new_w)
    spread_cost = 0.0
    impact_cost = 0.0
    total_traded = 0.0

    for t in tickers:
        dw = abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0))
        if dw <= 0:
            continue
        total_traded += dw
        spread_cost += dw * (SPREAD_BPS / 10_000.0)

        adv = dollar_volume.get(t, 0.0)
        sigma = volatility.get(t, 0.0)
        if adv and adv > 0 and sigma and np.isfinite(sigma):
            participation = (dw * aum) / adv
            impact_cost += dw * IMPACT_COEFFICIENT * sigma * np.sqrt(min(participation, 1.0))

    return spread_cost + impact_cost, {
        "turnover": round(total_traded, 5),
        "spread_cost": round(spread_cost, 6),
        "impact_cost": round(impact_cost, 6),
    }


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run_backtest(
    panel: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    signal: pd.DataFrame | None = None,
    signal_name: str = "model",
    top_frac: float = 0.3,
    aum: float = ASSUMED_AUM,
) -> dict:
    """Backtest a cross-sectional long/short strategy on a signal.

    Parameters
    ----------
    panel : long factor panel (used for volatility and liquidity in the cost model)
    returns_matrix : wide daily simple returns, index=date, columns=ticker
    signal : long frame with columns [date, ticker, score]. When None, no
             strategy is run -- this agent never invents a signal of its own.
    """
    if signal is None or signal.empty:
        return {"status": "no_signal", "note": "No signal supplied; nothing to backtest."}
    if returns_matrix is None or returns_matrix.empty:
        return {"status": "insufficient_data", "note": "No returns matrix supplied."}

    sig = signal.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    wide_signal = sig.pivot_table(index="date", columns="ticker", values="score", aggfunc="last")
    if wide_signal.empty:
        return {"status": "insufficient_data", "note": "Signal contains no usable rows."}

    rets = returns_matrix.copy()
    rets.index = pd.to_datetime(rets.index)

    # Trade only on dates the signal actually covers (i.e. out-of-sample dates).
    trading_dates = rets.index[(rets.index >= wide_signal.index.min()) & (rets.index <= wide_signal.index.max())]
    if len(trading_dates) < 40:
        return {
            "status": "insufficient_data",
            "note": f"Only {len(trading_dates)} out-of-sample trading days available; "
            "need at least 40 for a meaningful backtest.",
        }

    # Per-ticker volatility and dollar volume for the cost model, keyed by date.
    vol_lookup = _build_lookup(panel, "volatility_20d")
    dv_lookup = _build_lookup(panel, "liquidity_log_dollar_vol", transform=np.exp)

    prev_w: dict = {}
    daily_returns: list[float] = []
    daily_dates: list = []
    weights_history: dict = {}
    cost_history: list[dict] = []
    gross_history: list[float] = []
    rebalance_count = 0

    for i, date in enumerate(trading_dates):
        is_rebalance = (i % REBALANCE_EVERY == 0)

        if is_rebalance:
            available = wide_signal.index[wide_signal.index <= date]
            if len(available):
                scores = wide_signal.loc[available[-1]].dropna()
                new_w = _form_weights(scores, top_frac=top_frac)
                if new_w:
                    cost, breakdown = _transaction_cost(
                        prev_w, new_w, vol_lookup.get(date, {}), dv_lookup.get(date, {}), aum=aum
                    )
                    breakdown["date"] = str(date.date())
                    cost_history.append(breakdown)
                    prev_w = new_w
                    weights_history[str(date.date())] = {k: round(v, 5) for k, v in new_w.items()}
                    rebalance_count += 1
                else:
                    cost = 0.0
            else:
                cost = 0.0
        else:
            cost = 0.0

        if not prev_w:
            continue

        # Mark to market on today's realized returns.
        day_ret = 0.0
        for t, w in prev_w.items():
            if t in rets.columns:
                r = rets.at[date, t] if date in rets.index else np.nan
                if np.isfinite(r):
                    day_ret += w * float(r)

        # Borrow cost accrues daily on the short leg only.
        short_notional = sum(-w for w in prev_w.values() if w < 0)
        borrow = short_notional * (BORROW_COST_ANNUAL / TRADING_DAYS)

        daily_returns.append(day_ret - cost - borrow)
        daily_dates.append(date)
        gross_history.append(sum(abs(w) for w in prev_w.values()))

        # Let weights drift with returns between rebalances, as a real book does.
        prev_w = _drift_weights(prev_w, rets, date)

    if len(daily_returns) < 40:
        return {"status": "insufficient_data", "note": "Too few traded days after signal alignment."}

    r = np.array(daily_returns)
    equity = np.concatenate([[1.0], np.cumprod(1 + r)])
    equity_dates = [daily_dates[0]] + daily_dates

    n_years = len(r) / TRADING_DAYS
    cagr = float(equity[-1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    vol_ann = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = sharpe_of(r, TRADING_DAYS)

    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (
        float(r.mean() * TRADING_DAYS / (downside_std * np.sqrt(TRADING_DAYS)))
        if downside_std and downside_std > 0
        else None
    )

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else None

    total_turnover = sum(c["turnover"] for c in cost_history)
    total_spread = sum(c["spread_cost"] for c in cost_history)
    total_impact = sum(c["impact_cost"] for c in cost_history)
    avg_turnover = total_turnover / max(rebalance_count, 1)

    # --- statistical honesty ------------------------------------------------
    nw = newey_west_tstat(r, lags=REBALANCE_EVERY)
    boot = stationary_bootstrap_ci(
        r, lambda x: sharpe_of(x, TRADING_DAYS), n_boot=600, expected_block=REBALANCE_EVERY
    )
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis())
    dsr = deflated_sharpe_ratio(
        sharpe, n_obs=len(r), n_trials=1, skew=skew, excess_kurtosis=kurt, periods_per_year=TRADING_DAYS
    )

    # Gross-of-cost performance, so the cost drag is a number rather than a claim.
    gross_returns = r + np.array(
        [
            sum(c["spread_cost"] + c["impact_cost"] for c in cost_history) / max(len(r), 1)
        ]
        * len(r)
    )
    gross_sharpe = sharpe_of(gross_returns, TRADING_DAYS)

    return {
        "status": "ok",
        "signal_name": signal_name,
        "n_trading_days": int(len(r)),
        "n_rebalances": rebalance_count,
        "period": [str(daily_dates[0].date()), str(daily_dates[-1].date())],
        "daily_returns": [round(float(x), 6) for x in r],
        "equity_curve": [
            {"date": str(pd.Timestamp(d).date()), "equity": round(float(e), 5)}
            for d, e in zip(equity_dates, equity)
        ],
        "drawdown_curve": [
            {"date": str(pd.Timestamp(d).date()), "drawdown": round(float(x), 5)}
            for d, x in zip(equity_dates, drawdown)
        ],
        "metrics": {
            "cagr": round(cagr, 4),
            "volatility_annualized": round(vol_ann, 4),
            "sharpe_ratio": round(float(sharpe), 3) if np.isfinite(sharpe) else None,
            "sharpe_gross_of_costs": round(float(gross_sharpe), 3) if np.isfinite(gross_sharpe) else None,
            "sortino_ratio": round(sortino, 3) if sortino is not None else None,
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 3) if calmar is not None else None,
            "hit_rate": round(float((r > 0).mean()), 3),
            "total_return": round(float(equity[-1] - 1), 4),
            "avg_gross_exposure": round(float(np.mean(gross_history)), 3) if gross_history else None,
            "skew": round(skew, 3),
            "excess_kurtosis": round(kurt, 3),
        },
        "costs": {
            "avg_turnover_per_rebalance": round(avg_turnover, 4),
            "total_turnover": round(total_turnover, 4),
            "total_spread_cost": round(total_spread, 5),
            "total_impact_cost": round(total_impact, 5),
            "total_cost_drag_on_return": round(total_spread + total_impact, 5),
            "borrow_cost_annual_rate": BORROW_COST_ANNUAL,
            "assumed_aum": aum,
            "model": "spread + square-root market impact "
            f"(k={IMPACT_COEFFICIENT} * sigma * sqrt(participation)) + borrow on the "
            "short leg. Impact scales with trade size relative to each name's dollar "
            "volume, so the cost of the same strategy rises with AUM.",
        },
        "significance": {
            "mean_daily_return": nw["mean"],
            "t_stat_hac": nw["t_stat"],
            "p_value_hac": nw["p_value"],
            "t_stat_naive": nw["naive_t_stat"],
            "t_inflation_factor": nw["inflation_factor"],
            "sharpe_bootstrap_ci": boot,
            "deflated_sharpe": dsr,
        },
        "weights_history": weights_history,
        "cost_history": cost_history[:50],
    }


def _drift_weights(w: dict, rets: pd.DataFrame, date) -> dict:
    """Let positions drift with realized returns between rebalances."""
    if date not in rets.index:
        return w
    new = {}
    for t, wt in w.items():
        r = rets.at[date, t] if t in rets.columns else 0.0
        new[t] = wt * (1 + (float(r) if np.isfinite(r) else 0.0))
    gross = sum(abs(v) for v in new.values())
    return {t: v / gross for t, v in new.items()} if gross > 0 else w


def _build_lookup(panel: pd.DataFrame, column: str, transform=None) -> dict:
    """date -> {ticker: value} lookup used by the cost model."""
    if column not in panel.columns:
        return {}
    out: dict = {}
    sub = panel[["date", "ticker", column]].dropna()
    for date, grp in sub.groupby("date"):
        vals = grp.set_index("ticker")[column]
        if transform is not None:
            vals = transform(vals)
        out[pd.Timestamp(date)] = vals.to_dict()
    return out


def build_returns_matrix(clean_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide daily simple returns, index=date, columns=ticker."""
    series = []
    for t, f in clean_frames.items():
        s = f.set_index("date")["close"].astype(float).pct_change().rename(t)
        series.append(s)
    return pd.concat(series, axis=1).sort_index()


def run_strategy_comparison(
    panel: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    factor_cols: list[str],
    top_frac: float = 0.3,
) -> dict:
    """Backtest every validated model plus the hand-weighted baseline.

    Running all of them through identical mechanics is the only way the
    comparison means anything -- and it is also what makes the multiple-testing
    correction necessary, which is why the deflated Sharpe below is computed
    against the number of strategies actually tried.
    """
    results: dict[str, dict] = {}

    if oos_predictions is not None and not oos_predictions.empty:
        for model_name, grp in oos_predictions.groupby("model"):
            results[model_name] = run_backtest(
                panel,
                returns_matrix,
                signal=grp[["date", "ticker", "score"]],
                signal_name=model_name,
                top_frac=top_frac,
            )

    # Baseline: the hand-weighted composite, scored on the same dates the
    # models were tested on so the comparison is like for like.
    baseline_panel = panel.copy()
    baseline_panel["score"] = composite_score(baseline_panel, factor_cols)
    if oos_predictions is not None and not oos_predictions.empty:
        oos_dates = set(pd.to_datetime(oos_predictions["date"]).unique())
        baseline_panel = baseline_panel[baseline_panel["date"].isin(oos_dates)]
    baseline_signal = baseline_panel[["date", "ticker", "score"]].dropna()
    if not baseline_signal.empty:
        results["composite_baseline"] = run_backtest(
            panel, returns_matrix, signal=baseline_signal, signal_name="composite_baseline", top_frac=top_frac
        )

    # Multiple-testing correction across every strategy actually run.
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    n_trials = max(len(ok), 1)
    for name, res in ok.items():
        m = res["metrics"]
        if m.get("sharpe_ratio") is not None:
            res["significance"]["deflated_sharpe"] = deflated_sharpe_ratio(
                m["sharpe_ratio"],
                n_obs=res["n_trading_days"],
                n_trials=n_trials,
                skew=m.get("skew", 0.0),
                excess_kurtosis=m.get("excess_kurtosis", 0.0),
                periods_per_year=TRADING_DAYS,
            )

    ranked = sorted(
        ok.items(),
        key=lambda kv: kv[1]["metrics"].get("sharpe_ratio") or -99,
        reverse=True,
    )
    best = ranked[0][0] if ranked else None
    baseline_sharpe = (
        ok.get("composite_baseline", {}).get("metrics", {}).get("sharpe_ratio")
        if "composite_baseline" in ok
        else None
    )

    return {
        "status": "ok" if ok else "no_strategies",
        "strategies": results,
        "n_strategies_tested": n_trials,
        "best_strategy": best,
        "best_sharpe": ranked[0][1]["metrics"].get("sharpe_ratio") if ranked else None,
        "baseline_sharpe": baseline_sharpe,
        "model_beats_baseline": (
            bool(ranked and baseline_sharpe is not None and best != "composite_baseline"
                 and (ranked[0][1]["metrics"].get("sharpe_ratio") or -99) > baseline_sharpe)
            if baseline_sharpe is not None
            else None
        ),
        "multiple_testing_note": f"{n_trials} strategies were backtested on the same data. "
        "Each strategy's deflated Sharpe is computed against that trial count, so the "
        "best result is judged against what the best of "
        f"{n_trials} worthless strategies would be expected to produce.",
    }
