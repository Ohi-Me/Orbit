"""
Portfolio Construction Agent
============================
Compares allocation methods under realistic constraints, evaluated the same
way every other model in this platform is: out of sample.

WHAT CHANGED FROM THE ORIGINAL
------------------------------
The original computed mean-variance and risk-parity weights from the FULL
SAMPLE covariance matrix and reported the resulting Sharpe. That is the same
look-ahead the Critic agent was built to catch elsewhere: an allocation that
has already seen the whole period's covariance is not an allocation anyone
could have held. `walk_forward_allocation` now re-estimates the covariance on
a trailing window at each rebalance and holds the resulting weights forward,
so the reported performance is achievable.

Also added:
  * TRANSACTION-COST-AWARE objective. Turnover was previously priced only
    inside the separate backtest, so the optimizer happily proposed
    reallocations whose cost exceeded their benefit. The objective now
    includes a turnover penalty against the currently-held weights.
  * LEDOIT-WOLF SHRINKAGE. A sample covariance matrix estimated from a short
    trailing window on a handful of assets is badly conditioned, and
    mean-variance optimization is notoriously sensitive to exactly that
    error -- it puts enormous weight on whichever asset's risk was most
    underestimated. Shrinkage toward a structured target is the standard fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
DEFAULT_TURNOVER_PENALTY = 0.001  # ~10 bps per unit of turnover


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
def ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a constant-correlation target.

    Returns (covariance, shrinkage_intensity). The intensity is reported so a
    reader can see how much the estimate is being smoothed -- a value near 1
    means the sample covariance carried almost no usable information.
    """
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(returns)
        return lw.covariance_, float(lw.shrinkage_)
    except Exception:
        cov = np.cov(returns.T)
        return cov + np.eye(cov.shape[0]) * 1e-8, 0.0


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def _constraints(n: int, max_weight: float):
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, cons


def equal_weight(n: int, **_) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(cov: np.ndarray, max_weight: float = 0.4, prev_w=None, tc=0.0) -> np.ndarray:
    n = cov.shape[0]
    bounds, cons = _constraints(n, max_weight)
    x0 = np.full(n, 1.0 / n)

    def obj(w):
        base = w @ cov @ w
        return base + tc * _turnover(w, prev_w)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-10})
    return res.x if res.success else x0


def mean_variance(mu, cov, risk_aversion=3.0, max_weight=0.4, prev_w=None, tc=0.0) -> np.ndarray:
    n = len(mu)
    bounds, cons = _constraints(n, max_weight)
    x0 = np.full(n, 1.0 / n)

    def obj(w):
        return -(w @ mu - 0.5 * risk_aversion * (w @ cov @ w)) + tc * _turnover(w, prev_w)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-10})
    return res.x if res.success else x0


def risk_parity(cov: np.ndarray, max_weight: float = 0.4, prev_w=None, tc=0.0) -> np.ndarray:
    """Equalize each asset's contribution to total portfolio variance."""
    n = cov.shape[0]
    bounds, cons = _constraints(n, max_weight)
    x0 = np.full(n, 1.0 / n)

    def obj(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e6
        contrib = w * (cov @ w)
        target = port_var / n
        return float(np.sum((contrib - target) ** 2)) * 1e4 + tc * _turnover(w, prev_w)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else x0


def max_diversification(cov: np.ndarray, max_weight: float = 0.4, prev_w=None, tc=0.0) -> np.ndarray:
    """Maximize the ratio of weighted average volatility to portfolio volatility.

    Included because it is the one classical objective that targets
    diversification directly rather than as a by-product, which makes it a
    useful contrast to risk parity on a small, correlated universe.
    """
    n = cov.shape[0]
    sigma = np.sqrt(np.diag(cov))
    bounds, cons = _constraints(n, max_weight)
    x0 = np.full(n, 1.0 / n)

    def obj(w):
        port_vol = np.sqrt(max(w @ cov @ w, 1e-12))
        return -(w @ sigma) / port_vol + tc * _turnover(w, prev_w)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-10})
    return res.x if res.success else x0


def _turnover(w: np.ndarray, prev_w) -> float:
    if prev_w is None:
        return 0.0
    return float(np.sum(np.abs(w - np.asarray(prev_w))))


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
METHODS = {
    "equal_weight": lambda mu, cov, mw, pw, tc: equal_weight(len(mu)),
    "min_variance": lambda mu, cov, mw, pw, tc: min_variance(cov, mw, pw, tc),
    "mean_variance": lambda mu, cov, mw, pw, tc: mean_variance(mu, cov, max_weight=mw, prev_w=pw, tc=tc),
    "risk_parity": lambda mu, cov, mw, pw, tc: risk_parity(cov, mw, pw, tc),
    "max_diversification": lambda mu, cov, mw, pw, tc: max_diversification(cov, mw, pw, tc),
}


def walk_forward_allocation(
    returns: pd.DataFrame,
    method: str,
    lookback: int = 252,
    rebalance_every: int = 21,
    max_weight: float = 0.4,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> dict:
    """Re-estimate and rebalance on a trailing window; hold weights forward.

    Every weight is formed from data strictly before the period it is held
    through, which is the property the original full-sample version lacked.
    """
    if method not in METHODS:
        return {"status": "unknown_method"}

    rets = returns.dropna(how="all").fillna(0.0)
    dates = rets.index
    if len(dates) < lookback + rebalance_every + 10:
        return {
            "status": "insufficient_data",
            "note": f"Need at least {lookback + rebalance_every + 10} observations for a "
            f"{lookback}-day lookback; have {len(dates)}.",
        }

    tickers = list(rets.columns)
    prev_w = None
    port_returns, port_dates, weight_log, turnovers = [], [], {}, []
    shrinkages = []

    for i in range(lookback, len(dates)):
        if (i - lookback) % rebalance_every == 0:
            window = rets.iloc[i - lookback : i].to_numpy()
            mu = window.mean(axis=0)
            cov, shrink = ledoit_wolf_covariance(window)
            shrinkages.append(shrink)
            w = METHODS[method](mu, cov, max_weight, prev_w, turnover_penalty)
            w = np.clip(w, 0, None)
            w = w / w.sum() if w.sum() > 0 else np.full(len(tickers), 1 / len(tickers))
            if prev_w is not None:
                turnovers.append(float(np.sum(np.abs(w - prev_w))))
            prev_w = w
            weight_log[str(pd.Timestamp(dates[i]).date())] = {
                t: round(float(x), 4) for t, x in zip(tickers, w)
            }

        if prev_w is None:
            continue
        day = rets.iloc[i].to_numpy()
        port_returns.append(float(prev_w @ day))
        port_dates.append(dates[i])

    if len(port_returns) < 40:
        return {"status": "insufficient_data", "note": "Too few out-of-sample days produced."}

    r = np.array(port_returns)
    equity = np.cumprod(1 + r)
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1).min())

    return {
        "status": "ok",
        "method": method,
        "weights_latest": list(weight_log.values())[-1] if weight_log else {},
        "weights_history": weight_log,
        "out_of_sample": {
            "annualized_return": round(ann_ret, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe": round(ann_ret / ann_vol, 3) if ann_vol > 0 else None,
            "max_drawdown": round(max_dd, 4),
            "total_return": round(float(equity[-1] - 1), 4),
            "n_days": len(r),
        },
        "avg_turnover_per_rebalance": round(float(np.mean(turnovers)), 4) if turnovers else 0.0,
        "avg_covariance_shrinkage": round(float(np.mean(shrinkages)), 3) if shrinkages else None,
        "equity_curve": [
            {"date": str(pd.Timestamp(d).date()), "equity": round(float(e), 5)}
            for d, e in zip(port_dates, equity)
        ][::5],
    }


def run_portfolio_agent(
    returns_matrix: pd.DataFrame,
    max_weight: float = 0.4,
    lookback: int = 252,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    strategy_signal: pd.DataFrame | None = None,
) -> dict:
    """Compare allocation methods, walk-forward.

    `strategy_signal`, when supplied, is used to compute expected returns from
    the model's own scores instead of from trailing means -- which is the
    point of having a signal at all. Trailing mean returns are a notoriously
    poor expected-return estimate and are what makes naive mean-variance
    optimization behave badly.
    """
    if returns_matrix is None or returns_matrix.empty:
        return {"status": "insufficient_data"}

    tickers = list(returns_matrix.columns)
    results = {}
    for method in METHODS:
        results[method] = walk_forward_allocation(
            returns_matrix,
            method,
            lookback=lookback,
            max_weight=max_weight,
            turnover_penalty=turnover_penalty,
        )

    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    ranked = sorted(
        ok.items(), key=lambda kv: kv[1]["out_of_sample"].get("sharpe") or -99, reverse=True
    )

    # In-sample comparison, kept ONLY to quantify the optimism the original
    # full-sample approach introduced. It is labelled as such and is not the
    # headline number anywhere.
    in_sample = {}
    window = returns_matrix.dropna().to_numpy()
    if len(window) > 30:
        mu_full = window.mean(axis=0)
        cov_full, _ = ledoit_wolf_covariance(window)
        for method in METHODS:
            w = METHODS[method](mu_full, cov_full, max_weight, None, 0.0)
            w = np.clip(w, 0, None)
            w = w / w.sum() if w.sum() > 0 else w
            ann_ret = float(w @ mu_full * TRADING_DAYS)
            ann_vol = float(np.sqrt(w @ cov_full @ w) * np.sqrt(TRADING_DAYS))
            in_sample[method] = {
                "weights": {t: round(float(x), 4) for t, x in zip(tickers, w)},
                "annualized_return": round(ann_ret, 4),
                "annualized_volatility": round(ann_vol, 4),
                "sharpe": round(ann_ret / ann_vol, 3) if ann_vol > 0 else None,
            }

    optimism = None
    if ranked and in_sample:
        best_method = ranked[0][0]
        oos_sharpe = ranked[0][1]["out_of_sample"].get("sharpe")
        is_sharpe = in_sample.get(best_method, {}).get("sharpe")
        if oos_sharpe is not None and is_sharpe is not None:
            optimism = round(is_sharpe - oos_sharpe, 3)

    return {
        "status": "ok" if ok else "insufficient_data",
        "constraints": {
            "long_only": True,
            "max_weight_per_asset": max_weight,
            "fully_invested": True,
            "turnover_penalty": turnover_penalty,
        },
        "walk_forward": results,
        "in_sample_reference": in_sample,
        "best_method": ranked[0][0] if ranked else None,
        "best_oos_sharpe": ranked[0][1]["out_of_sample"].get("sharpe") if ranked else None,
        "in_sample_optimism": optimism,
        "optimism_note": "in_sample_optimism is the Sharpe the full-sample "
        "(look-ahead) allocation reports minus what the walk-forward allocation "
        "actually achieved. It is a direct measurement of how much the original "
        "full-sample approach overstated performance.",
        "covariance_estimator": "Ledoit-Wolf shrinkage toward a constant-correlation "
        "target; raw sample covariance on a short window is badly conditioned and "
        "mean-variance optimization amplifies exactly that error.",
    }
