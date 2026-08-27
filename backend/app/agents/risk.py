"""
Risk Agent
==========
Decomposes and stresses the strategy's realized return series.

WHAT CHANGED FROM THE ORIGINAL
------------------------------
The first version computed VaR, CVaR, and a beta against a benchmark that
was aligned by *subsampling* every Nth daily return to match the count of
rebalance periods. That is not an alignment -- it throws away most of the
benchmark's path and compares two series that describe different periods, so
the resulting beta and alpha were not meaningful numbers. Alignment is now a
real inner join on dates.

Three additions the review called for:
  * FACTOR RISK DECOMPOSITION. VaR on a return series treats the strategy as
    a black box. Regressing returns on the factors the strategy actually
    trades says *where the risk comes from* -- how much is a market bet, a
    momentum bet, a volatility bet, and how much is genuinely idiosyncratic.
  * CONCENTRATION AND LIQUIDITY. A max-weight constraint says nothing about
    whether that weight can be traded. Position size is compared against each
    name's dollar volume to produce days-to-liquidate.
  * HISTORICAL SCENARIO REPLAY. Mean/sigma shocks assume a normal world.
    Real crisis windows are replayed against the strategy's estimated beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

# Real crisis windows, replayed against the strategy's estimated market beta.
# Dated windows rather than invented shocks, so the number has a provenance.
HISTORICAL_SCENARIOS = {
    "gfc_2008_q4": {"benchmark_return": -0.2260, "description": "Q4 2008, global financial crisis"},
    "covid_crash_2020": {"benchmark_return": -0.3390, "description": "19 Feb - 23 Mar 2020, COVID crash"},
    "rates_shock_2022": {"benchmark_return": -0.2520, "description": "Jan - Oct 2022, inflation/rates repricing"},
    "vol_spike_2018q4": {"benchmark_return": -0.1400, "description": "Q4 2018, growth scare and liquidity withdrawal"},
}


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------
def historical_var(returns: np.ndarray, conf: float = 0.95) -> float:
    return float(-np.percentile(returns, (1 - conf) * 100))


def parametric_var(returns: np.ndarray, conf: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return float(-(mu + stats.norm.ppf(1 - conf) * sigma))


def cornish_fisher_var(returns: np.ndarray, conf: float = 0.95) -> float:
    """VaR adjusted for skewness and excess kurtosis.

    Strategy returns are rarely normal, and a Gaussian VaR on a left-skewed,
    fat-tailed series systematically understates the loss it is supposed to
    bound. Cornish-Fisher corrects the quantile using the sample's own third
    and fourth moments -- cheap, and much closer to the truth than parametric.
    """
    mu, sigma = returns.mean(), returns.std(ddof=1)
    s = float(pd.Series(returns).skew())
    k = float(pd.Series(returns).kurtosis())
    z = stats.norm.ppf(1 - conf)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * (s**2) / 36
    )
    return float(-(mu + z_cf * sigma))


def cvar(returns: np.ndarray, conf: float = 0.95) -> float:
    var = historical_var(returns, conf)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) else var


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------
def compute_beta_alpha(
    strategy: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS
) -> dict:
    """Beta, alpha, and information ratio on DATE-ALIGNED series.

    The join is on dates, not on position. The original implementation
    subsampled the benchmark to match the strategy's observation count, which
    silently compared different time periods.
    """
    df = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 20:
        return {"beta": None, "alpha_annualized": None, "information_ratio": None,
                "n_aligned_observations": int(len(df)),
                "note": "Fewer than 20 overlapping dates; beta not estimated."}

    s, b = df["s"].to_numpy(), df["b"].to_numpy()
    var_b = np.var(b, ddof=1)
    beta = float(np.cov(s, b)[0, 1] / var_b) if var_b > 0 else float("nan")
    alpha_per_period = float(s.mean() - beta * b.mean())
    active = s - b
    ir = float(active.mean() / active.std(ddof=1)) if active.std(ddof=1) > 0 else float("nan")
    r2 = float(np.corrcoef(s, b)[0, 1] ** 2) if len(s) > 2 else float("nan")

    return {
        "beta": round(beta, 3) if np.isfinite(beta) else None,
        "alpha_per_period": round(alpha_per_period, 6),
        "alpha_annualized": round(alpha_per_period * periods_per_year, 4),
        "information_ratio": round(ir * np.sqrt(periods_per_year), 3) if np.isfinite(ir) else None,
        "r_squared_vs_benchmark": round(r2, 3) if np.isfinite(r2) else None,
        "n_aligned_observations": int(len(df)),
        "note": "Aligned by an inner join on dates.",
    }


# ---------------------------------------------------------------------------
# Factor risk decomposition
# ---------------------------------------------------------------------------
def factor_risk_decomposition(
    strategy_returns: pd.Series, panel: pd.DataFrame, factor_cols: list[str], label_free: bool = True
) -> dict:
    """Attribute strategy variance to factor exposures vs. idiosyncratic risk.

    Method: build a daily factor-mimicking return for each factor (the return
    of a long-top-tercile / short-bottom-tercile portfolio on that factor,
    formed from the previous day's ranking so it is tradeable), then regress
    the strategy's returns on those. R-squared is the share of variance the
    factors explain; the residual is genuinely idiosyncratic.

    This is what turns "the strategy has 12% volatility" into "the volatility
    is mostly a momentum bet, and momentum is what will hurt it".
    """
    if not factor_cols or panel.empty:
        return {"status": "no_factors"}

    fmp = build_factor_mimicking_returns(panel, factor_cols)
    if fmp.empty:
        return {"status": "insufficient_data", "note": "Could not build factor-mimicking portfolios."}

    df = pd.concat([strategy_returns.rename("strategy"), fmp], axis=1).dropna()
    if len(df) < 40:
        return {"status": "insufficient_data", "n_aligned": int(len(df)),
                "note": "Fewer than 40 aligned observations for the decomposition."}

    y = df["strategy"].to_numpy()
    used = [c for c in fmp.columns if c in df.columns and df[c].std() > 0]
    if not used:
        return {"status": "insufficient_data", "note": "All factor-mimicking series are constant."}

    X = df[used].to_numpy()
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    fitted = Xd @ beta
    resid = y - fitted

    total_var = float(np.var(y, ddof=1))
    resid_var = float(np.var(resid, ddof=1))
    r2 = 1 - resid_var / total_var if total_var > 0 else float("nan")

    # Per-factor variance contribution: beta_i^2 * var(f_i) plus covariance
    # terms; reported as the marginal share so it sums approximately to R^2.
    contributions = {}
    for i, f in enumerate(used):
        b = float(beta[i + 1])
        contrib = b * float(np.cov(X[:, i], y)[0, 1])
        contributions[f] = {
            "exposure": round(b, 4),
            "variance_share": round(contrib / total_var, 4) if total_var > 0 else None,
        }

    ranked = sorted(
        contributions.items(), key=lambda kv: -(abs(kv[1]["variance_share"] or 0))
    )

    return {
        "status": "ok",
        "r_squared": round(float(r2), 4) if np.isfinite(r2) else None,
        "systematic_variance_share": round(float(r2), 4) if np.isfinite(r2) else None,
        "idiosyncratic_variance_share": round(1 - float(r2), 4) if np.isfinite(r2) else None,
        "annualized_idiosyncratic_vol": round(float(np.sqrt(resid_var * TRADING_DAYS)), 4),
        "factor_exposures": contributions,
        "dominant_risk_factors": [k for k, _ in ranked[:3]],
        "n_observations": int(len(df)),
        "method": "factor-risk-v1: OLS of strategy returns on daily long/short "
        "factor-mimicking portfolio returns formed from lagged rankings.",
    }


def build_factor_mimicking_returns(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Daily return of a long/short portfolio formed on each factor.

    Rankings are LAGGED one day before the return is realized, so each series
    is a tradeable portfolio rather than a same-day fit.
    """
    if "ret_1d" not in panel.columns:
        if "close_px" not in panel.columns:
            return pd.DataFrame()
        panel = panel.copy()
        panel["ret_1d"] = panel.sort_values(["ticker", "date"]).groupby("ticker")["close_px"].pct_change()
    if panel["ret_1d"].isna().all():
        return pd.DataFrame()

    out = {}
    for f in factor_cols:
        if f not in panel.columns:
            continue
        sub = panel[["date", "ticker", f, "ret_1d"]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values(["ticker", "date"])
        # Lag the factor so today's return is earned on yesterday's ranking.
        sub["signal"] = sub.groupby("ticker")[f].shift(1)
        sub = sub.dropna(subset=["signal"])
        series = {}
        for date, grp in sub.groupby("date"):
            if len(grp) < 4:
                continue
            hi, lo = grp["signal"].quantile(0.67), grp["signal"].quantile(0.33)
            longs = grp.loc[grp["signal"] >= hi, "ret_1d"]
            shorts = grp.loc[grp["signal"] <= lo, "ret_1d"]
            if len(longs) and len(shorts):
                series[pd.Timestamp(date)] = float(longs.mean() - shorts.mean())
        if len(series) > 20:
            out[f] = pd.Series(series)
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Concentration and liquidity
# ---------------------------------------------------------------------------
def concentration_risk(weights: dict, dollar_volume: dict, aum: float = 10_000_000.0) -> dict:
    """Position concentration and how long it would take to exit.

    Days-to-liquidate assumes the strategy can be at most 20% of a name's
    daily volume without moving the price materially -- the conventional
    participation ceiling. A position needing many days to exit is a risk that
    a weight cap alone does not express.
    """
    if not weights:
        return {"status": "no_positions"}

    gross = sum(abs(w) for w in weights.values())
    shares = {t: abs(w) / gross for t, w in weights.items()} if gross > 0 else {}
    hhi = float(sum(s**2 for s in shares.values()))

    liquidation = {}
    for t, w in weights.items():
        adv = dollar_volume.get(t)
        if not adv or adv <= 0:
            liquidation[t] = None
            continue
        position_value = abs(w) * aum
        liquidation[t] = round(position_value / (0.20 * adv), 4)

    worst = max((v for v in liquidation.values() if v is not None), default=None)

    return {
        "status": "ok",
        "n_positions": len(weights),
        "gross_exposure": round(gross, 4),
        "net_exposure": round(sum(weights.values()), 4),
        "largest_position": round(max(abs(w) for w in weights.values()), 4),
        "herfindahl_index": round(hhi, 4),
        "effective_n_positions": round(1 / hhi, 2) if hhi > 0 else None,
        "days_to_liquidate": liquidation,
        "worst_days_to_liquidate": worst,
        "concentration_flag": bool(hhi > 0.25),
        "liquidity_flag": bool(worst is not None and worst > 3.0),
        "note": "Days-to-liquidate assumes a 20% participation cap in each name's "
        f"average daily dollar volume at ${aum:,.0f} of strategy AUM.",
    }


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------
def scenario_analysis(returns: np.ndarray, beta: float | None) -> dict:
    """Statistical shocks plus replay of real historical crisis windows."""
    mu, sd = returns.mean(), returns.std(ddof=1)
    statistical = {
        "2sigma_daily_shock": round(float(mu - 2 * sd), 5),
        "3sigma_daily_shock": round(float(mu - 3 * sd), 5),
        "worst_observed_day": round(float(returns.min()), 5),
        "worst_observed_5d": round(float(pd.Series(returns).rolling(5).sum().min()), 5),
    }

    historical = {}
    for name, sc in HISTORICAL_SCENARIOS.items():
        if beta is None or not np.isfinite(beta):
            historical[name] = {"estimated_strategy_return": None, **sc,
                                "note": "No reliable beta estimate; cannot map this scenario."}
        else:
            historical[name] = {
                "estimated_strategy_return": round(float(beta * sc["benchmark_return"]), 4),
                **sc,
            }

    return {
        "statistical_shocks": statistical,
        "historical_scenarios": historical,
        "method": "Historical scenarios map a real benchmark drawdown through the "
        "strategy's estimated market beta. This captures directional exposure "
        "only -- it does not model the factor-correlation breakdown, liquidity "
        "withdrawal, or short-squeeze dynamics that make real crises worse than "
        "beta predicts.",
    }


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------
def run_risk_agent(
    backtest_result: dict,
    benchmark_returns: pd.Series | None = None,
    panel: pd.DataFrame | None = None,
    factor_cols: list[str] | None = None,
    dollar_volume: dict | None = None,
    aum: float = 10_000_000.0,
) -> dict:
    if backtest_result.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No successful backtest to analyze."}

    r = np.array(backtest_result["daily_returns"], dtype=float)
    dates = pd.DatetimeIndex([p["date"] for p in backtest_result["equity_curve"][1:]])
    strategy = pd.Series(r, index=dates[: len(r)])

    out = {
        "status": "ok",
        "n_observations": int(len(r)),
        "value_at_risk_daily": {
            "historical_95": round(historical_var(r, 0.95), 5),
            "historical_99": round(historical_var(r, 0.99), 5),
            "parametric_95": round(parametric_var(r, 0.95), 5),
            "cornish_fisher_95": round(cornish_fisher_var(r, 0.95), 5),
            "cornish_fisher_99": round(cornish_fisher_var(r, 0.99), 5),
        },
        "conditional_var_95": round(cvar(r, 0.95), 5),
        "conditional_var_99": round(cvar(r, 0.99), 5),
        "volatility_annualized": backtest_result["metrics"]["volatility_annualized"],
        "max_drawdown": backtest_result["metrics"]["max_drawdown"],
        "var_interpretation": "Daily VaR at 95% is the loss exceeded on roughly one "
        "trading day in twenty. Cornish-Fisher adjusts for the return series' own "
        "skew and fat tails and is the number to trust when they are material.",
    }

    beta_alpha = None
    if benchmark_returns is not None and len(benchmark_returns):
        beta_alpha = compute_beta_alpha(strategy, benchmark_returns)
    out["benchmark_relative"] = beta_alpha

    beta = beta_alpha.get("beta") if beta_alpha else None
    out["scenarios"] = scenario_analysis(r, beta)

    if panel is not None and factor_cols:
        out["factor_risk"] = factor_risk_decomposition(strategy, panel, factor_cols)

    latest_weights = {}
    if backtest_result.get("weights_history"):
        latest_weights = list(backtest_result["weights_history"].values())[-1]
    if latest_weights:
        out["concentration"] = concentration_risk(latest_weights, dollar_volume or {}, aum=aum)

    return out
