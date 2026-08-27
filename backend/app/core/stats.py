"""
Statistical utilities for evaluating strategy and signal performance.

WHY THIS MODULE EXISTS
----------------------
The first version of this platform reported a plain one-sample t-test on
signal returns and a raw Sharpe ratio. Both are optimistic in ways that
matter specifically in this setting, and both were flagged in review:

1. OVERLAPPING RETURNS. The label is a 21-day forward return sampled every
   trading day, so consecutive observations share 20 of 21 days of price
   path. They are heavily autocorrelated, which violates the independence
   assumption a plain t-test rests on and inflates |t| by roughly sqrt(h)
   for horizon h. `newey_west_tstat` corrects the standard error for that
   autocorrelation (HAC), which is the standard fix.

2. CROSS-SECTIONAL DEPENDENCE. Pooling several tickers on the same date
   adds a second dependence axis: six names in one market move together, so
   six observations are not six independent draws. `effective_sample_size`
   discounts the count by the average pairwise correlation so the reported
   n reflects information, not row count.

3. MULTIPLE TESTING. Comparing several models across several folds and
   picking the winner means the best Sharpe is partly the luckiest Sharpe.
   `deflated_sharpe_ratio` adjusts the significance threshold for how many
   configurations were actually tried, following Bailey & Lopez de Prado.

None of these make a weak signal look strong -- they all make it look
weaker, which is the point. A number that survives them is worth reporting.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Autocorrelation-robust inference
# ---------------------------------------------------------------------------
def newey_west_tstat(returns: np.ndarray, lags: int | None = None) -> dict:
    """t-statistic for mean(returns) != 0 with a Newey-West (HAC) standard error.

    `lags` defaults to the overlap horizon implied by the data via the
    standard Bartlett rule-of-thumb floor(4 * (n/100)^(2/9)). When the caller
    knows the true overlap (e.g. a 21-day forward return sampled daily), pass
    lags=20 explicitly -- that is strictly more correct than the rule.

    Returns the naive t alongside the corrected one so the inflation factor
    is visible rather than silently absorbed.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return {
            "n": int(n),
            "mean": float(r.mean()) if n else None,
            "t_stat": None,
            "p_value": None,
            "naive_t_stat": None,
            "inflation_factor": None,
            "lags": None,
            "note": "Too few observations for a meaningful test (need >= 8).",
        }

    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 2))

    mu = r.mean()
    resid = r - mu

    # Long-run variance: gamma_0 + 2 * sum_k w_k * gamma_k, Bartlett weights.
    gamma0 = float(resid @ resid) / n
    lrv = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(resid[k:] @ resid[:-k]) / n
        lrv += 2.0 * w * gamma_k

    # A negative long-run variance estimate is possible in small samples with
    # strong negative autocorrelation; fall back to the naive variance rather
    # than returning a NaN t-stat.
    if lrv <= 0:
        lrv = gamma0
        degenerate = True
    else:
        degenerate = False

    se_hac = np.sqrt(lrv / n)
    se_naive = r.std(ddof=1) / np.sqrt(n)

    t_hac = mu / se_hac if se_hac > 0 else np.nan
    t_naive = mu / se_naive if se_naive > 0 else np.nan
    p_hac = 2 * (1 - stats.norm.cdf(abs(t_hac))) if np.isfinite(t_hac) else None

    return {
        "n": int(n),
        "mean": float(mu),
        "t_stat": round(float(t_hac), 3) if np.isfinite(t_hac) else None,
        "p_value": round(float(p_hac), 4) if p_hac is not None else None,
        "naive_t_stat": round(float(t_naive), 3) if np.isfinite(t_naive) else None,
        "inflation_factor": round(float(abs(t_naive / t_hac)), 2)
        if np.isfinite(t_hac) and np.isfinite(t_naive) and t_hac != 0
        else None,
        "lags": int(lags),
        "note": "Newey-West HAC standard error; corrects for the autocorrelation "
        "induced by overlapping forward-return windows."
        + (" Long-run variance estimate was non-positive; fell back to naive variance." if degenerate else ""),
    }


def effective_sample_size(n_obs: int, n_assets: int, avg_correlation: float, overlap: int = 1) -> dict:
    """Discount a raw observation count for cross-sectional and serial dependence.

    Cross-section: n_assets observations on one date carry the information of
    n_assets / (1 + (n_assets - 1) * rho) independent ones.
    Time: an h-day overlapping window sampled daily carries roughly 1/h of
    the independent observations.

    This is an approximation, deliberately a conservative one -- it exists to
    stop a table reporting "n = 2,760" when the independent information is
    closer to 30.
    """
    rho = float(np.clip(avg_correlation, 0.0, 0.999))
    cross_factor = n_assets / (1.0 + (n_assets - 1) * rho) if n_assets > 0 else 1.0
    n_dates = n_obs / max(n_assets, 1)
    independent_dates = n_dates / max(overlap, 1)
    ess = independent_dates * cross_factor
    return {
        "raw_n": int(n_obs),
        "effective_n": int(max(1, round(ess))),
        "avg_cross_correlation": round(rho, 3),
        "overlap_days": int(overlap),
        "note": "Effective sample size after discounting for overlapping return "
        "windows and cross-sectional correlation. Significance should be read "
        "against the effective n, not the raw row count.",
    }


# ---------------------------------------------------------------------------
# Sharpe ratio inference
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark: float = 0.0,
    periods_per_year: float = 1.0,
) -> float | None:
    """P(true Sharpe > benchmark), adjusting for non-normal returns.

    Bailey & Lopez de Prado (2012). A Sharpe of 1.0 on 20 observations of
    skewed, fat-tailed returns is much weaker evidence than the same number
    on 200 well-behaved ones, and PSR is what makes that difference visible.

    UNITS -- the easy way to get this badly wrong: the PSR formula requires
    the Sharpe expressed in the SAME periodicity as n_obs. Feeding it an
    annualized Sharpe alongside a count of monthly observations overstates
    the statistic by sqrt(periods_per_year) and will report near-certainty
    for a result that is actually a coin flip. Pass `periods_per_year` (12
    for monthly rebalances, 252 for daily) and the de-annualization happens
    here; leave it at 1.0 only if `sharpe` is already per-period.
    """
    if n_obs < 4 or not np.isfinite(sharpe):
        return None
    sr = sharpe / np.sqrt(periods_per_year) if periods_per_year > 0 else sharpe
    bm = benchmark / np.sqrt(periods_per_year) if periods_per_year > 0 else benchmark
    denom = 1.0 - skew * sr + (excess_kurtosis / 4.0) * sr**2
    if denom <= 0:
        return None
    z = (sr - bm) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    periods_per_year: float = 1.0,
) -> dict:
    """Sharpe significance after accounting for the number of configurations tried.

    The expected maximum Sharpe from n_trials independent *worthless*
    strategies is strictly greater than zero and grows with n_trials. DSR
    tests the observed Sharpe against that inflated null rather than against
    zero, so "best of 12 model/factor combinations" is judged as what it is.

    `sharpe` is annualized and `periods_per_year` de-annualizes it internally
    (see probabilistic_sharpe_ratio for why that matters). The threshold is
    reported back in annualized units so it is comparable to the input.
    """
    if n_obs < 4 or n_trials < 1 or not np.isfinite(sharpe):
        return {
            "deflated_sharpe": None,
            "expected_max_sharpe_under_null": None,
            "n_trials_considered": int(n_trials),
            "significant_at_95": None,
            "note": "Insufficient observations to deflate.",
        }

    ppy = periods_per_year if periods_per_year > 0 else 1.0
    sr_period = sharpe / np.sqrt(ppy)

    # Expected maximum of n_trials standard normals (Euler-Mascheroni approx).
    if n_trials == 1:
        expected_max_z = 0.0
    else:
        gamma = 0.5772156649
        e = np.e
        expected_max_z = (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + gamma * stats.norm.ppf(
            1 - 1.0 / (n_trials * e)
        )

    # Scale the null threshold by the sampling error of a per-period Sharpe.
    sharpe_std = np.sqrt(
        (1 - skew * sr_period + (excess_kurtosis / 4.0) * sr_period**2) / max(n_obs - 1, 1)
    )
    threshold_period = expected_max_z * sharpe_std
    threshold_annual = threshold_period * np.sqrt(ppy)

    psr = probabilistic_sharpe_ratio(
        sharpe, n_obs, skew, excess_kurtosis, benchmark=threshold_annual, periods_per_year=ppy
    )
    return {
        "deflated_sharpe": round(float(psr), 4) if psr is not None else None,
        "expected_max_sharpe_under_null": round(float(threshold_annual), 4),
        "observed_sharpe": round(float(sharpe), 4),
        "n_trials_considered": int(n_trials),
        "n_observations": int(n_obs),
        "significant_at_95": bool(psr is not None and psr > 0.95),
        "note": "Probability the true Sharpe exceeds the best a worthless strategy "
        f"would be expected to produce across {n_trials} trials, on {n_obs} "
        "observations. Below 0.95 means the result is not distinguishable from "
        "selection luck. Threshold shown in annualized units.",
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def stationary_bootstrap_ci(
    returns: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    expected_block: int = 5,
    conf: float = 0.95,
    seed: int = 42,
) -> dict:
    """Confidence interval for a statistic of a return series.

    Uses Politis-Romano stationary bootstrap (geometric block lengths) rather
    than an i.i.d. bootstrap, because resampling autocorrelated returns one
    at a time destroys exactly the dependence structure that makes the naive
    interval too narrow in the first place.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "n_boot": 0,
                "note": "Too few observations to bootstrap (need >= 8)."}

    rng = np.random.default_rng(seed)
    p = 1.0 / max(expected_block, 1)
    stats_out = np.empty(n_boot)

    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for j in range(n):
            idx[j] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        try:
            stats_out[b] = stat_fn(r[idx])
        except Exception:
            stats_out[b] = np.nan

    stats_out = stats_out[np.isfinite(stats_out)]
    if len(stats_out) < n_boot * 0.5:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "n_boot": int(len(stats_out)),
                "note": "Bootstrap statistic failed on too many resamples."}

    alpha = (1 - conf) / 2
    return {
        "point_estimate": round(float(stat_fn(r)), 4),
        "ci_low": round(float(np.percentile(stats_out, alpha * 100)), 4),
        "ci_high": round(float(np.percentile(stats_out, (1 - alpha) * 100)), 4),
        "confidence": conf,
        "n_boot": int(len(stats_out)),
        "note": "Stationary (block) bootstrap, preserving serial dependence.",
    }


def sharpe_of(returns: np.ndarray, periods_per_year: float = 12.0) -> float:
    """Annualized Sharpe of a period-return series. Zero risk-free rate assumed
    and stated -- with a non-zero rate this becomes an excess-return Sharpe."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Cross-sectional signal quality
# ---------------------------------------------------------------------------
def information_coefficient(signal: np.ndarray, forward_return: np.ndarray, method: str = "spearman") -> float | None:
    """Rank correlation between a signal and the return it is meant to predict.

    IC is how a quant desk actually judges a factor -- it measures monotonic
    ranking skill directly, independent of any threshold, position sizing, or
    classification cutoff. Spearman by default because outliers in forward
    returns otherwise dominate a Pearson estimate.
    """
    s = np.asarray(signal, dtype=float)
    f = np.asarray(forward_return, dtype=float)
    mask = np.isfinite(s) & np.isfinite(f)
    if mask.sum() < 4:
        return None
    s, f = s[mask], f[mask]
    if np.all(s == s[0]) or np.all(f == f[0]):
        return None
    if method == "spearman":
        rho, _ = stats.spearmanr(s, f)
    else:
        rho = np.corrcoef(s, f)[0, 1]
    return float(rho) if np.isfinite(rho) else None


def ic_summary(ic_series: list[float]) -> dict:
    """Aggregate a time series of per-date ICs.

    The information ratio of the IC series (mean/std) is the standard summary
    -- it captures both how strong the ranking skill is and how consistently
    it shows up, which mean IC alone hides.
    """
    arr = np.array([x for x in ic_series if x is not None and np.isfinite(x)], dtype=float)
    if len(arr) < 3:
        return {"mean_ic": None, "ic_std": None, "ic_ir": None, "hit_rate": None, "n_periods": int(len(arr))}
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1))
    return {
        "mean_ic": round(mean_ic, 4),
        "ic_std": round(std_ic, 4),
        "ic_ir": round(mean_ic / std_ic, 3) if std_ic > 0 else None,
        "hit_rate": round(float((arr > 0).mean()), 3),
        "n_periods": int(len(arr)),
        "note": "IC = per-date Spearman rank correlation between signal and forward "
        "return. IC_IR = mean/std of that series. Mean IC above ~0.03 with a "
        "positive IR is a conventional threshold for a usable equity factor.",
    }
