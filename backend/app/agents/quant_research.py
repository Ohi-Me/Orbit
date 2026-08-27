"""
Quant Research Agent
====================
Builds the factor panel: one row per (ticker, date), one column per factor,
plus the forward-return label. This is the feature layer the ML Research and
Backtesting agents both consume.

THREE LEAKAGE RULES, ENFORCED HERE
----------------------------------
1. TIME. Every factor uses only information available as of date t -- rolling
   windows that end at t, never centered, never forward.

2. PUBLICATION. Fundamental factors read through `edgar.pit_series`, which
   steps only on a filing's *filed* date, not its period end. Apple's FY2023
   revenue is not a fact about June 2023; it is a fact about November 2023,
   when the 10-K was filed.

3. CONSTANCY IS ALSO LEAKAGE. This is the subtle one, and the original build
   got it wrong. Sentiment and earnings-surprise were computed once from
   today's headlines and then written into every historical row as a constant
   per ticker. A constant is not a signal -- but worse, a constant derived
   from *today* embeds today's knowledge into every past date, so the
   backtest silently knew which names would do well. Both factors are now
   genuinely time-varying and dated: sentiment is NaN before its first
   observation, and earnings surprise steps only on filing dates.
   `factor_coverage()` reports how much of the panel each factor actually
   covers so a factor that is mostly NaN can be excluded on evidence rather
   than assumed useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.stats import ic_summary, information_coefficient

# Factors derived only from price/volume. Always available.
PRICE_FACTORS = [
    "momentum_12_1",
    "momentum_3m",
    "volatility_20d",
    "volatility_60d",
    "mean_reversion_z",
    "liquidity_log_dollar_vol",
    "downside_beta_proxy",
]

# Factors requiring point-in-time fundamentals from EDGAR.
FUNDAMENTAL_FACTORS = [
    "earnings_yield",
    "book_to_price",
    "return_on_equity",
    "operating_margin",
    "leverage_ratio",
    "asset_growth",
    "earnings_surprise",
]

# Factors requiring dated news.
SENTIMENT_FACTORS = ["sentiment_score"]

# Macro is a shared time series, not cross-sectional. It enters as a regime
# conditioner rather than a rankable factor -- every name has the same value
# on a date, so it cannot rank a cross-section, but it can tell a model which
# regime the ranking is happening in.
MACRO_FACTORS = ["vix_percentile", "term_spread", "credit_spread"]

ALL_FACTORS = PRICE_FACTORS + FUNDAMENTAL_FACTORS + SENTIMENT_FACTORS + MACRO_FACTORS

# Kept for backwards compatibility with the original module's public name.
FACTOR_COLUMNS = ALL_FACTORS

LABEL = "fwd_return_21d"
LABEL_HORIZON = 21


# ---------------------------------------------------------------------------
# Price/volume factors
# ---------------------------------------------------------------------------
def _momentum(close: pd.Series, lookback: int, skip: int = 5) -> pd.Series:
    """12-1 style momentum: return over `lookback` days, skipping the most
    recent `skip` days to avoid short-term reversal contamination."""
    return close.shift(skip) / close.shift(skip + lookback) - 1.0


def _volatility(close: pd.Series, window: int) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(252)


def _mean_reversion_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    m = close.rolling(window).mean()
    s = close.rolling(window).std().replace(0, np.nan)
    return (close - m) / s


def _liquidity(volume: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    dollar_vol = (volume * close).rolling(window).mean()
    return np.log(dollar_vol.replace(0, np.nan))


def _downside_beta_proxy(close: pd.Series, window: int = 63) -> pd.Series:
    """Ratio of downside to total volatility -- an asymmetry measure.

    A name whose volatility is concentrated in down moves behaves differently
    in a drawdown than one with symmetric volatility, and plain 20d vol cannot
    tell them apart. Not a true beta (no market series here); named as a proxy.
    """
    ret = np.log(close).diff()
    down = ret.where(ret < 0)
    return down.rolling(window, min_periods=window // 3).std() / ret.rolling(
        window, min_periods=window // 3
    ).std().replace(0, np.nan)


# ---------------------------------------------------------------------------
# Fundamental factors (point-in-time)
# ---------------------------------------------------------------------------
def _ttm(pit_frame: pd.DataFrame, metric: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Trailing-twelve-month sum of a flow metric, point-in-time.

    Flow metrics (revenue, net income) are reported per-quarter and per-year;
    summing the last four quarterly observations as of each date gives a TTM
    figure that only ever uses already-filed numbers.
    """
    from app.data.providers.edgar import pit_series

    if pit_frame is None or pit_frame.empty:
        return pd.Series(np.nan, index=dates)

    sub = pit_frame[pit_frame["metric"] == metric].copy()
    if sub.empty:
        return pd.Series(np.nan, index=dates)

    # Quarterly rows only (10-Q, plus the 10-K's Q4-equivalent annual row).
    quarterly = sub[sub["form"].str.startswith("10-Q")].sort_values("filed_date")
    if len(quarterly) >= 4:
        quarterly = quarterly.set_index("filed_date")
        ttm = quarterly["value"].rolling(4).sum()
        ttm = ttm[~ttm.index.duplicated(keep="last")].sort_index()
        return ttm.reindex(ttm.index.union(dates)).ffill().reindex(dates)

    # Not enough quarterly history -- fall back to the annual figure.
    return pit_series(sub, metric, dates)


def _fundamental_factors(
    ticker: str,
    dates: pd.DatetimeIndex,
    close: pd.Series,
    pit_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute point-in-time fundamental factors for one asset.

    Every ratio here is a genuine fundamental quantity, replacing the naive
    price-derived 'value_proxy' and 'quality_proxy' the original build used
    and correctly labelled as weak.
    """
    from app.data.providers.edgar import pit_series

    out = pd.DataFrame(index=dates)
    if pit_frame is None or pit_frame.empty:
        for f in FUNDAMENTAL_FACTORS:
            out[f] = np.nan
        return out

    eps_ttm = _ttm(pit_frame, "eps_diluted", dates)
    net_income_ttm = _ttm(pit_frame, "net_income", dates)
    revenue_ttm = _ttm(pit_frame, "revenue", dates)
    operating_income_ttm = _ttm(pit_frame, "operating_income", dates)

    equity = pit_series(pit_frame, "stockholders_equity", dates)
    assets = pit_series(pit_frame, "total_assets", dates)
    liabilities = pit_series(pit_frame, "total_liabilities", dates)
    shares = pit_series(pit_frame, "shares_diluted", dates)

    price = close.reindex(dates)
    market_cap = shares * price

    # VALUE -- earnings yield (inverse P/E). Yield rather than P/E because a
    # near-zero denominator makes P/E explode; the yield stays well-behaved
    # and keeps its sign, which is what a cross-sectional rank needs.
    out["earnings_yield"] = (eps_ttm / price.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # VALUE -- book to price.
    out["book_to_price"] = (equity / market_cap.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # QUALITY -- return on equity.
    out["return_on_equity"] = (net_income_ttm / equity.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # QUALITY -- operating margin.
    out["operating_margin"] = (operating_income_ttm / revenue_ttm.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # RISK -- balance-sheet leverage.
    out["leverage_ratio"] = (liabilities / assets.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # GROWTH -- year-over-year asset growth. Negatively related to future
    # returns in the published cross-sectional literature (the asset-growth
    # anomaly), so it is included as a real candidate, not a filler column.
    out["asset_growth"] = assets.pct_change(252).replace([np.inf, -np.inf], np.nan)

    # SURPRISE -- year-over-year change in TTM EPS, scaled by its own
    # volatility. Steps only on filing dates, so it is genuinely dated rather
    # than a constant broadcast across history (see module docstring, rule 3).
    eps_yoy = eps_ttm - eps_ttm.shift(252)
    eps_scale = eps_ttm.rolling(504, min_periods=120).std().replace(0, np.nan)
    out["earnings_surprise"] = (eps_yoy / eps_scale).replace([np.inf, -np.inf], np.nan)

    return out


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------
def build_factor_panel(
    clean_frames: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame] | None = None,
    sentiment_series: dict[str, pd.Series] | None = None,
    macro_df: pd.DataFrame | None = None,
    label_horizon: int = LABEL_HORIZON,
) -> pd.DataFrame:
    """Assemble the long-format factor panel.

    Columns: date, ticker, <factors...>, fwd_return_21d.

    `sentiment_series` maps ticker -> a dated pd.Series of sentiment scores.
    Dates with no news observation stay NaN rather than being filled with 0:
    "no news" and "neutral news" are different states, and conflating them
    manufactures a signal where there is only silence.
    """
    fundamentals = fundamentals or {}
    sentiment_series = sentiment_series or {}
    rows = []

    for ticker, df in clean_frames.items():
        df = df.copy().reset_index(drop=True)
        dates = pd.DatetimeIndex(df["date"])
        close = pd.Series(df["close"].to_numpy(dtype=float), index=dates)
        volume = pd.Series(df["volume"].to_numpy(dtype=float), index=dates)

        feat = pd.DataFrame({"date": dates, "ticker": ticker})
        feat = feat.set_index(dates)

        # --- price/volume ---------------------------------------------------
        feat["momentum_12_1"] = _momentum(close, lookback=252, skip=21)
        feat["momentum_3m"] = _momentum(close, lookback=63, skip=5)
        feat["volatility_20d"] = _volatility(close, 20)
        feat["volatility_60d"] = _volatility(close, 60)
        feat["mean_reversion_z"] = _mean_reversion_zscore(close, 20)
        feat["liquidity_log_dollar_vol"] = _liquidity(volume, close)
        feat["downside_beta_proxy"] = _downside_beta_proxy(close)

        # --- fundamentals (point-in-time) -----------------------------------
        fund = _fundamental_factors(ticker, dates, close, fundamentals.get(ticker))
        for c in FUNDAMENTAL_FACTORS:
            feat[c] = fund[c].to_numpy()

        # --- sentiment (dated; NaN where no news exists) ---------------------
        s = sentiment_series.get(ticker)
        if s is not None and len(s):
            aligned = pd.Series(s.to_numpy(dtype=float), index=pd.DatetimeIndex(s.index))
            aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
            feat["sentiment_score"] = aligned.reindex(aligned.index.union(dates)).ffill().reindex(dates).to_numpy()
        else:
            feat["sentiment_score"] = np.nan

        # --- macro (shared across the cross-section) -------------------------
        if macro_df is not None and not macro_df.empty:
            m = macro_df.reindex(macro_df.index.union(dates)).ffill().reindex(dates)
            feat["vix_percentile"] = m.get("vix_percentile_2y", pd.Series(np.nan, index=dates)).to_numpy()
            feat["term_spread"] = m.get("term_spread_10y2y", pd.Series(np.nan, index=dates)).to_numpy()
            feat["credit_spread"] = m.get("hy_credit_spread", pd.Series(np.nan, index=dates)).to_numpy()
        else:
            feat["vix_percentile"] = np.nan
            feat["term_spread"] = np.nan
            feat["credit_spread"] = np.nan

        # --- accounting columns (NOT features) --------------------------------
        # ret_1d is the return realized ON date t. It is same-day information,
        # so it must never enter the factor set -- it is here only so the
        # Backtesting and Risk agents can compute P&L and factor-mimicking
        # portfolio returns without re-deriving prices. `ALL_FACTORS` is the
        # single source of truth for what a model is allowed to see, and
        # ret_1d is deliberately absent from it.
        feat["ret_1d"] = close.pct_change().to_numpy()
        feat["close_px"] = close.to_numpy()

        # --- label -----------------------------------------------------------
        # A FUTURE shift. Never usable as a feature; the Critic re-verifies this.
        feat[LABEL] = (close.shift(-label_horizon) / close - 1.0).to_numpy()

        rows.append(feat.reset_index(drop=True))

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factor diagnostics
# ---------------------------------------------------------------------------
def factor_coverage(panel: pd.DataFrame, factors: list[str] | None = None) -> dict:
    """Fraction of panel rows where each factor is present.

    A factor covering 4% of the panel is not a weak factor -- it is an absent
    one, and dropping every row it is missing from would delete the panel.
    This is what lets the pipeline choose its factor set on evidence.
    """
    factors = factors or ALL_FACTORS
    n = len(panel)
    out = {}
    for f in factors:
        if f not in panel.columns:
            out[f] = {"coverage": 0.0, "n_present": 0, "usable": False}
            continue
        present = int(panel[f].notna().sum())
        cov = present / max(n, 1)
        out[f] = {
            "coverage": round(cov, 4),
            "n_present": present,
            "usable": cov >= 0.30,
            "is_constant_per_ticker": bool(
                panel.groupby("ticker")[f].nunique(dropna=True).max() <= 1
            ) if present else False,
        }
    return out


def select_usable_factors(
    panel: pd.DataFrame, factors: list[str] | None = None, min_coverage: float = 0.30
) -> tuple[list[str], dict]:
    """Choose the factor set for this run from actual coverage.

    `factors` is the CANDIDATE set — normally the factors the plan asked for.
    Only candidates are measured and only candidates can be rejected. Passing
    None measures every known factor.

    This distinction matters for honest reporting: a factor the plan never
    requested has no coverage measurement, and listing it as "0.0% coverage,
    rejected" states a finding about a column that was never in the panel. The
    rejected list should contain factors that were asked for and found wanting,
    nothing else.

    Also drops any factor that is constant within every ticker: such a column
    carries no cross-sectional-through-time information and, if it came from
    present-day data, is an active look-ahead channel (module docstring, rule 3).
    """
    candidates = [f for f in (factors or ALL_FACTORS)]
    cov = factor_coverage(panel, candidates)
    selected, rejected = [], {}
    for f in candidates:
        info = cov.get(f)
        if info is None:
            continue
        if f not in panel.columns:
            rejected[f] = "not present in the factor panel"
        elif info["coverage"] < min_coverage:
            rejected[f] = f"coverage {info['coverage']:.1%} below {min_coverage:.0%} minimum"
        elif info.get("is_constant_per_ticker"):
            rejected[f] = (
                "constant within each ticker across the whole sample -- carries no "
                "time variation and risks embedding present-day information into "
                "historical rows"
            )
        else:
            selected.append(f)
    return selected, {
        "coverage": cov,
        "rejected": rejected,
        "min_coverage": min_coverage,
        "n_candidates": len(candidates),
        "candidates": candidates,
    }


def compute_factor_ic(panel: pd.DataFrame, factors: list[str], label: str = LABEL) -> dict:
    """Per-factor information coefficient time series and summary.

    IC is how a quant desk judges a factor: the rank correlation between the
    factor and the return it is meant to predict, computed cross-sectionally
    on each date. It is independent of any threshold, position size, or model,
    which makes it the cleanest available read on whether a factor has signal.
    """
    out = {}
    for f in factors:
        if f not in panel.columns:
            continue
        ics = []
        for _, grp in panel.groupby("date"):
            sub = grp[[f, label]].dropna()
            if len(sub) < 3:
                continue
            ic = information_coefficient(sub[f].to_numpy(), sub[label].to_numpy())
            if ic is not None:
                ics.append(ic)
        summary = ic_summary(ics)
        summary["factor"] = f
        out[f] = summary
    return out


def fama_macbeth(panel: pd.DataFrame, factors: list[str], label: str = LABEL) -> dict:
    """Fama-MacBeth cross-sectional regression.

    Runs one cross-sectional OLS of forward return on the factors per date,
    then tests whether the time series of each coefficient has a non-zero
    mean. This is the standard academic test for whether a factor is priced,
    and it answers a different question than the ML classifier does: it asks
    about the *linear premium* attached to a factor, controlling for the
    others, rather than about directional classification accuracy.

    IDENTIFICATION: a cross-sectional regression needs more names than
    regressors. With a 6-name universe and 14 factors the multivariate
    regression is not merely noisy, it is unidentified -- there is no unique
    solution, and returning coefficients anyway would be fabricating them.
    So the mode is chosen from the data:

        multivariate -- n_names >= n_factors + 3, all factors jointly.
        univariate   -- otherwise, one regression per factor. This estimates
                        each factor's premium WITHOUT controlling for the
                        others, which is a real and stated limitation:
                        correlated factors will each claim the same premium.

    Standard errors are Newey-West either way, because the coefficient series
    inherits the autocorrelation of the overlapping forward-return windows.
    """
    from app.core.stats import newey_west_tstat

    usable = [f for f in factors if f in panel.columns]
    if not usable:
        return {"status": "no_factors", "per_factor": {}}

    n_names = int(panel["ticker"].nunique())
    multivariate = n_names >= len(usable) + 3
    mode = "multivariate" if multivariate else "univariate"

    coefs: dict[str, list[float]] = {f: [] for f in usable}
    n_dates_used = 0

    if multivariate:
        for _, grp in panel.groupby("date"):
            sub = grp[usable + [label]].dropna()
            if len(sub) < len(usable) + 3:
                continue
            X = sub[usable].to_numpy(dtype=float)
            y = sub[label].to_numpy(dtype=float)
            mu, sd = X.mean(axis=0), X.std(axis=0)
            sd = np.where(sd == 0, np.nan, sd)
            Xz = (X - mu) / sd
            keep = ~np.isnan(Xz).any(axis=0)
            if not keep.any():
                continue
            Xz = np.nan_to_num(Xz[:, keep])
            Xd = np.column_stack([np.ones(len(Xz)), Xz])
            try:
                beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            for f, b in zip([f for f, k in zip(usable, keep) if k], beta[1:]):
                coefs[f].append(float(b))
            n_dates_used += 1
    else:
        dates_used = set()
        for f in usable:
            for date, grp in panel.groupby("date"):
                sub = grp[[f, label]].dropna()
                if len(sub) < 4:
                    continue
                x = sub[f].to_numpy(dtype=float)
                y = sub[label].to_numpy(dtype=float)
                sd = x.std()
                if sd == 0 or not np.isfinite(sd):
                    continue
                xz = (x - x.mean()) / sd
                Xd = np.column_stack([np.ones(len(xz)), xz])
                try:
                    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                coefs[f].append(float(beta[1]))
                dates_used.add(date)
        n_dates_used = len(dates_used)

    results = {}
    for f, series in coefs.items():
        if len(series) < 8:
            results[f] = {
                "factor": f,
                "n_periods": len(series),
                "status": "insufficient_periods",
                "detail": "Fewer than 8 cross-sections produced a coefficient.",
            }
            continue
        nw = newey_west_tstat(np.array(series), lags=LABEL_HORIZON - 1)
        results[f] = {
            "factor": f,
            "status": "ok",
            "mean_premium": round(float(np.mean(series)), 6),
            "t_stat": nw["t_stat"],
            "p_value": nw["p_value"],
            "naive_t_stat": nw["naive_t_stat"],
            "t_inflation_vs_naive": nw["inflation_factor"],
            "n_periods": len(series),
            "significant_at_95": bool(nw["p_value"] is not None and nw["p_value"] < 0.05),
        }

    caveat = (
        "Multivariate: each premium is estimated controlling for the other factors."
        if multivariate
        else (
            f"UNIVARIATE MODE -- the universe has {n_names} names but {len(usable)} "
            "factors, too few to identify a joint cross-sectional regression. Each "
            "premium below is estimated in isolation and does NOT control for the "
            "other factors, so correlated factors will each appear to earn the same "
            "premium. Widen the universe to at least "
            f"{len(usable) + 3} names for the multivariate estimate."
        )
    )

    return {
        "status": "ok",
        "mode": mode,
        "n_names": n_names,
        "n_factors": len(usable),
        "n_cross_sections": n_dates_used,
        "per_factor": results,
        "caveat": caveat,
        "method": "fama-macbeth-v1: per-date cross-sectional OLS on standardized "
        "factors, Newey-West HAC t-stats on the coefficient time series "
        f"(lags={LABEL_HORIZON - 1}, matching the forward-return overlap).",
    }


def cross_sectional_zscore(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Standardize each factor within each date.

    Cross-sectional standardization is the right normalization for a ranking
    strategy: it removes the market-wide level (which no long/short position
    can capture anyway) and puts every factor on a comparable scale. It uses
    only same-date information, so it introduces no look-ahead.
    """
    out = panel.copy()
    grouped = out.groupby("date")[factors]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    for f in factors:
        out[f + "_z"] = ((out[f] - mean[f]) / std[f]).clip(-3, 3)
    return out
