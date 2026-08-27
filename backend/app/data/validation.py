"""
Data validation -- the gate between ingestion and everything downstream.

Every check here answers a question that, left unasked, silently corrupts a
backtest rather than crashing it. Crashes are cheap; a quietly wrong Sharpe
is expensive. Each check returns a structured finding with a severity, and
the findings travel with the run so the Critic agent and the report can cite
them instead of re-deriving them.

SEVERITY CONTRACT
  error   -- the data is unusable for research; the run should not proceed
             on this asset (it is dropped, and the drop is reported).
  warning -- usable but the result must be qualified.
  info    -- worth recording for provenance, not a problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A single-day move this large in an adjusted series is far more likely to be
# an unadjusted corporate action than a real return. 2008 and 2020 both had
# real -20% index days, so the threshold sits well above that.
SPLIT_LIKE_RETURN = 0.35
STALE_PRICE_RUN = 5          # identical closes for this many consecutive days
MIN_USABLE_BARS = 120        # below this, factor warm-up windows cannot fill
MAX_MISSING_FRACTION = 0.10


def _finding(check: str, severity: str, detail: str, **extra) -> dict:
    return {"check": check, "severity": severity, "detail": detail, **extra}


def validate_price_frame(ticker: str, df: pd.DataFrame) -> list[dict]:
    """Validate one asset's OHLCV frame. Returns a list of findings."""
    findings: list[dict] = []

    if df is None or df.empty:
        return [_finding("empty_frame", "error", f"{ticker}: no rows returned.")]

    required = {"date", "open", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        return [_finding("schema", "error", f"{ticker}: missing columns {sorted(missing_cols)}.")]

    n = len(df)
    if n < MIN_USABLE_BARS:
        findings.append(
            _finding(
                "insufficient_history",
                "error",
                f"{ticker}: only {n} bars; factor warm-up needs at least {MIN_USABLE_BARS}.",
                n_bars=n,
            )
        )

    # --- ordering and duplication -----------------------------------------
    if not df["date"].is_monotonic_increasing:
        findings.append(_finding("unsorted_dates", "warning", f"{ticker}: dates not sorted ascending; sorted on load."))
    n_dupes = int(df["date"].duplicated().sum())
    if n_dupes:
        findings.append(
            _finding("duplicate_dates", "warning", f"{ticker}: {n_dupes} duplicate date rows; last kept.", count=n_dupes)
        )

    # --- missingness -------------------------------------------------------
    price_cols = ["open", "high", "low", "close"]
    n_missing = int(df[price_cols].isna().any(axis=1).sum())
    frac_missing = n_missing / max(n, 1)
    if frac_missing > MAX_MISSING_FRACTION:
        findings.append(
            _finding(
                "excessive_missing",
                "error",
                f"{ticker}: {frac_missing:.1%} of bars have missing prices "
                f"(limit {MAX_MISSING_FRACTION:.0%}).",
                fraction=round(frac_missing, 4),
            )
        )
    elif n_missing:
        findings.append(
            _finding("missing_bars", "info", f"{ticker}: {n_missing} bars with missing prices, forward-filled.", count=n_missing)
        )

    # --- sign and ordering sanity -----------------------------------------
    nonpositive = int((df[price_cols] <= 0).any(axis=1).sum())
    if nonpositive:
        findings.append(
            _finding("nonpositive_price", "error", f"{ticker}: {nonpositive} bars with a non-positive price.", count=nonpositive)
        )

    bad_hl = int((df["high"] < df["low"]).sum())
    if bad_hl:
        findings.append(_finding("high_below_low", "warning", f"{ticker}: {bad_hl} bars where high < low; repaired.", count=bad_hl))

    bad_range = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
    if bad_range:
        findings.append(
            _finding("close_outside_range", "warning", f"{ticker}: {bad_range} bars where close sits outside [low, high]; repaired.", count=bad_range)
        )

    neg_volume = int((df["volume"] < 0).sum())
    if neg_volume:
        findings.append(_finding("negative_volume", "warning", f"{ticker}: {neg_volume} bars with negative volume; zeroed.", count=neg_volume))

    # --- corporate actions -------------------------------------------------
    close = df["close"].astype(float)
    ret = close.pct_change()
    split_like = ret[ret.abs() > SPLIT_LIKE_RETURN]
    if len(split_like):
        dates = [str(pd.Timestamp(d).date()) for d in df.loc[split_like.index, "date"].head(5)]
        findings.append(
            _finding(
                "possible_unadjusted_split",
                "warning",
                f"{ticker}: {len(split_like)} single-day moves above "
                f"{SPLIT_LIKE_RETURN:.0%}. In an adjusted series these are more "
                f"likely unadjusted corporate actions than real returns. Dates: {dates}",
                count=int(len(split_like)),
            )
        )

    # --- staleness ---------------------------------------------------------
    same_as_prev = close.diff().eq(0)
    if same_as_prev.any():
        # Longest run of identical consecutive closes.
        grp = (~same_as_prev).cumsum()
        longest = int(same_as_prev.groupby(grp).sum().max())
        if longest >= STALE_PRICE_RUN:
            findings.append(
                _finding(
                    "stale_prices",
                    "warning",
                    f"{ticker}: {longest} consecutive bars with an unchanged close -- "
                    "likely a halted, illiquid, or stale-quoted name. Volatility and "
                    "mean-reversion factors are unreliable across such runs.",
                    longest_run=longest,
                )
            )

    # --- liquidity ---------------------------------------------------------
    dollar_vol = (df["volume"].astype(float) * close).replace(0, np.nan)
    median_dv = float(dollar_vol.median()) if dollar_vol.notna().any() else 0.0
    if median_dv and median_dv < 1_000_000:
        findings.append(
            _finding(
                "thin_liquidity",
                "warning",
                f"{ticker}: median daily dollar volume ${median_dv:,.0f}. Transaction "
                "costs and short borrow assumptions in the backtest understate reality "
                "for a name this thin.",
                median_dollar_volume=round(median_dv, 2),
            )
        )

    return findings


def validate_universe(frames: dict[str, pd.DataFrame]) -> dict:
    """Validate every asset. Returns findings plus the tradeable subset.

    Assets with an `error`-severity finding are excluded rather than silently
    carried, because a factor panel built on a broken series contaminates the
    cross-sectional z-scores of every other name on that date.
    """
    per_ticker: dict[str, list[dict]] = {}
    dropped: dict[str, str] = {}

    for ticker, df in frames.items():
        findings = validate_price_frame(ticker, df)
        per_ticker[ticker] = findings
        errors = [f for f in findings if f["severity"] == "error"]
        if errors:
            dropped[ticker] = errors[0]["detail"]

    usable = [t for t in frames if t not in dropped]
    all_findings = [f for fs in per_ticker.values() for f in fs]

    return {
        "per_ticker": per_ticker,
        "usable_tickers": usable,
        "dropped_tickers": dropped,
        "n_errors": sum(1 for f in all_findings if f["severity"] == "error"),
        "n_warnings": sum(1 for f in all_findings if f["severity"] == "warning"),
        "n_info": sum(1 for f in all_findings if f["severity"] == "info"),
        "passed": len(dropped) == 0,
        "summary": (
            f"{len(usable)}/{len(frames)} assets usable; "
            f"{sum(1 for f in all_findings if f['severity'] == 'warning')} warnings."
        ),
    }


def check_calendar_alignment(frames: dict[str, pd.DataFrame]) -> dict:
    """Assets in one cross-section must share a trading calendar.

    A cross-sectional z-score computed on a date where only three of six names
    have a bar is not a ranking of the universe -- it is a ranking of whoever
    happened to trade. This reports the overlap so the factor layer can
    restrict itself to common dates.
    """
    if not frames:
        return {"n_common_dates": 0, "coverage": {}, "passed": False}

    date_sets = {t: set(pd.DatetimeIndex(f["date"]).normalize()) for t, f in frames.items()}
    common = set.intersection(*date_sets.values()) if date_sets else set()
    union = set.union(*date_sets.values()) if date_sets else set()

    coverage = {t: round(len(common) / max(len(ds), 1), 4) for t, ds in date_sets.items()}
    weakest = min(coverage.values()) if coverage else 0.0

    return {
        "n_common_dates": len(common),
        "n_union_dates": len(union),
        "coverage": coverage,
        "weakest_coverage": weakest,
        "passed": weakest >= 0.90,
        "detail": (
            f"{len(common)} dates common to all {len(frames)} assets "
            f"({weakest:.1%} coverage for the weakest). "
            + (
                "Below 90% -- the cross-section is uneven and factor z-scores on "
                "sparse dates rank whoever traded, not the universe."
                if weakest < 0.90
                else "Cross-section is well aligned."
            )
        ),
        "common_dates": sorted(common),
    }
