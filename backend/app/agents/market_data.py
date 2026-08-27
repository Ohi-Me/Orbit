"""
Market Data Agent
=================
Produces a clean, validated, calendar-aligned panel of OHLCV data plus the
macro frame, and reports exactly what it did to the data on the way.

Responsibilities, in order:
  1. Fetch  -- real prices (yfinance) with an honest synthetic fallback.
  2. Validate -- schema, corporate actions, staleness, liquidity, missingness.
  3. Clean  -- forward-fill gaps, winsorize outliers, repair OHLC ordering.
  4. Align  -- restrict every asset to the shared trading calendar.
  5. Report -- a per-asset quality record the Critic agent and report cite.

THE CLEANING BUG THAT MATTERS (kept from the original build, still true):
the outlier z-score is computed on a window ENDING BEFORE the point being
judged. An earlier version included the point in its own window, which
inflated the window's standard deviation and let large outliers escape
detection entirely. Do not "simplify" the shift(1) away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.providers import prices as price_provider
from app.data.validation import check_calendar_alignment, validate_universe

OUTLIER_Z = 6.0


def clean_asset_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean one asset's raw OHLCV frame; return (clean_df, quality_report).

    Every mutation is counted and reported so the Critic agent can audit what
    was changed rather than trusting that nothing was.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    missing_mask = df[cols].isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    df[cols] = df[cols].ffill().bfill()

    # Non-positive prices cannot be log-differenced; treat as missing and fill.
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    n_nonpos = int(nonpos.sum())
    if n_nonpos:
        df.loc[nonpos, ["open", "high", "low", "close"]] = np.nan
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill().bfill()

    log_ret = np.log(df["close"]).diff()
    # IMPORTANT: roll stats come from the window ENDING BEFORE the current
    # point (shift(1)) so an outlier cannot inflate the threshold it is being
    # judged against. See module docstring.
    prior = log_ret.shift(1)
    roll_mean = prior.rolling(20, min_periods=5).mean()
    roll_std = prior.rolling(20, min_periods=5).std().replace(0, np.nan)
    zscore = (log_ret - roll_mean) / roll_std
    outlier_mask = (zscore.abs() > OUTLIER_Z).fillna(False)
    n_outliers = int(outlier_mask.sum())

    if n_outliers:
        capped = log_ret.copy()
        cap = OUTLIER_Z * roll_std
        capped[outlier_mask] = np.sign(log_ret[outlier_mask]) * cap[outlier_mask]
        fixed = df["close"].to_numpy(dtype=float).copy()
        for i in np.where(outlier_mask.to_numpy())[0]:
            if i == 0 or not np.isfinite(capped.iloc[i]):
                continue
            fixed[i] = fixed[i - 1] * np.exp(capped.iloc[i])
        df["close"] = fixed

    # Enforce OHLC ordering after any close repair.
    n_bad_range = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume"] = df["volume"].clip(lower=0).fillna(0)

    quality_report = {
        "n_rows": int(len(df)),
        "n_missing_bars_filled": n_missing,
        "n_nonpositive_prices_repaired": n_nonpos,
        "n_price_outliers_winsorized": n_outliers,
        "n_ohlc_ranges_repaired": n_bad_range,
        "outlier_threshold_sigma": OUTLIER_Z,
        "first_date": str(df["date"].iloc[0].date()) if len(df) else None,
        "last_date": str(df["date"].iloc[-1].date()) if len(df) else None,
    }
    return df, quality_report


def load_universe(
    tickers: list[str],
    n_days: int = 756,
    seed: int = 42,
    prefer_live: bool | None = None,
    align_calendar: bool = True,
) -> dict:
    """Fetch, validate, clean, and align the universe.

    Returns a dict rather than a tuple because this agent now produces four
    distinct things (frames, benchmark, quality, provenance) and positional
    unpacking of four values at every call site is how mismatches happen.
    """
    raw_frames, bench_raw, provenance = price_provider.fetch_prices(
        tickers, n_days=n_days, seed=seed, prefer_live=prefer_live
    )

    validation = validate_universe(raw_frames)
    usable = validation["usable_tickers"]
    if not usable:
        return {
            "status": "no_usable_data",
            "frames": {},
            "benchmark": None,
            "quality_reports": {},
            "provenance": provenance,
            "validation": validation,
        }

    clean: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for t in usable:
        clean[t], reports[t] = clean_asset_frame(raw_frames[t])

    bench_clean, bench_report = clean_asset_frame(bench_raw)
    reports["__benchmark__"] = bench_report

    alignment = check_calendar_alignment(clean)
    if align_calendar and alignment["n_common_dates"] >= 120:
        common = pd.DatetimeIndex(alignment["common_dates"])
        for t in list(clean):
            f = clean[t]
            clean[t] = f[f["date"].isin(common)].reset_index(drop=True)
            reports[t]["n_rows_after_alignment"] = int(len(clean[t]))
        bench_clean = bench_clean[bench_clean["date"].isin(common)].reset_index(drop=True)

    # Do not carry the full date list into the run record; it is large and the
    # count plus range already answer every question anyone asks of it.
    alignment_summary = {k: v for k, v in alignment.items() if k != "common_dates"}

    return {
        "status": "ok",
        "frames": clean,
        "benchmark": bench_clean,
        "quality_reports": reports,
        "provenance": provenance,
        "validation": {k: v for k, v in validation.items() if k != "per_ticker"},
        "validation_detail": validation["per_ticker"],
        "alignment": alignment_summary,
        "tickers": list(clean.keys()),
        "is_synthetic": provenance.get("is_synthetic", False),
    }
