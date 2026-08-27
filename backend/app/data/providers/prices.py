"""
Market price provider.

Two backends behind one interface, following the platform's standing rule
that a missing dependency degrades honestly rather than silently:

  1. YahooPriceProvider  -- real daily OHLCV via yfinance.
  2. SyntheticPriceProvider -- the original regime-switching generator.

Which one ran is recorded in the returned provenance dict and stored on the
run (DataSnapshot), so a synthetic run can never be mistaken for a live one
after the fact. The Report and the UI both read that flag.

ADJUSTED PRICES: we request auto_adjust=True so close is split- and
dividend-adjusted. This matters more than it sounds: an unadjusted series
shows a 2-for-1 split as a -50% single-day return, which every momentum and
volatility factor in the platform would read as a real crash. The validation
layer separately checks for residual split-like jumps as a backstop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.data.synthetic import generate_benchmark, generate_universe

# Yahoo's own symbol for the S&P 500 index, used as the default benchmark.
DEFAULT_BENCHMARK = "^GSPC"

_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class PriceFetchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Live provider
# ---------------------------------------------------------------------------
def _normalize_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance frame into the platform's canonical OHLCV shape."""
    out = df.reset_index()
    rename = {}
    for c in out.columns:
        lc = str(c).lower().replace(" ", "_")
        if lc in {"date", "datetime", "index"}:
            rename[c] = "date"
        elif lc in {"open", "high", "low", "close", "volume"}:
            rename[c] = lc
        elif lc == "adj_close":
            rename[c] = "adj_close"
    out = out.rename(columns=rename)
    missing = [c for c in _COLUMNS if c not in out.columns]
    if missing:
        raise PriceFetchError(f"Provider frame missing columns: {missing}")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out[_COLUMNS].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0).astype("int64")
    return out


def fetch_live_prices(tickers: list[str], n_days: int, benchmark: str = DEFAULT_BENCHMARK) -> tuple[dict, pd.DataFrame, dict]:
    """Fetch real daily OHLCV. Raises PriceFetchError if the provider is unusable."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - guarded by capability probe
        raise PriceFetchError("yfinance is not installed") from e

    # Ask for calendar days generously: n_days is *trading* days, and we need
    # slack for weekends, holidays, and the factor warm-up window.
    calendar_days = int(n_days * 1.6) + 40
    start = (datetime.now(timezone.utc) - timedelta(days=calendar_days)).strftime("%Y-%m-%d")

    symbols = list(dict.fromkeys([*tickers, benchmark]))
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for sym in symbols:
        try:
            raw = yf.Ticker(sym).history(start=start, interval="1d", auto_adjust=True)
            if raw is None or raw.empty:
                failures[sym] = "provider returned no rows"
                continue
            frame = _normalize_yf_frame(raw)
            if len(frame) < 60:
                failures[sym] = f"only {len(frame)} bars returned"
                continue
            frames[sym] = frame.tail(n_days).reset_index(drop=True)
        except Exception as e:
            failures[sym] = f"{type(e).__name__}: {str(e)[:120]}"

    missing_tickers = [t for t in tickers if t not in frames]
    if missing_tickers:
        raise PriceFetchError(
            f"No usable data for {missing_tickers}. Details: "
            + "; ".join(f"{k}={v}" for k, v in failures.items())
        )

    bench_df = frames.get(benchmark)
    if bench_df is None:
        # A missing benchmark degrades beta/alpha only -- not worth failing the
        # whole run over, so synthesize an equal-weight proxy from the universe.
        bench_df = _equal_weight_proxy({t: frames[t] for t in tickers})
        bench_source = "equal_weight_universe_proxy"
    else:
        bench_source = benchmark

    universe = {t: frames[t] for t in tickers}
    all_dates = pd.concat([f["date"] for f in universe.values()])
    provenance = {
        "provider": "yfinance",
        "is_synthetic": False,
        "benchmark": bench_source,
        "start_date": str(all_dates.min().date()),
        "end_date": str(all_dates.max().date()),
        "n_rows": int(sum(len(f) for f in universe.values())),
        "failures": failures,
        "price_basis": "split/dividend-adjusted (auto_adjust=True)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return universe, bench_df, provenance


def _equal_weight_proxy(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an equal-weight index from the universe when no benchmark is available."""
    closes = []
    for t, f in frames.items():
        s = f.set_index("date")["close"]
        closes.append((s / s.iloc[0]).rename(t))
    idx = pd.concat(closes, axis=1).dropna().mean(axis=1) * 100.0
    return pd.DataFrame(
        {
            "date": idx.index,
            "open": idx.values,
            "high": idx.values,
            "low": idx.values,
            "close": idx.values,
            "volume": np.full(len(idx), 0, dtype="int64"),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic provider
# ---------------------------------------------------------------------------
def fetch_synthetic_prices(tickers: list[str], n_days: int, seed: int) -> tuple[dict, pd.DataFrame, dict]:
    raw = generate_universe(tickers, n_days=n_days, seed=seed)
    bench = generate_benchmark(n_days=n_days)
    return (
        raw,
        bench,
        {
            "provider": "synthetic_regime_switching",
            "is_synthetic": True,
            "benchmark": "synthetic_index",
            "seed": seed,
            "start_date": str(min(f["date"].min() for f in raw.values()).date()),
            "end_date": str(max(f["date"].max() for f in raw.values()).date()),
            "n_rows": int(sum(len(f) for f in raw.values())),
            "price_basis": "generated; no corporate actions to adjust for",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warning": "SYNTHETIC DATA -- results demonstrate methodology only and "
            "carry no information about real markets.",
        },
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: list[str],
    n_days: int = 756,
    seed: int = 42,
    prefer_live: bool | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
) -> tuple[dict, pd.DataFrame, dict]:
    """Return (universe_frames, benchmark_frame, provenance).

    Tries the live provider when enabled and falls back to synthetic on any
    failure, recording *why* it fell back in provenance["fallback_reason"] so
    a degraded run is visible rather than merely quiet.
    """
    settings = get_settings()
    want_live = settings.allow_live_market_data if prefer_live is None else prefer_live

    if want_live:
        try:
            return fetch_live_prices(tickers, n_days, benchmark=benchmark)
        except Exception as e:
            frames, bench, prov = fetch_synthetic_prices(tickers, n_days, seed)
            prov["fallback_reason"] = f"live provider unavailable: {type(e).__name__}: {str(e)[:200]}"
            prov["attempted_live"] = True
            return frames, bench, prov

    frames, bench, prov = fetch_synthetic_prices(tickers, n_days, seed)
    prov["attempted_live"] = False
    return frames, bench, prov
