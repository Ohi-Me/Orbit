"""
Macroeconomic data provider (FRED).

Uses the public fredgraph CSV endpoint, which needs no API key. Series are
chosen to span the axes that actually move cross-sectional equity factor
returns rather than to be an impressive-looking list:

  DGS10  / DGS2   -- the level and (via the spread) the slope of the curve;
                     the term spread is the classic recession/regime proxy.
  T10Y2Y          -- that spread published directly.
  VIXCLS          -- implied volatility: the single best regime marker for
                     when momentum breaks down and mean-reversion works.
  BAMLH0A0HYM2    -- high-yield credit spread: risk appetite, and it leads
                     equity drawdowns more reliably than equity vol does.
  UNRATE / CPIAUCSL -- growth and inflation, the two macro factors most
                     equity factor timing literature conditions on.

RELEASE LAG IS REAL: CPI and unemployment are published weeks after the
month they describe. `fetch_macro` shifts any series flagged `lagged` by its
publication delay so a factor computed on date t cannot see a statistic that
was not yet released. Rates and VIX are observed same-day and are not shifted.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# series_id -> (platform name, publication lag in days)
MACRO_SERIES: dict[str, tuple[str, int]] = {
    "DGS10": ("treasury_10y", 0),
    "DGS2": ("treasury_2y", 0),
    "T10Y2Y": ("term_spread_10y2y", 0),
    "VIXCLS": ("vix", 0),
    "BAMLH0A0HYM2": ("hy_credit_spread", 0),
    "UNRATE": ("unemployment_rate", 21),
    "CPIAUCSL": ("cpi_index", 30),
}


class MacroFetchError(RuntimeError):
    pass


def _fetch_series(series_id: str, timeout: float = 25.0) -> pd.Series:
    resp = httpx.get(
        FRED_CSV.format(series=series_id),
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "agentic-quant-research-platform/1.0"},
    )
    if resp.status_code != 200:
        raise MacroFetchError(f"FRED returned HTTP {resp.status_code} for {series_id}")

    df = pd.read_csv(io.StringIO(resp.text))
    date_col = df.columns[0]
    value_col = next((c for c in df.columns[1:] if c.lower() != date_col.lower()), df.columns[-1])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # FRED encodes missing observations as "."; coerce turns those into NaN.
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    s = df.dropna(subset=[date_col]).set_index(date_col)[value_col].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s.sort_index()


def fetch_macro(start: str | None = None, series: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (macro_frame indexed by date, provenance).

    Each series is shifted forward by its publication lag before being
    forward-filled onto a daily index, so the value visible on date t is the
    value an investor could genuinely have known on date t.
    """
    series = series or MACRO_SERIES
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=365 * 12)).strftime("%Y-%m-%d")

    frames: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    lags: dict[str, int] = {}

    for series_id, (name, lag_days) in series.items():
        try:
            s = _fetch_series(series_id)
            if lag_days:
                # Move the observation to the date it was actually published.
                s.index = s.index + pd.Timedelta(days=lag_days)
            frames[name] = s[s.index >= pd.Timestamp(start)]
            lags[name] = lag_days
        except Exception as e:
            failures[name] = f"{type(e).__name__}: {str(e)[:120]}"

    if not frames:
        raise MacroFetchError(f"No macro series could be fetched. Failures: {failures}")

    df = pd.concat(frames, axis=1).sort_index().ffill()

    # Derived features that are more useful than the raw levels.
    if "treasury_10y" in df and "treasury_2y" in df and "term_spread_10y2y" not in df:
        df["term_spread_10y2y"] = df["treasury_10y"] - df["treasury_2y"]
    if "cpi_index" in df:
        # Year-over-year inflation from the index level (252 business days).
        df["cpi_yoy"] = df["cpi_index"].pct_change(252) * 100.0
    if "vix" in df:
        # Percentile rank of VIX in its own trailing 2y history -- a bounded,
        # stationary regime marker that a level in points is not.
        df["vix_percentile_2y"] = df["vix"].rolling(504, min_periods=60).rank(pct=True)

    provenance = {
        "provider": "fred_stlouisfed",
        "is_synthetic": False,
        "series": sorted(frames.keys()),
        "publication_lags_days": lags,
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "n_rows": int(len(df)),
        "failures": failures,
        "lag_policy": "series shifted forward by publication lag before use; "
        "value visible on date t was knowable on date t",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return df, provenance


def macro_regime_label(macro_df: pd.DataFrame, date) -> dict:
    """Classify the macro regime on a date from observable indicators.

    Deliberately a transparent rule, not a model: a hidden-Markov regime
    classifier here would add opacity without adding information the two
    indicators below do not already carry, and the platform's standing rule
    is that heuristics say what they are.
    """
    if macro_df.empty:
        return {"regime": "unknown", "basis": "no macro data"}
    ts = pd.Timestamp(date)
    row = macro_df[macro_df.index <= ts]
    if row.empty:
        return {"regime": "unknown", "basis": "date precedes macro history"}
    row = row.iloc[-1]

    vix_pct = row.get("vix_percentile_2y")
    spread = row.get("term_spread_10y2y")

    if vix_pct is not None and pd.notna(vix_pct) and vix_pct > 0.80:
        regime = "high_volatility"
    elif spread is not None and pd.notna(spread) and spread < 0:
        regime = "inverted_curve"
    elif vix_pct is not None and pd.notna(vix_pct) and vix_pct < 0.30:
        regime = "calm"
    else:
        regime = "normal"

    return {
        "regime": regime,
        "vix_percentile_2y": round(float(vix_pct), 3) if pd.notna(vix_pct) else None,
        "term_spread_10y2y": round(float(spread), 3) if spread is not None and pd.notna(spread) else None,
        "basis": "rule-based-regime-v1: VIX 2y percentile and 10y-2y term spread",
    }
