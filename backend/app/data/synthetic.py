"""
Synthetic market data generator.

WHY SYNTHETIC: This platform is designed to plug into real data providers
(see MarketDataAgent docstring for the swap points). In this environment /
for a from-scratch demo, we generate statistically realistic multi-regime
price paths instead of pretending to hit a live market data API. Every
series is seeded, so results are reproducible run-to-run -- an explicit
requirement for the Critic Agent's leakage/robustness checks.

The generator deliberately injects:
  * regime switches (trend / mean-reversion / high-vol) so factor and ML
    experiments have something non-trivial to detect
  * missing bars and price outliers so MarketDataAgent has real cleaning
    work to do, not a no-op
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LIST = ["trend_up", "trend_down", "mean_revert", "high_vol"]


def _regime_params(regime: str) -> dict:
    return {
        "trend_up": dict(mu=0.00035, sigma=0.011, phi=0.0),
        "trend_down": dict(mu=-0.00030, sigma=0.013, phi=0.0),
        "mean_revert": dict(mu=0.0000, sigma=0.009, phi=-0.18),
        "high_vol": dict(mu=0.0001, sigma=0.028, phi=0.05),
    }[regime]


def generate_asset_path(
    n_days: int,
    seed: int,
    start_price: float = 100.0,
    regime_length: int = 90,
) -> pd.DataFrame:
    """Generate one asset's daily OHLCV path with regime switches.

    Uses an AR(1)-on-returns model per regime segment (phi captures
    momentum when >0 or mean-reversion when <0), stitched into a
    continuous price series.
    """
    rng = np.random.default_rng(seed)
    n_segments = int(np.ceil(n_days / regime_length)) + 1
    regimes = rng.choice(REGIME_LIST, size=n_segments, replace=True)

    returns = np.zeros(n_days)
    last_ret = 0.0
    day = 0
    seg_idx = 0
    while day < n_days:
        params = _regime_params(regimes[seg_idx])
        seg_len = min(regime_length, n_days - day)
        eps = rng.normal(0, params["sigma"], seg_len)
        for i in range(seg_len):
            r = params["mu"] + params["phi"] * last_ret + eps[i]
            returns[day + i] = r
            last_ret = r
        day += seg_len
        seg_idx += 1

    prices = start_price * np.exp(np.cumsum(returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    close = prices
    open_ = close * (1 + rng.normal(0, 0.0015, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
    volume = np.maximum(
        1000, (rng.normal(1_000_000, 250_000, n_days) * (1 + np.abs(returns) * 8)).astype(int)
    )

    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )

    # Inject data quality issues for MarketDataAgent to actually clean.
    n_missing = max(1, n_days // 60)
    n_outliers = max(1, n_days // 80)
    missing_idx = rng.choice(n_days, size=n_missing, replace=False)
    outlier_idx = rng.choice(
        [i for i in range(n_days) if i not in set(missing_idx)], size=n_outliers, replace=False
    )
    df.loc[missing_idx, ["open", "high", "low", "close", "volume"]] = np.nan
    df.loc[outlier_idx, "close"] = df.loc[outlier_idx, "close"] * rng.choice(
        [0.5, 1.8, 2.5], size=n_outliers
    )

    return df


def generate_universe(
    tickers: list[str], n_days: int = 756, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Generate a synthetic multi-asset universe (~3 trading years by default)."""
    out = {}
    for i, t in enumerate(tickers):
        out[t] = generate_asset_path(n_days=n_days, seed=seed + i * 97, start_price=50 + 10 * i)
    return out


def generate_benchmark(n_days: int = 756, seed: int = 7) -> pd.DataFrame:
    """A lower-volatility 'market index' benchmark series for beta/alpha calcs."""
    return generate_asset_path(n_days=n_days, seed=seed, start_price=1000.0, regime_length=140)
