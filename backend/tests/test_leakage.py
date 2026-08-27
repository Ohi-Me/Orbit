"""
Leakage tests -- the guarantees that make every other number meaningful.

If these fail, nothing else in the platform is trustworthy, so they are
written as adversarial checks: each one constructs the leak it is meant to
detect and asserts that the system catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.critic import check_lookahead_empirical, check_validation_scheme
from app.agents.ml_research import LABEL_HORIZON, purged_walk_forward_folds
from app.agents.quant_research import build_factor_panel, select_usable_factors


@pytest.fixture
def price_frames():
    """Two assets, 600 business days of random-walk prices."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=600)
    frames = {}
    for i, t in enumerate(["AAA", "BBB"]):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 600)))
        frames[t] = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, 600),
            }
        )
    return frames


class TestPurgedWalkForward:
    def test_train_never_overlaps_test(self, price_frames):
        folds = purged_walk_forward_folds(
            pd.Series(price_frames["AAA"]["date"]), n_folds=4, horizon=21, embargo=5
        )
        assert folds
        for f in folds:
            assert f["train_end"] < f["test_start"]

    def test_purge_gap_covers_the_label_horizon(self, price_frames):
        """A training sample's 21-day label must resolve BEFORE the test window.

        Without this gap the model is trained on the outcome it is then scored
        on -- the single most damaging and most common time-series ML mistake.
        """
        dates = pd.DatetimeIndex(sorted(price_frames["AAA"]["date"].unique()))
        folds = purged_walk_forward_folds(pd.Series(dates), n_folds=4, horizon=21, embargo=5)
        for f in folds:
            train_end_idx = dates.get_loc(f["train_end"])
            test_start_idx = dates.get_loc(f["test_start"])
            gap = test_start_idx - train_end_idx
            assert gap >= LABEL_HORIZON, (
                f"gap of {gap} trading days does not cover the {LABEL_HORIZON}-day "
                "label horizon; labels leak across the fold boundary"
            )

    def test_folds_move_forward_in_time(self, price_frames):
        folds = purged_walk_forward_folds(pd.Series(price_frames["AAA"]["date"]), n_folds=4)
        for a, b in zip(folds, folds[1:]):
            assert b["test_start"] > a["test_start"]

    def test_too_little_history_yields_no_folds(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        assert purged_walk_forward_folds(pd.Series(dates), n_folds=4) == []


class TestFactorPanel:
    def test_label_is_a_future_shift(self, price_frames):
        """fwd_return_21d at t must equal the realized return from t to t+21."""
        panel = build_factor_panel(price_frames)
        sub = panel[panel.ticker == "AAA"].reset_index(drop=True)
        close = price_frames["AAA"]["close"].to_numpy()
        expected = close[21] / close[0] - 1
        assert sub["fwd_return_21d"].iloc[0] == pytest.approx(expected, rel=1e-9)

    def test_label_is_nan_at_the_end(self, price_frames):
        """The last 21 rows cannot have a forward return -- they must be NaN,
        not zero, or the backtest silently trades a fabricated final period."""
        panel = build_factor_panel(price_frames)
        sub = panel[panel.ticker == "AAA"]
        assert sub["fwd_return_21d"].tail(21).isna().all()

    def test_accounting_columns_are_not_factors(self):
        """ret_1d is same-day information and must never be a candidate factor."""
        from app.agents.quant_research import ALL_FACTORS

        assert "ret_1d" not in ALL_FACTORS
        assert "close_px" not in ALL_FACTORS
        assert "fwd_return_21d" not in ALL_FACTORS

    def test_momentum_uses_only_past_data(self, price_frames):
        """Truncating the future must not change a past factor value."""
        panel_full = build_factor_panel(price_frames)
        truncated = {t: f.iloc[:400].copy() for t, f in price_frames.items()}
        panel_trunc = build_factor_panel(truncated)

        a = panel_full[(panel_full.ticker == "AAA")].set_index("date")["momentum_12_1"]
        b = panel_trunc[(panel_trunc.ticker == "AAA")].set_index("date")["momentum_12_1"]
        common = a.index.intersection(b.index)[-50:]
        pd.testing.assert_series_equal(a.loc[common], b.loc[common], check_names=False)


class TestConstantFactorDetection:
    def test_constant_factor_is_rejected(self, price_frames):
        """The bug the original build shipped: today's sentiment written across
        every historical row. It is a constant per ticker, and a constant
        derived from the present embeds future knowledge in the past."""
        panel = build_factor_panel(price_frames)
        panel["sentiment_score"] = panel["ticker"].map({"AAA": 0.7, "BBB": -0.3})
        selected, diagnostics = select_usable_factors(panel, factors=["momentum_3m", "sentiment_score"])
        assert "sentiment_score" not in selected
        assert "constant" in diagnostics["rejected"]["sentiment_score"]

    def test_only_requested_factors_are_reported_as_rejected(self, price_frames):
        """A factor that was never a candidate must not appear as 'rejected'.

        Reporting 0% coverage for a column nobody asked for states a measured
        finding about something that was never measured.
        """
        panel = build_factor_panel(price_frames)
        _, diagnostics = select_usable_factors(panel, factors=["momentum_3m"])
        assert set(diagnostics["rejected"]) <= {"momentum_3m"}
        assert "earnings_yield" not in diagnostics["rejected"]


class TestCriticLeakageChecks:
    def test_critic_catches_a_leaking_factor(self, price_frames):
        """Inject the label into a feature; the empirical check must catch it.

        The original name-matching check would have passed this, because the
        column is innocently named.
        """
        panel = build_factor_panel(price_frames)
        panel["momentum_3m"] = panel["fwd_return_21d"] * 0.9  # blatant leak
        result = check_lookahead_empirical(panel, ["momentum_3m"])
        assert result["passed"] is False
        assert result["severity"] == "error"

    def test_critic_passes_clean_factors(self, price_frames):
        panel = build_factor_panel(price_frames)
        result = check_lookahead_empirical(panel, ["momentum_12_1", "momentum_3m"])
        assert result["passed"] is True

    def test_critic_rejects_insufficient_purge(self):
        bad = {
            "status": "ok",
            "validation_scheme": {"purge_days": 0, "label_horizon_days": 21, "embargo_days": 0},
        }
        assert check_validation_scheme(bad)["passed"] is False

    def test_critic_accepts_adequate_purge(self):
        good = {
            "status": "ok",
            "validation_scheme": {"purge_days": 21, "label_horizon_days": 21, "embargo_days": 5},
        }
        assert check_validation_scheme(good)["passed"] is True
