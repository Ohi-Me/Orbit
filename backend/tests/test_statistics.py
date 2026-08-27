"""
Tests for the statistical layer.

These are the tests that matter most in this codebase, because every claim the
platform makes rests on them. They are written as *properties that must hold*
rather than as snapshots of current output: a test asserting "Sharpe == 0.93"
breaks on every data refresh and proves nothing, while a test asserting
"a HAC t-statistic on overlapping returns is smaller than the naive one"
encodes the actual guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.stats import (
    deflated_sharpe_ratio,
    effective_sample_size,
    ic_summary,
    information_coefficient,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    sharpe_of,
    stationary_bootstrap_ci,
)


def _overlapping_returns(n_daily=2000, horizon=21, mu=0.0, seed=0):
    """Build the exact pathology the platform faces: an h-day forward return
    sampled every day, so consecutive observations share h-1 days of path."""
    rng = np.random.default_rng(seed)
    daily = rng.normal(mu, 0.01, n_daily)
    return np.array([daily[i : i + horizon].sum() for i in range(n_daily - horizon)])


class TestNeweyWest:
    def test_hac_shrinks_t_stat_on_overlapping_data(self):
        """The core guarantee: overlap inflates the naive t, HAC corrects it."""
        r = _overlapping_returns(seed=0)
        out = newey_west_tstat(r, lags=20)
        assert abs(out["t_stat"]) < abs(out["naive_t_stat"])
        assert out["inflation_factor"] > 1.5

    def test_no_false_significance_on_zero_edge_data(self):
        """With zero true edge, HAC must not report significance across seeds.

        The naive test fails this badly -- that is the whole point of using it.
        """
        false_positives_hac = 0
        false_positives_naive = 0
        for seed in range(12):
            r = _overlapping_returns(mu=0.0, seed=seed)
            out = newey_west_tstat(r, lags=20)
            if out["p_value"] is not None and out["p_value"] < 0.05:
                false_positives_hac += 1
            if abs(out["naive_t_stat"]) > 1.96:
                false_positives_naive += 1
        assert false_positives_hac <= 2, "HAC should rarely fire on zero-edge data"
        assert false_positives_naive > false_positives_hac

    def test_independent_data_barely_corrected(self):
        """On genuinely independent draws there is little to correct."""
        rng = np.random.default_rng(3)
        r = rng.normal(0.001, 0.01, 800)
        out = newey_west_tstat(r, lags=0)
        assert out["inflation_factor"] == pytest.approx(1.0, abs=0.05)

    def test_too_few_observations_returns_none_not_a_number(self):
        out = newey_west_tstat(np.array([0.01, -0.02, 0.03]))
        assert out["t_stat"] is None
        assert "Too few observations" in out["note"]


class TestDeflatedSharpe:
    def test_more_trials_raises_the_bar(self):
        """The whole purpose: the same Sharpe is less impressive after more tries."""
        one = deflated_sharpe_ratio(1.2, n_obs=36, n_trials=1, periods_per_year=12)
        many = deflated_sharpe_ratio(1.2, n_obs=36, n_trials=20, periods_per_year=12)
        assert many["deflated_sharpe"] < one["deflated_sharpe"]
        assert many["expected_max_sharpe_under_null"] > one["expected_max_sharpe_under_null"]

    def test_units_are_de_annualized(self):
        """Regression test for a real bug.

        Feeding an annualized Sharpe against a monthly observation count
        overstated significance by sqrt(12) and reported near-certainty for a
        coin-flip result. A Sharpe of 1.2 on 36 monthly observations across 12
        trials must NOT be significant.
        """
        d = deflated_sharpe_ratio(1.2, n_obs=36, n_trials=12, periods_per_year=12)
        assert d["significant_at_95"] is False
        assert d["deflated_sharpe"] < 0.95

    def test_strong_result_still_survives(self):
        d = deflated_sharpe_ratio(3.0, n_obs=252, n_trials=12, periods_per_year=252)
        assert d["deflated_sharpe"] > 0.9

    def test_weak_result_fails(self):
        d = deflated_sharpe_ratio(0.2, n_obs=36, n_trials=12, periods_per_year=12)
        assert d["significant_at_95"] is False


class TestProbabilisticSharpe:
    def test_more_observations_increases_confidence(self):
        short = probabilistic_sharpe_ratio(1.0, n_obs=20, periods_per_year=12)
        long = probabilistic_sharpe_ratio(1.0, n_obs=200, periods_per_year=12)
        assert long > short

    def test_negative_skew_reduces_confidence(self):
        """Fat left tails should make the same Sharpe less trustworthy."""
        symmetric = probabilistic_sharpe_ratio(1.0, 100, skew=0.0, periods_per_year=12)
        skewed = probabilistic_sharpe_ratio(1.0, 100, skew=-1.5, excess_kurtosis=4.0, periods_per_year=12)
        assert skewed < symmetric


class TestEffectiveSampleSize:
    def test_overlap_and_correlation_shrink_n(self):
        ess = effective_sample_size(n_obs=2760, n_assets=6, avg_correlation=0.5, overlap=21)
        assert ess["effective_n"] < ess["raw_n"] / 10

    def test_independent_assets_lose_less(self):
        correlated = effective_sample_size(1000, 10, 0.9, 1)["effective_n"]
        independent = effective_sample_size(1000, 10, 0.0, 1)["effective_n"]
        assert independent > correlated


class TestBootstrap:
    def test_interval_brackets_point_estimate(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.001, 0.01, 300)
        out = stationary_bootstrap_ci(r, lambda x: sharpe_of(x, 252), n_boot=300)
        assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]

    def test_zero_edge_interval_spans_zero(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.0, 0.01, 300)
        out = stationary_bootstrap_ci(r, lambda x: sharpe_of(x, 252), n_boot=300)
        assert out["ci_low"] < 0 < out["ci_high"]


class TestInformationCoefficient:
    def test_perfect_ranking_scores_one(self):
        x = np.arange(30, dtype=float)
        assert information_coefficient(x, x) == pytest.approx(1.0)

    def test_inverted_ranking_scores_minus_one(self):
        x = np.arange(30, dtype=float)
        assert information_coefficient(x, -x) == pytest.approx(-1.0)

    def test_constant_signal_is_undefined_not_zero(self):
        """A constant factor cannot rank anything; None is the honest answer."""
        assert information_coefficient(np.ones(30), np.arange(30, dtype=float)) is None

    def test_ic_summary_information_ratio(self):
        s = ic_summary([0.05, 0.04, 0.06, 0.05])
        assert s["mean_ic"] == pytest.approx(0.05, abs=0.01)
        assert s["ic_ir"] > 1
        assert s["hit_rate"] == 1.0
