"""
Agent-level tests: backtest mechanics, risk, portfolio, planner, validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.backtest import _form_weights, _transaction_cost, build_returns_matrix, run_backtest
from app.agents.critic import check_significance, check_turnover_and_costs, run_critic_agent
from app.agents.planner import ResearchPlan, revise_plan, rule_based_plan
from app.agents.portfolio import ledoit_wolf_covariance, risk_parity, walk_forward_allocation
from app.agents.risk import compute_beta_alpha, cornish_fisher_var, cvar, historical_var
from app.data.validation import validate_price_frame, validate_universe


# ---------------------------------------------------------------------------
# Position construction and costs
# ---------------------------------------------------------------------------
class TestWeights:
    def test_dollar_neutral_and_gross_normalized(self):
        scores = pd.Series({"A": 5.0, "B": 4.0, "C": 0.0, "D": -4.0, "E": -5.0, "F": -6.0})
        w = _form_weights(scores, top_frac=0.33)
        assert sum(w.values()) == pytest.approx(0.0, abs=1e-9), "book must be dollar neutral"
        assert sum(abs(v) for v in w.values()) == pytest.approx(1.0), "gross must normalize to 1"

    def test_longs_are_the_highest_scores(self):
        scores = pd.Series({"A": 9.0, "B": 1.0, "C": -9.0})
        w = _form_weights(scores, top_frac=0.34)
        assert w["A"] > 0 and w["C"] < 0

    def test_universe_too_small_returns_nothing(self):
        """Rather than fabricating a one-name 'strategy'."""
        assert _form_weights(pd.Series({"A": 1.0}), top_frac=0.3) == {}


class TestTransactionCosts:
    def test_no_trade_costs_nothing(self):
        w = {"A": 0.5, "B": -0.5}
        cost, _ = _transaction_cost(w, w, {}, {})
        assert cost == 0.0

    def test_cost_rises_with_turnover(self):
        vol = {"A": 0.25, "B": 0.25}
        adv = {"A": 5e7, "B": 5e7}
        small, _ = _transaction_cost({"A": 0.5, "B": -0.5}, {"A": 0.45, "B": -0.55}, vol, adv)
        large, _ = _transaction_cost({"A": 0.5, "B": -0.5}, {"A": -0.5, "B": 0.5}, vol, adv)
        assert large > small

    def test_impact_rises_as_liquidity_falls(self):
        """The square-root law: the same trade costs more in a thinner name."""
        prev, new = {"A": 0.0}, {"A": 1.0}
        vol = {"A": 0.3}
        liquid, _ = _transaction_cost(prev, new, vol, {"A": 1e9})
        thin, _ = _transaction_cost(prev, new, vol, {"A": 1e6})
        assert thin > liquid

    def test_impact_rises_with_aum(self):
        prev, new = {"A": 0.0}, {"A": 1.0}
        vol, adv = {"A": 0.3}, {"A": 1e7}
        small, _ = _transaction_cost(prev, new, vol, adv, aum=1e6)
        big, _ = _transaction_cost(prev, new, vol, adv, aum=1e9)
        assert big > small


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@pytest.fixture
def market():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-03", periods=400)
    tickers = ["A", "B", "C", "D", "E", "F"]
    frames, rets = {}, {}
    for i, t in enumerate(tickers):
        r = rng.normal(0.0004, 0.013, 400)
        close = 100 * np.exp(np.cumsum(r))
        frames[t] = pd.DataFrame(
            {"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": rng.integers(2_000_000, 8_000_000, 400)}
        )
    returns = build_returns_matrix(frames)
    panel = pd.DataFrame(
        [
            {"date": d, "ticker": t, "volatility_20d": 0.25,
             "liquidity_log_dollar_vol": np.log(5e7)}
            for d in dates for t in tickers
        ]
    )
    signal = pd.DataFrame(
        [
            {"date": d, "ticker": t, "score": rng.random()}
            for d in dates[::5] for t in tickers
        ]
    )
    return frames, returns, panel, signal


class TestBacktest:
    def test_refuses_to_invent_a_signal(self, market):
        _, returns, panel, _ = market
        out = run_backtest(panel, returns, signal=None)
        assert out["status"] == "no_signal"

    def test_produces_daily_observations_not_monthly(self, market):
        """The original compounded ~36 period returns over 3 years; marking to
        market daily is what makes the statistics usable at all."""
        _, returns, panel, signal = market
        out = run_backtest(panel, returns, signal=signal)
        assert out["status"] == "ok"
        assert out["n_trading_days"] > 200

    def test_costs_reduce_net_performance(self, market):
        _, returns, panel, signal = market
        out = run_backtest(panel, returns, signal=signal)
        m = out["metrics"]
        assert m["sharpe_gross_of_costs"] >= m["sharpe_ratio"]

    def test_reports_corrected_and_naive_significance(self, market):
        _, returns, panel, signal = market
        out = run_backtest(panel, returns, signal=signal)
        sig = out["significance"]
        assert sig["t_stat_hac"] is not None
        assert sig["t_stat_naive"] is not None

    def test_equity_curve_matches_returns(self, market):
        _, returns, panel, signal = market
        out = run_backtest(panel, returns, signal=signal)
        r = np.array(out["daily_returns"])
        expected_final = float(np.prod(1 + r))
        assert out["equity_curve"][-1]["equity"] == pytest.approx(expected_final, rel=1e-4)

    def test_max_drawdown_is_non_positive(self, market):
        _, returns, panel, signal = market
        out = run_backtest(panel, returns, signal=signal)
        assert out["metrics"]["max_drawdown"] <= 0


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
class TestRisk:
    def test_var_increases_with_confidence(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.02, 1000)
        assert historical_var(r, 0.99) > historical_var(r, 0.95)

    def test_cvar_exceeds_var(self):
        """CVaR is the mean of the tail beyond VaR, so it must be worse."""
        rng = np.random.default_rng(2)
        r = rng.normal(0, 0.02, 1000)
        assert cvar(r, 0.95) >= historical_var(r, 0.95)

    def test_cornish_fisher_penalizes_left_skew(self):
        rng = np.random.default_rng(3)
        symmetric = rng.normal(0, 0.02, 3000)
        skewed = np.concatenate([rng.normal(0.002, 0.01, 2900), rng.normal(-0.12, 0.02, 100)])
        assert cornish_fisher_var(skewed, 0.99) > cornish_fisher_var(symmetric, 0.99)

    def test_beta_aligns_on_dates_not_position(self):
        """The original subsampled the benchmark to match a length, comparing
        different periods. Misaligned dates must reduce the overlap, not
        silently produce a number."""
        idx = pd.bdate_range("2023-01-02", periods=200)
        rng = np.random.default_rng(4)
        bench = pd.Series(rng.normal(0, 0.01, 200), index=idx)
        strat = pd.Series(bench.to_numpy() * 1.5, index=idx)
        out = compute_beta_alpha(strat, bench)
        assert out["beta"] == pytest.approx(1.5, abs=0.05)
        assert out["n_aligned_observations"] == 200

        offset = pd.Series(bench.to_numpy(), index=idx.shift(400, freq="D"))
        assert compute_beta_alpha(strat, offset)["beta"] is None

    def test_perfect_hedge_has_zero_beta(self):
        idx = pd.bdate_range("2023-01-02", periods=150)
        rng = np.random.default_rng(6)
        bench = pd.Series(rng.normal(0, 0.01, 150), index=idx)
        flat = pd.Series(np.full(150, 0.0001), index=idx)
        assert compute_beta_alpha(flat, bench)["beta"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class TestPortfolio:
    def test_risk_parity_equalizes_risk_contributions(self):
        cov = np.diag([0.04, 0.01, 0.09])
        w = risk_parity(cov, max_weight=1.0)
        contrib = w * (cov @ w)
        assert contrib.max() / contrib.min() < 1.35

    def test_weights_respect_the_cap(self):
        rng = np.random.default_rng(8)
        data = rng.normal(0, 0.01, (300, 4))
        cov, _ = ledoit_wolf_covariance(data)
        w = risk_parity(cov, max_weight=0.3)
        assert w.max() <= 0.3 + 1e-6

    def test_shrinkage_is_reported(self):
        rng = np.random.default_rng(9)
        _, shrink = ledoit_wolf_covariance(rng.normal(0, 0.01, (60, 8)))
        assert 0.0 <= shrink <= 1.0

    def test_walk_forward_needs_enough_history(self):
        idx = pd.bdate_range("2024-01-01", periods=100)
        rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (100, 3)), index=idx,
                            columns=["A", "B", "C"])
        out = walk_forward_allocation(rets, "risk_parity", lookback=252)
        assert out["status"] == "insufficient_data"

    def test_walk_forward_produces_out_of_sample_results(self):
        idx = pd.bdate_range("2021-01-01", periods=700)
        rets = pd.DataFrame(np.random.default_rng(1).normal(0.0003, 0.011, (700, 4)), index=idx,
                            columns=["A", "B", "C", "D"])
        out = walk_forward_allocation(rets, "min_variance", lookback=252)
        assert out["status"] == "ok"
        assert out["out_of_sample"]["n_days"] > 300


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class TestPlanner:
    def test_defaults_to_a_wide_enough_universe(self):
        """A cross-sectional strategy on six names is a bet on six companies."""
        plan = rule_based_plan("Do factors work?")
        assert len(plan.universe) >= 15

    def test_detects_a_sector_from_the_question(self):
        assert "JPM" in rule_based_plan("Do value factors work in banks?").universe

    def test_detects_factor_families(self):
        plan = rule_based_plan("Does momentum still work after costs?")
        assert "momentum" in plan.factor_families

    def test_schema_rejects_an_out_of_bounds_plan(self):
        with pytest.raises(Exception):
            ResearchPlan(question="q", universe=["A", "B"], n_days=99, factor_families=["momentum"])

    def test_schema_rejects_unknown_factor_family(self):
        with pytest.raises(Exception):
            ResearchPlan(question="q", universe=["A", "B"], factor_families=["astrology"])

    def test_schema_rejects_single_ticker_universe(self):
        with pytest.raises(Exception):
            ResearchPlan(question="q", universe=["AAPL"], factor_families=["momentum"])

    def test_revision_widens_a_thin_universe(self):
        plan = ResearchPlan(question="q", universe=["A", "B", "C"], factor_families=["momentum"])
        revised, applied = revise_plan(plan.model_dump(), ["widen_universe"])
        assert len(revised["universe"]) > 3
        assert applied

    def test_revision_drops_leaking_sentiment(self):
        plan = ResearchPlan(question="q", universe=["A", "B"], factor_families=["momentum", "sentiment"])
        revised, _ = revise_plan(plan.model_dump(), ["drop_leaking_factors"])
        assert "sentiment" not in revised["factor_families"]


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------
def _frame(close, volume=None):
    n = len(close)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=n),
            "open": close, "high": np.array(close) * 1.01, "low": np.array(close) * 0.99,
            "close": close, "volume": volume if volume is not None else np.full(n, 3_000_000),
        }
    )


class TestValidation:
    def test_flags_an_unadjusted_split(self):
        close = list(np.linspace(100, 110, 200))
        close[150] = close[149] / 2  # 50% overnight drop
        findings = validate_price_frame("X", _frame(close))
        assert any(f["check"] == "possible_unadjusted_split" for f in findings)

    def test_flags_stale_prices(self):
        close = list(np.linspace(100, 120, 200))
        for i in range(100, 112):
            close[i] = close[99]
        findings = validate_price_frame("X", _frame(close))
        assert any(f["check"] == "stale_prices" for f in findings)

    def test_flags_thin_liquidity(self):
        close = list(np.linspace(10, 12, 200))
        findings = validate_price_frame("X", _frame(close, volume=np.full(200, 500)))
        assert any(f["check"] == "thin_liquidity" for f in findings)

    def test_rejects_too_little_history(self):
        findings = validate_price_frame("X", _frame(list(np.linspace(100, 101, 30))))
        assert any(f["check"] == "insufficient_history" and f["severity"] == "error" for f in findings)

    def test_drops_broken_assets_from_the_universe(self):
        good = _frame(list(np.linspace(100, 120, 300)))
        bad = _frame(list(np.linspace(100, 101, 20)))
        result = validate_universe({"GOOD": good, "BAD": bad})
        assert result["usable_tickers"] == ["GOOD"]
        assert "BAD" in result["dropped_tickers"]

    def test_clean_data_passes(self):
        rng = np.random.default_rng(12)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 400)))
        findings = validate_price_frame("X", _frame(list(close)))
        assert not [f for f in findings if f["severity"] == "error"]


# ---------------------------------------------------------------------------
# Critic verdicts
# ---------------------------------------------------------------------------
class TestCriticVerdicts:
    def test_flags_cost_destroying_the_edge(self):
        bt = {
            "status": "ok",
            "costs": {"avg_turnover_per_rebalance": 1.0, "total_impact_cost": 0.1, "total_spread_cost": 0.01},
            "metrics": {"sharpe_ratio": -0.4, "sharpe_gross_of_costs": 0.8},
        }
        out = check_turnover_and_costs(bt)
        assert out["passed"] is False
        assert "before costs and unprofitable after" in out["detail"]

    def test_flags_insignificant_result(self):
        bt = {"status": "ok", "significance": {"p_value_hac": 0.42, "t_stat_hac": 0.8,
                                               "t_stat_naive": 3.1, "t_inflation_factor": 3.9}}
        assert check_significance(bt)["passed"] is False

    def test_verdict_is_reject_when_errors_exist(self):
        """A small universe is an error-level finding and must force a rejection."""
        panel = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10).repeat(2),
                              "ticker": ["A", "B"] * 10, "fwd_return_21d": np.random.default_rng(0).normal(0, 0.05, 20),
                              "momentum_3m": np.random.default_rng(1).normal(0, 1, 20)})
        out = run_critic_agent(panel, ["momentum_3m"], {"status": "insufficient_data"}, {}, {"is_synthetic": False})
        assert out["overall_verdict"] == "REJECT_INSUFFICIENT_EVIDENCE"
        assert out["recommended_action"] == "revise_and_rerun"
        assert "widen_universe" in out["recommended_revisions"]

    def test_synthetic_data_is_an_error_level_finding(self):
        out = run_critic_agent(None, [], {}, {}, {"is_synthetic": True})
        data_check = next(c for c in out["checks"] if c["check"] == "data_quality")
        assert data_check["passed"] is False
        assert data_check["severity"] == "error"
