"""
Research Report Agent
=====================
Assembles a markdown research report from the other agents' outputs.

DELIBERATELY TEMPLATE-DRIVEN. Every number is read from an upstream agent's
structured output; no LLM is asked to "write up the results". This is the one
place where the temptation to use a language model is strongest and the cost
of doing so is highest -- an LLM writing a research summary will round, infer,
and smooth numbers, and a report that states a figure no agent computed is
precisely the failure the Critic exists to catch elsewhere.

The report leads with the LIMITATIONS and the Critic's verdict rather than
burying them, because a research document whose caveats are in an appendix is
a marketing document.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _pct(x) -> str:
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "n/a"


def _num(x, nd=3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def _status_icon(passed) -> str:
    return "PASS" if passed is True else "FLAG" if passed is False else "SKIP"


def build_report(state: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    plan = state.get("plan") or {}
    md = state.get("market_data") or {}
    critic = state.get("critic_result") or {}
    ml = state.get("ml_result") or {}
    comparison = state.get("backtest_comparison") or {}
    risk = state.get("risk_result") or {}
    portfolio = state.get("portfolio_result") or {}
    diagnostics = state.get("factor_diagnostics") or {}
    ic = state.get("factor_ic") or {}
    fm = state.get("fama_macbeth") or {}

    L: list[str] = []
    A = L.append

    # ---------------------------------------------------------------- header
    A("# Quantitative Research Report")
    A("")
    A(f"**Question:** {state.get('question', 'n/a')}")
    A("")
    A(f"_Generated {ts} · run `{state.get('run_id', 'n/a')}`_")
    A("")

    # ------------------------------------------------------- headline verdict
    A("## Verdict")
    A("")
    verdict = critic.get("overall_verdict", "NOT_EVALUATED")
    A(f"**{verdict}** — {critic.get('n_errors', 0)} error-level and "
      f"{critic.get('n_warnings', 0)} warning-level checks failed of "
      f"{critic.get('n_checks_run', 0)} run.")
    A("")
    A(f"> {critic.get('verdict_note', '')}")
    A("")

    ml_verdict = ml.get("model_verdict")
    if ml_verdict:
        A(f"**Model finding:** `{ml_verdict}` — {ml.get('model_verdict_detail', '')}")
        A("")

    if comparison.get("status") == "ok":
        best = comparison.get("best_strategy")
        A(f"**Best strategy:** `{best}` (Sharpe {_num(comparison.get('best_sharpe'))}). "
          f"Hand-weighted baseline Sharpe: {_num(comparison.get('baseline_sharpe'))}. "
          f"Model beats baseline: **{comparison.get('model_beats_baseline')}**.")
        A("")

    # -------------------------------------------------------- data provenance
    A("## 1. Data and provenance")
    A("")
    prov = (md.get("provenance") or {})
    is_synth = md.get("is_synthetic")
    if is_synth:
        A("> **THIS RUN USED SYNTHETIC PRICE DATA.** Every performance figure below "
          "describes a random-process generator, not a market. The methodology is "
          "demonstrable; the results carry no information about real assets.")
        A("")
    A(f"- Price provider: `{prov.get('provider')}` ({'SYNTHETIC' if is_synth else 'live'})")
    A(f"- Price basis: {prov.get('price_basis', 'n/a')}")
    A(f"- Window: {prov.get('start_date')} to {prov.get('end_date')} "
      f"({prov.get('n_rows', 0)} asset-days)")
    if prov.get("fallback_reason"):
        A(f"- **Fell back to synthetic because:** {prov['fallback_reason']}")

    fund_prov = (state.get("provenance") or {}).get("fundamentals")
    if fund_prov:
        A(f"- Fundamentals: `{fund_prov.get('provider')}`, point-in-time="
          f"{fund_prov.get('point_in_time')}, {fund_prov.get('n_tickers')} tickers with XBRL facts")
    macro_prov = (state.get("provenance") or {}).get("macro")
    if macro_prov:
        A(f"- Macro: `{macro_prov.get('provider')}`, {len(macro_prov.get('series', []))} series, "
          f"publication lags applied")
    news_prov = (state.get("provenance") or {}).get("news")
    if news_prov:
        A(f"- News: `{news_prov.get('provider')}`, {news_prov.get('n_items', 0)} items spanning "
          f"{news_prov.get('coverage_days', 0)} days")

    validation = md.get("validation") or {}
    A(f"- Validation: {validation.get('summary', 'n/a')}")
    if validation.get("dropped_tickers"):
        A(f"- **Assets dropped for data errors:** {validation['dropped_tickers']}")
    A("")

    # ----------------------------------------------------------- factor layer
    A("## 2. Factor construction and diagnostics")
    A("")
    selected = state.get("selected_factors") or []
    A(f"Factors selected: {len(selected)} — `{'`, `'.join(selected)}`")
    A("")
    rejected = diagnostics.get("rejected") or {}
    if rejected:
        A("**Factors rejected, and why:**")
        A("")
        for f, reason in rejected.items():
            A(f"- `{f}` — {reason}")
        A("")

    if ic:
        A("### Information coefficient by factor")
        A("")
        A("| Factor | Mean IC | IC IR | Hit rate | Periods |")
        A("|---|---|---|---|---|")
        for f, s in sorted(ic.items(), key=lambda kv: -(abs(kv[1].get("mean_ic") or 0))):
            A(f"| `{f}` | {_num(s.get('mean_ic'), 4)} | {_num(s.get('ic_ir'))} | "
              f"{_num(s.get('hit_rate'))} | {s.get('n_periods', 0)} |")
        A("")
        A("_IC is the per-date rank correlation between a factor and the forward "
          "return it is meant to predict. A mean IC around 0.03 with a positive IR "
          "is the conventional threshold for a usable equity factor._")
        A("")

    if fm.get("status") == "ok":
        A("### Fama-MacBeth factor premia")
        A("")
        A(f"_Mode: **{fm.get('mode')}** — {fm.get('caveat', '')}_")
        A("")
        A("| Factor | Premium | HAC t | Naive t | Inflation | p | Significant |")
        A("|---|---|---|---|---|---|---|")
        for f, r in sorted(
            fm.get("per_factor", {}).items(), key=lambda kv: -(abs(kv[1].get("t_stat") or 0))
        ):
            if r.get("status") != "ok":
                continue
            A(f"| `{f}` | {_num(r.get('mean_premium'), 5)} | {_num(r.get('t_stat'))} | "
              f"{_num(r.get('naive_t_stat'))} | {_num(r.get('t_inflation_vs_naive'), 2)}x | "
              f"{_num(r.get('p_value'), 4)} | {r.get('significant_at_95')} |")
        A("")
        A("_The naive column is shown deliberately. Overlapping forward-return "
          "windows inflate an uncorrected t-statistic; the HAC column is the one "
          "to read, and the gap between them is why._")
        A("")

    # ------------------------------------------------------------ ML section
    A("## 3. Model comparison (purged walk-forward)")
    A("")
    if ml.get("status") == "ok":
        scheme = ml.get("validation_scheme", {})
        A(f"Validation: **{scheme.get('type')}** — purge {scheme.get('purge_days')}d, "
          f"embargo {scheme.get('embargo_days')}d, label horizon "
          f"{scheme.get('label_horizon_days')}d.")
        A("")
        A(f"> {scheme.get('note', '')}")
        A("")
        A("| Model | Accuracy | Base rate | Lift | AUC | Mean IC | Signal return | Degenerate folds |")
        A("|---|---|---|---|---|---|---|---|")
        for s in ml.get("summary", []):
            if s.get("status") != "ok":
                A(f"| {s.get('model')} | _{s.get('status')}_ | | | | | | |")
                continue
            A(f"| `{s['model']}` | {_num(s.get('mean_accuracy'), 4)} | "
              f"{_num(s.get('mean_base_rate'), 4)} | {_num(s.get('mean_accuracy_lift'), 4)} | "
              f"{_num(s.get('mean_auc'), 4)} | {_num(s.get('mean_information_coefficient'), 4)} | "
              f"{_num(s.get('mean_signal_return'), 5)} | {s.get('n_degenerate_folds', 0)} |")
        A("")
        A("_**Base rate** is the share of the majority direction. Accuracy below the "
          "base rate means the model is worse than always predicting the same way; "
          "**lift** is the only column in which a directional model can claim skill. "
          "A **degenerate fold** is one where the model predicted a single class for "
          ">95% of samples._")
        A("")
        ess = ml.get("effective_sample_size", {})
        if ess:
            A(f"**Effective sample size:** {ess.get('effective_n')} independent "
              f"observations from {ess.get('raw_n')} raw rows "
              f"(overlap {ess.get('overlap_days')}d, average cross-correlation "
              f"{ess.get('avg_cross_correlation')}). Significance must be read against "
              "the effective figure.")
            A("")
    else:
        A(f"_Skipped: {ml.get('note', ml.get('status'))}_")
        A("")

    # ------------------------------------------------------------- backtests
    A("## 4. Backtest — strategies traded on out-of-sample model scores")
    A("")
    if comparison.get("status") == "ok":
        A("| Strategy | CAGR | Sharpe (net) | Sharpe (gross) | Max DD | Turnover | HAC t | p | Deflated Sharpe |")
        A("|---|---|---|---|---|---|---|---|---|")
        for name, r in comparison.get("strategies", {}).items():
            if r.get("status") != "ok":
                A(f"| `{name}` | _{r.get('status')}_ | | | | | | | |")
                continue
            m, sig, c = r["metrics"], r["significance"], r["costs"]
            dsr = sig.get("deflated_sharpe", {}) or {}
            A(f"| `{name}` | {_pct(m.get('cagr'))} | {_num(m.get('sharpe_ratio'))} | "
              f"{_num(m.get('sharpe_gross_of_costs'))} | {_pct(m.get('max_drawdown'))} | "
              f"{_num(c.get('avg_turnover_per_rebalance'), 2)}x | {_num(sig.get('t_stat_hac'))} | "
              f"{_num(sig.get('p_value_hac'), 4)} | {_num(dsr.get('deflated_sharpe'), 4)} |")
        A("")
        A(f"_{comparison.get('multiple_testing_note', '')}_")
        A("")
        best = comparison.get("best_strategy")
        bt = comparison.get("strategies", {}).get(best, {})
        if bt.get("status") == "ok":
            costs = bt["costs"]
            A(f"**Cost model for `{best}`:** {costs.get('model')}")
            A("")
            A(f"- Spread cost: {_num(costs.get('total_spread_cost'), 5)} · "
              f"Impact cost: {_num(costs.get('total_impact_cost'), 5)} · "
              f"Total drag: {_num(costs.get('total_cost_drag_on_return'), 5)}")
            A(f"- Assumed AUM: ${costs.get('assumed_aum', 0):,.0f} — impact scales with "
              "size, so this strategy's costs rise with capital deployed.")
            boot = bt["significance"].get("sharpe_bootstrap_ci", {})
            if boot.get("ci_low") is not None:
                A(f"- Sharpe 95% bootstrap CI: [{_num(boot['ci_low'])}, {_num(boot['ci_high'])}] "
                  f"({boot.get('n_boot')} stationary-block resamples)")
            A("")
    else:
        A(f"_No backtest: {comparison.get('status')}_")
        A("")

    # ------------------------------------------------------------------ risk
    A("## 5. Risk")
    A("")
    if risk.get("status") == "ok":
        var = risk.get("value_at_risk_daily", {})
        A(f"- Daily VaR 95%: historical {_pct(var.get('historical_95'))}, "
          f"Cornish-Fisher {_pct(var.get('cornish_fisher_95'))}")
        A(f"- Daily VaR 99%: historical {_pct(var.get('historical_99'))}, "
          f"Cornish-Fisher {_pct(var.get('cornish_fisher_99'))}")
        A(f"- CVaR 95%: {_pct(risk.get('conditional_var_95'))} · "
          f"CVaR 99%: {_pct(risk.get('conditional_var_99'))}")
        A(f"- {risk.get('var_interpretation', '')}")
        A("")

        br = risk.get("benchmark_relative") or {}
        if br.get("beta") is not None:
            A(f"- Beta {br['beta']} · annualized alpha {_pct(br.get('alpha_annualized'))} · "
              f"IR {_num(br.get('information_ratio'))} · R² {_num(br.get('r_squared_vs_benchmark'))} "
              f"({br.get('n_aligned_observations')} aligned days)")
            A("")

        fr = risk.get("factor_risk") or {}
        if fr.get("status") == "ok":
            A(f"**Factor risk decomposition:** {_pct(fr.get('systematic_variance_share'))} of "
              f"variance is explained by factor exposures; "
              f"{_pct(fr.get('idiosyncratic_variance_share'))} is idiosyncratic "
              f"(annualized idiosyncratic vol {_pct(fr.get('annualized_idiosyncratic_vol'))}).")
            A("")
            A(f"Dominant risk factors: `{'`, `'.join(fr.get('dominant_risk_factors', []))}`")
            A("")

        conc = risk.get("concentration") or {}
        if conc.get("status") == "ok":
            A(f"**Concentration:** {conc['n_positions']} positions, largest "
              f"{_pct(conc['largest_position'])}, HHI {_num(conc['herfindahl_index'])} "
              f"(effective N = {_num(conc.get('effective_n_positions'), 1)}). "
              f"Worst days-to-liquidate: {conc.get('worst_days_to_liquidate')}.")
            if conc.get("concentration_flag"):
                A("")
                A("> Concentration flag raised: the book is dominated by a small number "
                  "of positions, so idiosyncratic risk exceeds factor risk.")
            A("")

        sc = risk.get("scenarios", {})
        hist = sc.get("historical_scenarios", {})
        if hist:
            A("**Historical scenario replay:**")
            A("")
            A("| Scenario | Benchmark | Estimated strategy return |")
            A("|---|---|---|")
            for name, s in hist.items():
                A(f"| {s.get('description', name)} | {_pct(s.get('benchmark_return'))} | "
                  f"{_pct(s.get('estimated_strategy_return'))} |")
            A("")
            A(f"_{sc.get('method', '')}_")
            A("")
    else:
        A("_Skipped: insufficient data._")
        A("")

    # ------------------------------------------------------------- portfolio
    A("## 6. Portfolio construction (walk-forward)")
    A("")
    if portfolio.get("status") == "ok":
        A("| Method | OOS Sharpe | OOS return | OOS vol | Max DD | Turnover |")
        A("|---|---|---|---|---|---|")
        for name, r in portfolio.get("walk_forward", {}).items():
            if r.get("status") != "ok":
                A(f"| `{name}` | _{r.get('status')}_ | | | | |")
                continue
            o = r["out_of_sample"]
            A(f"| `{name}` | {_num(o.get('sharpe'))} | {_pct(o.get('annualized_return'))} | "
              f"{_pct(o.get('annualized_volatility'))} | {_pct(o.get('max_drawdown'))} | "
              f"{_num(r.get('avg_turnover_per_rebalance'), 3)} |")
        A("")
        if portfolio.get("in_sample_optimism") is not None:
            A(f"**In-sample optimism: {_num(portfolio['in_sample_optimism'])} Sharpe.** "
              f"{portfolio.get('optimism_note', '')}")
            A("")
        A(f"_Covariance: {portfolio.get('covariance_estimator', '')}_")
        A("")
    else:
        A("_Skipped._")
        A("")

    # ---------------------------------------------------------------- critic
    A("## 7. Independent critique")
    A("")
    for c in critic.get("checks", []):
        A(f"**[{_status_icon(c['passed'])}] {c['check']}** _(severity: {c.get('severity', 'n/a')})_")
        detail = c.get("detail")
        if isinstance(detail, list):
            for d in detail:
                A(f"  - {d}")
        else:
            A(f"  - {detail}")
        A("")

    if critic.get("recommended_revisions"):
        A(f"**Recommended revisions:** `{'`, `'.join(critic['recommended_revisions'])}`")
        A("")

    revisions = state.get("revisions_applied") or []
    if revisions:
        A("### Revisions applied during this run")
        A("")
        for r in revisions:
            A(f"- {r}")
        A("")

    # ------------------------------------------------------------ documents
    docs = state.get("documents") or {}
    if docs.get("status") == "ok" and docs.get("answer"):
        ans = docs["answer"]
        A("## 8. Document evidence (RAG)")
        A("")
        A(f"**Query:** {ans.get('query')}")
        A("")
        A(f"Retrieval: {ans.get('retrieval_mode')} over {ans.get('n_candidates_considered')} chunks. "
          f"Verification verdict: **{ans.get('verification', {}).get('verdict')}**")
        A("")
        gen = ans.get("generation", {})
        if gen.get("answer"):
            A(gen["answer"])
            A("")
            ver = ans.get("verification", {})
            ng = ver.get("numeric_grounding") or {}
            A(f"_Numeric grounding: {ng.get('n_grounded')}/{ng.get('n_numbers_in_answer')} "
              f"figures traced to source text._")
            if ng.get("ungrounded_values"):
                A("")
                A(f"> **Ungrounded figures flagged:** {ng['ungrounded_values']}")
            A("")
        A("**Sources:**")
        A("")
        for s in ans.get("sources", [])[:6]:
            A(f"- `{s['chunk_id']}` — {s['ticker']} {s['doc_type']} filed {s['filing_date']}"
              + (f", {s['section']}" if s.get("section") else ""))
        A("")

    # --------------------------------------------------------- how it's wrong
    A("## How this research could be wrong")
    A("")
    limitations = []
    if is_synth:
        limitations.append(
            "**The price data is synthetic.** Nothing below is evidence about real markets."
        )
    n_names = len(md.get("tickers") or [])
    if n_names and n_names < 15:
        limitations.append(
            f"**The universe has {n_names} names.** A cross-sectional long/short book on "
            "this few names holds one or two positions per leg, so idiosyncratic risk "
            "dominates any factor signal and the tercile construction is nominal."
        )
    if fm.get("mode") == "univariate":
        limitations.append(
            "**Factor premia are univariate.** They do not control for one another, so "
            "correlated factors each claim the same premium."
        )
    limitations.append(
        "**Transaction costs are modelled, not measured.** The square-root impact law "
        "with k=1.0 is a literature estimate; real fills depend on venue, urgency, and "
        "the state of the book. Costs are understated for thin names and at larger AUM."
    )
    limitations.append(
        "**Short selling assumes available, cheap borrow** at a flat 50 bps annually. "
        "Hard-to-borrow names cost far more and can be recalled, which no line of this "
        "backtest models."
    )
    limitations.append(
        "**Survivorship.** The universe is defined from currently-listed tickers, so "
        "companies that delisted or were acquired during the window are absent. That "
        "biases historical results upward by construction."
    )
    limitations.append(
        "**The backtest has no market impact feedback.** The strategy's own trading is "
        "assumed not to move prices or to be detected by other participants."
    )
    if ml.get("status") == "ok":
        ess = ml.get("effective_sample_size", {})
        if ess.get("effective_n") and ess["effective_n"] < 100:
            limitations.append(
                f"**The effective sample is {ess['effective_n']} independent observations.** "
                "Every p-value and confidence interval here rests on that, not on the raw "
                f"{ess.get('raw_n')} rows."
            )
    for lim in limitations:
        A(f"- {lim}")
    A("")

    # ------------------------------------------------------------- approval
    A("---")
    A("")
    A("## Approval status")
    A("")
    A(f"**{state.get('status', 'unknown').upper()}** — this report is a research artefact, "
      "not an investment recommendation. No result from this platform is decision-grade "
      "until a named human has reviewed and approved it. The system cannot approve itself.")
    A("")

    # ------------------------------------------------------------ agent trace
    steps = state.get("steps") or []
    if steps:
        A("## Agent execution trace")
        A("")
        A("| # | Agent | Status | Seconds |")
        A("|---|---|---|---|")
        for i, s in enumerate(steps, 1):
            A(f"| {i} | `{s.get('agent')}` | {s.get('status')} | {_num(s.get('seconds'), 2)} |")
        A("")
        degraded = [s for s in steps if s.get("degraded_reason")]
        if degraded:
            A("**Degraded steps:**")
            A("")
            for s in degraded:
                A(f"- `{s['agent']}` — {s['degraded_reason']}")
            A("")

    return "\n".join(L)
