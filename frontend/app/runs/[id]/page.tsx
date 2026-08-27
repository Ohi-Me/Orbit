"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  ApiError,
  decideRun,
  getRun,
  getRunLineage,
  getRunModels,
  getRunReport,
  getRunSteps,
} from "../../lib/api";
import type { ModelFold, RunDetail, RunStep } from "../../lib/types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Stat,
  SyntheticWarning,
  Table,
  Tabs,
  Td,
  num,
  pct,
  shortDate,
  statusTone,
  verdictTone,
} from "../../components/ui";

const CHART_COLORS = ["#1C2B45", "#1F7A5C", "#C97A2B", "#B23B2E", "#5B6259", "#28405F"];

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [run, setRun] = useState<RunDetail | null>(null);
  const [steps, setSteps] = useState<RunStep[]>([]);
  const [folds, setFolds] = useState<ModelFold[]>([]);
  const [lineage, setLineage] = useState<any>(null);
  const [report, setReport] = useState<string>("");
  const [tab, setTab] = useState("overview");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [deciding, setDeciding] = useState(false);
  const [feedback, setFeedback] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await getRun(id);
      setRun(r);
      setError(null);
      const [st, md, ln] = await Promise.allSettled([
        getRunSteps(id),
        getRunModels(id),
        getRunLineage(id),
      ]);
      if (st.status === "fulfilled") setSteps(st.value.steps);
      if (md.status === "fulfilled") setFolds(md.value.folds);
      if (ln.status === "fulfilled") setLineage(ln.value);
      // The report is served from its own endpoint rather than inlined in the
      // run payload, so it is fetched separately and may not exist yet.
      try {
        setReport(await getRunReport(id));
      } catch {
        /* no report until the run reaches the report agent */
      }
      return r.status;
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
      return "failed";
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    let alive = true;
    load().then((status) => {
      if (!alive) return;
      if (status === "queued" || status === "running") {
        const iv = setInterval(async () => {
          const s = await load();
          if (s !== "queued" && s !== "running") clearInterval(iv);
        }, 4000);
      }
    });
    return () => {
      alive = false;
    };
  }, [load]);

  const decide = async (decision: "approved" | "rejected") => {
    setDeciding(true);
    try {
      await decideRun(id, decision, feedback);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setDeciding(false);
    }
  };

  if (loading) return <Spinner label="Loading run…" />;
  if (error && !run) return <ErrorNote>{error}</ErrorNote>;
  if (!run) return <Empty>Run not found.</Empty>;

  const result = run.result || {};
  const comparison = result.backtest_comparison;
  const ml = result.ml_result || {};
  const critic = result.critic_result;
  const risk = result.risk_result || {};
  const portfolio = result.portfolio_result || {};
  const isSynthetic = run.is_synthetic;
  const busy = run.status === "queued" || run.status === "running";

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "factors", label: "Factors" },
    { id: "models", label: "Models" },
    { id: "backtest", label: "Backtest" },
    { id: "risk", label: "Risk" },
    { id: "portfolio", label: "Portfolio" },
    {
      id: "critic",
      label: "Critique",
      badge: critic?.n_checks_failed ? (
        <Badge tone={critic.n_errors ? "fail" : "warn"}>{critic.n_checks_failed}</Badge>
      ) : undefined,
    },
    { id: "trace", label: "Trace" },
    { id: "report", label: "Report" },
  ];

  return (
    <div className="space-y-5">
      {/* ------------------------------------------------------------ header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="max-w-3xl">
          <h1 className="font-display text-2xl leading-snug text-navy">{run.question}</h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted">
            <span className="font-mono">{run.id.slice(0, 8)}</span>
            <Badge tone={statusTone(run.status)}>{run.status.replace(/_/g, " ")}</Badge>
            {critic && (
              <Badge tone={verdictTone(critic.overall_verdict)}>{critic.overall_verdict}</Badge>
            )}
            {isSynthetic === false && <Badge tone="ok">live data</Badge>}
            <span>{shortDate(run.created_at)}</span>
            {run.total_seconds && <span className="font-mono">{run.total_seconds.toFixed(0)}s</span>}
          </div>
        </div>
        <Button variant="ghost" onClick={() => router.push("/runs")}>
          All runs
        </Button>
      </div>

      {busy && <Spinner label="Run in progress — this page refreshes automatically." />}
      {run.error && <ErrorNote>{run.error}</ErrorNote>}
      <SyntheticWarning isSynthetic={isSynthetic} />

      {/* ------------------------------------------------------ approval gate */}
      {run.status === "awaiting_approval" && (
        <Card
          title="Human approval required"
          subtitle="Nothing here is decision-grade until a named person accepts it. The system cannot approve itself."
        >
          {critic && (
            <div className="mb-3 space-y-1 text-xs">
              <p>
                The Critic returned <strong>{critic.overall_verdict}</strong> with{" "}
                {critic.n_errors} error-level and {critic.n_warnings} warning-level flags.
              </p>
              {critic.recommended_revisions?.length > 0 && (
                <p className="text-muted">
                  Recommended revisions: {critic.recommended_revisions.join(", ")}
                </p>
              )}
            </div>
          )}
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={2}
            placeholder="Reviewer notes — what you checked, and what you are accepting or rejecting."
            className="w-full border border-grid bg-white px-3 py-2 text-sm outline-none focus:border-navy"
          />
          <div className="mt-3 flex gap-2">
            <Button onClick={() => decide("approved")} disabled={deciding}>
              Approve
            </Button>
            <Button variant="danger" onClick={() => decide("rejected")} disabled={deciding}>
              Reject
            </Button>
          </div>
          <p className="mt-2 text-[11px] text-muted">
            Approval requires a signed-in account so the decision is attributable.
          </p>
        </Card>
      )}

      {run.approval && run.approval.status !== "pending" && (
        <div className="border border-grid bg-white/70 px-4 py-2 text-xs">
          <Badge tone={run.approval.status === "approved" ? "ok" : "fail"}>
            {run.approval.status}
          </Badge>{" "}
          by <strong>{run.approval.decided_by}</strong> on {shortDate(run.approval.decided_at)}
          {run.approval.feedback && (
            <p className="mt-1 text-muted">&ldquo;{run.approval.feedback}&rdquo;</p>
          )}
        </div>
      )}

      <Tabs tabs={tabs} active={tab} onChange={setTab} />

      {/* --------------------------------------------------------- overview */}
      {tab === "overview" && (
        <div className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Sharpe (net of costs)"
              value={num(run.sharpe, 2)}
              tone={(run.sharpe ?? 0) > 0 ? "gain" : "loss"}
              hint="Best strategy, after modelled spread, impact and borrow"
            />
            <Stat label="CAGR" value={pct(run.cagr)} tone={(run.cagr ?? 0) > 0 ? "gain" : "loss"} />
            <Stat label="Max drawdown" value={pct(run.max_drawdown)} tone="loss" />
            <Stat
              label="Critic flags"
              value={String(run.n_critic_flags ?? "—")}
              tone={critic?.n_errors ? "loss" : critic?.n_warnings ? "warn" : "gain"}
              hint={critic ? `${critic.n_errors} errors, ${critic.n_warnings} warnings` : undefined}
            />
          </div>

          {ml.model_verdict && (
            <Card title="Model finding">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={ml.model_verdict === "candidate_signal" ? "ok" : "warn"}>
                  {ml.model_verdict.replace(/_/g, " ")}
                </Badge>
                {ml.best_model && <span className="font-mono text-xs">{ml.best_model}</span>}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">{ml.model_verdict_detail}</p>
            </Card>
          )}

          {run.plan && (
            <Card title="Research plan" subtitle={`Produced by ${run.plan.planner_backend}`}>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-1.5 text-xs">
                  <p>
                    <span className="text-muted">Universe:</span>{" "}
                    <span className="font-mono">{run.plan.universe.length} names</span>
                  </p>
                  <p className="font-mono text-[11px] leading-relaxed text-muted">
                    {run.plan.universe.join(" · ")}
                  </p>
                  <p className="text-muted">{run.plan.universe_rationale}</p>
                </div>
                <div className="space-y-1.5 text-xs">
                  <p>
                    <span className="text-muted">History:</span>{" "}
                    <span className="font-mono">{run.plan.n_days}d</span> ·{" "}
                    <span className="text-muted">Horizon:</span>{" "}
                    <span className="font-mono">{run.plan.label_horizon}d</span> ·{" "}
                    <span className="text-muted">Folds:</span>{" "}
                    <span className="font-mono">{run.plan.n_folds}</span>
                  </p>
                  <p>
                    <span className="text-muted">Factor families:</span>{" "}
                    {run.plan.factor_families.join(", ")}
                  </p>
                  <p>
                    <span className="text-muted">Deep learning:</span>{" "}
                    {String(run.plan.include_deep_learning)}
                  </p>
                </div>
              </div>
              {run.plan.notes?.length > 0 && (
                <ul className="mt-3 space-y-1">
                  {run.plan.notes.map((n, i) => (
                    <li key={i} className="text-[11px] text-muted">
                      · {n}
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          )}

          {(result.revisions_applied?.length ?? 0) > 0 && (
            <Card title="Revisions the Critic forced" subtitle="The verdict changed what ran">
              <ul className="space-y-1.5 text-xs text-muted">
                {result.revisions_applied!.map((r, i) => (
                  <li key={i}>· {r}</li>
                ))}
              </ul>
            </Card>
          )}

          {lineage && (
            <Card title="Data lineage" subtitle="What this run actually consumed">
              <Table head={["Dataset", "Provider", "Synthetic", "Window", "Rows"]} dense>
                {(lineage.snapshots ?? []).map((s: any, i: number) => (
                  <tr key={i} className="border-b border-grid/60">
                    <Td>{s.dataset}</Td>
                    <Td mono>{s.provider}</Td>
                    <Td>
                      {s.is_synthetic ? <Badge tone="warn">yes</Badge> : <Badge tone="ok">no</Badge>}
                    </Td>
                    <Td mono>
                      {s.start_date ?? "—"} → {s.end_date ?? "—"}
                    </Td>
                    <Td mono>{s.n_rows?.toLocaleString() ?? "—"}</Td>
                  </tr>
                ))}
              </Table>
            </Card>
          )}
        </div>
      )}

      {/* ---------------------------------------------------------- factors */}
      {tab === "factors" && (
        <div className="space-y-5">
          <Card
            title="Factor selection"
            subtitle="Factors are admitted on measured coverage, not on intent"
          >
            <div className="mb-3 flex flex-wrap gap-1.5">
              {(result.selected_factors ?? []).map((f) => (
                <Badge key={f} tone="info">
                  {f}
                </Badge>
              ))}
            </div>
            {result.factor_diagnostics?.rejected &&
              Object.keys(result.factor_diagnostics.rejected).length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-[11px] uppercase tracking-wider text-muted">Rejected</p>
                  {Object.entries(result.factor_diagnostics.rejected).map(([f, reason]) => (
                    <p key={f} className="text-xs text-muted">
                      <span className="font-mono text-loss">{f}</span> — {reason}
                    </p>
                  ))}
                </div>
              )}
          </Card>

          {result.factor_ic && Object.keys(result.factor_ic).length > 0 && (
            <Card
              title="Information coefficient"
              subtitle="Per-date rank correlation between each factor and the forward return it predicts"
            >
              <div className="h-64 w-full">
                <ResponsiveContainer>
                  <BarChart
                    data={Object.values(result.factor_ic)
                      .filter((s: any) => s.mean_ic !== null && s.mean_ic !== undefined)
                      .sort((a: any, b: any) => Math.abs(b.mean_ic) - Math.abs(a.mean_ic))}
                    margin={{ top: 5, right: 10, left: 0, bottom: 60 }}
                  >
                    <CartesianGrid strokeDasharray="2 2" stroke="#D7D9D0" />
                    <XAxis
                      dataKey="factor"
                      angle={-45}
                      textAnchor="end"
                      interval={0}
                      tick={{ fontSize: 10, fill: "#5B6259" }}
                    />
                    <YAxis tick={{ fontSize: 10, fill: "#5B6259" }} />
                    <Tooltip
                      contentStyle={{ fontSize: 11, borderRadius: 0, border: "1px solid #D7D9D0" }}
                      formatter={(v: any) => Number(v).toFixed(4)}
                    />
                    <ReferenceLine y={0} stroke="#14181C" />
                    <Bar dataKey="mean_ic" name="Mean IC">
                      {Object.values(result.factor_ic)
                        .filter((s: any) => s.mean_ic !== null)
                        .sort((a: any, b: any) => Math.abs(b.mean_ic) - Math.abs(a.mean_ic))
                        .map((s: any, i: number) => (
                          <Cell key={i} fill={s.mean_ic > 0 ? "#1F7A5C" : "#B23B2E"} />
                        ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <Note>
                A mean IC around 0.03 with a positive information ratio is the conventional
                threshold for a usable equity factor. With a small universe the per-date
                correlation is estimated on few names and is correspondingly noisy.
              </Note>
            </Card>
          )}

          {result.fama_macbeth?.status === "ok" && (
            <Card
              title="Fama-MacBeth factor premia"
              subtitle={`Mode: ${result.fama_macbeth.mode} · ${result.fama_macbeth.n_names} names, ${result.fama_macbeth.n_factors} factors`}
            >
              <Table head={["Factor", "Premium", "HAC t", "Naive t", "Inflation", "p", "Sig"]} dense>
                {Object.values(result.fama_macbeth.per_factor ?? {})
                  .filter((r: any) => r.status === "ok")
                  .sort((a: any, b: any) => Math.abs(b.t_stat ?? 0) - Math.abs(a.t_stat ?? 0))
                  .map((r: any) => (
                    <tr key={r.factor} className="border-b border-grid/60">
                      <Td mono>{r.factor}</Td>
                      <Td mono>{num(r.mean_premium, 5)}</Td>
                      <Td mono className={Math.abs(r.t_stat) > 2 ? "text-gain" : ""}>
                        {num(r.t_stat, 2)}
                      </Td>
                      <Td mono className="text-muted line-through">
                        {num(r.naive_t_stat, 2)}
                      </Td>
                      <Td mono className="text-warn">
                        {r.t_inflation_vs_naive ? `${num(r.t_inflation_vs_naive, 2)}×` : "—"}
                      </Td>
                      <Td mono>{num(r.p_value, 4)}</Td>
                      <Td>
                        {r.significant_at_95 ? <Badge tone="ok">yes</Badge> : <Badge>no</Badge>}
                      </Td>
                    </tr>
                  ))}
              </Table>
              <div className="mt-3">
                <Note>{result.fama_macbeth.caveat}</Note>
              </div>
              <div className="mt-2">
                <Note>
                  The struck-through naive t-statistic is shown deliberately. Overlapping
                  forward-return windows inflate it by the factor in the next column — several
                  factors that look decisively significant naively are not significant at all once
                  corrected.
                </Note>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ----------------------------------------------------------- models */}
      {tab === "models" && (
        <div className="space-y-5">
          {ml.validation_scheme && (
            <Card title="Validation scheme">
              <div className="flex flex-wrap gap-3 text-xs">
                <Stat label="Type" value={<span className="text-xs">{ml.validation_scheme.type}</span>} />
                <Stat label="Purge" value={`${ml.validation_scheme.purge_days}d`} />
                <Stat label="Embargo" value={`${ml.validation_scheme.embargo_days}d`} />
                <Stat label="Horizon" value={`${ml.validation_scheme.label_horizon_days}d`} />
              </div>
              <div className="mt-3">
                <Note>{ml.validation_scheme.note}</Note>
              </div>
            </Card>
          )}

          {ml.summary && (
            <Card
              title="Model comparison"
              subtitle="Identical folds, features and standardization — so differences are attributable to the model"
            >
              <Table
                head={["Model", "Accuracy", "Base rate", "Lift", "AUC", "Mean IC", "Signal return", "Degenerate", "Train"]}
              >
                {ml.summary.map((s: any) => (
                  <tr key={s.model} className="border-b border-grid/60">
                    <Td mono>{s.model}</Td>
                    <Td mono>{num(s.mean_accuracy, 3)}</Td>
                    <Td mono className="text-muted">
                      {num(s.mean_base_rate, 3)}
                    </Td>
                    <Td mono className={(s.mean_accuracy_lift ?? 0) > 0 ? "text-gain" : "text-loss"}>
                      {num(s.mean_accuracy_lift, 3)}
                    </Td>
                    <Td mono>{num(s.mean_auc, 3)}</Td>
                    <Td
                      mono
                      className={(s.mean_information_coefficient ?? 0) > 0 ? "text-gain" : "text-loss"}
                    >
                      {num(s.mean_information_coefficient, 4)}
                    </Td>
                    <Td mono>{num(s.mean_signal_return, 5)}</Td>
                    <Td mono>
                      {s.n_degenerate_folds > 0 ? (
                        <Badge tone="fail">{s.n_degenerate_folds}</Badge>
                      ) : (
                        "0"
                      )}
                    </Td>
                    <Td mono className="text-muted">
                      {s.train_seconds}s
                    </Td>
                  </tr>
                ))}
              </Table>
              <div className="mt-3">
                <Note>
                  <strong>Base rate</strong> is the share of the majority direction. Accuracy below
                  it means the model is worse than always predicting the same way — which is why
                  raw accuracy is never the headline. <strong>Lift</strong> is the only column in
                  which a directional model can claim skill. A <strong>degenerate</strong> fold is
                  one where the model predicted a single class for over 95% of samples.
                </Note>
              </div>
            </Card>
          )}

          {folds.length > 0 && (
            <Card title="Per-fold detail" subtitle="Stability across time matters more than the mean">
              <Table head={["Model", "Fold", "Test window", "n", "Accuracy", "AUC", "IC", "t", "p"]} dense>
                {folds.map((f, i) => (
                  <tr key={i} className="border-b border-grid/60">
                    <Td mono>{f.model}</Td>
                    <Td mono>{f.fold}</Td>
                    <Td mono className="text-[10px]">
                      {f.test[0]} → {f.test[1]}
                    </Td>
                    <Td mono>{f.n_test}</Td>
                    <Td mono>{num(f.accuracy, 3)}</Td>
                    <Td mono>{num(f.auc, 3)}</Td>
                    <Td mono>{num(f.information_coefficient, 4)}</Td>
                    <Td mono>{num(f.signal_t_stat, 2)}</Td>
                    <Td mono>{num(f.signal_p_value, 3)}</Td>
                  </tr>
                ))}
              </Table>
            </Card>
          )}

          {ml.effective_sample_size && (
            <Card title="Effective sample size">
              <div className="grid gap-3 sm:grid-cols-3">
                <Stat label="Raw rows" value={ml.effective_sample_size.raw_n?.toLocaleString()} />
                <Stat
                  label="Effective observations"
                  value={ml.effective_sample_size.effective_n?.toLocaleString()}
                  tone="warn"
                />
                <Stat
                  label="Cross-correlation"
                  value={num(ml.effective_sample_size.avg_cross_correlation, 2)}
                />
              </div>
              <div className="mt-3">
                <Note>{ml.effective_sample_size.note}</Note>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* --------------------------------------------------------- backtest */}
      {tab === "backtest" && comparison?.status === "ok" && (
        <div className="space-y-5">
          <Card
            title="Strategy comparison"
            subtitle="Every strategy trades out-of-sample model scores through identical mechanics"
          >
            <Table
              head={["Strategy", "CAGR", "Sharpe net", "Sharpe gross", "Max DD", "Turnover", "HAC t", "p", "Deflated Sharpe"]}
            >
              {Object.entries(comparison.strategies).map(([name, r]: [string, any]) => {
                if (r.status !== "ok")
                  return (
                    <tr key={name} className="border-b border-grid/60">
                      <Td mono>{name}</Td>
                      <Td className="text-xs text-muted" mono>
                        {r.status}
                      </Td>
                    </tr>
                  );
                const m = r.metrics;
                const sig = r.significance ?? {};
                const dsr = sig.deflated_sharpe ?? {};
                const isBest = name === comparison.best_strategy;
                return (
                  <tr
                    key={name}
                    className={`border-b border-grid/60 ${isBest ? "bg-navy/[0.04]" : ""}`}
                  >
                    <Td mono>
                      {name} {isBest && <Badge tone="info">best</Badge>}
                    </Td>
                    <Td mono className={m.cagr > 0 ? "text-gain" : "text-loss"}>
                      {pct(m.cagr)}
                    </Td>
                    <Td mono className={m.sharpe_ratio > 0 ? "text-gain" : "text-loss"}>
                      {num(m.sharpe_ratio, 2)}
                    </Td>
                    <Td mono className="text-muted">
                      {num(m.sharpe_gross_of_costs, 2)}
                    </Td>
                    <Td mono className="text-loss">
                      {pct(m.max_drawdown)}
                    </Td>
                    <Td mono>{num(r.costs?.avg_turnover_per_rebalance, 2)}×</Td>
                    <Td mono>{num(sig.t_stat_hac, 2)}</Td>
                    <Td mono className={sig.p_value_hac < 0.05 ? "text-gain" : "text-muted"}>
                      {num(sig.p_value_hac, 3)}
                    </Td>
                    <Td mono>
                      {dsr.deflated_sharpe !== null && dsr.deflated_sharpe !== undefined ? (
                        <span className={dsr.significant_at_95 ? "text-gain" : "text-loss"}>
                          {num(dsr.deflated_sharpe, 3)}
                        </span>
                      ) : (
                        "—"
                      )}
                    </Td>
                  </tr>
                );
              })}
            </Table>
            <div className="mt-3">
              <Note>{comparison.multiple_testing_note}</Note>
            </div>
            {comparison.model_beats_baseline === false && (
              <div className="mt-3 border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
                No machine-learning model beat the hand-weighted composite baseline. That is the
                finding: on this universe and horizon, the added model complexity did not earn its
                keep.
              </div>
            )}
          </Card>

          {(() => {
            const curves = Object.entries(comparison.strategies).filter(
              ([, r]: [string, any]) => r.status === "ok" && r.equity_curve?.length
            );
            if (!curves.length) return null;
            const dates = (curves[0][1] as any).equity_curve.map((p: any) => p.date);
            const merged = dates.map((d: string, i: number) => {
              const row: any = { date: d };
              curves.forEach(([name, r]: [string, any]) => {
                row[name] = r.equity_curve[i]?.equity;
              });
              return row;
            });
            return (
              <Card title="Equity curves" subtitle="Net of modelled spread, market impact and borrow">
                <div className="h-72 w-full">
                  <ResponsiveContainer>
                    <LineChart data={merged} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="2 2" stroke="#D7D9D0" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: "#5B6259" }}
                        minTickGap={40}
                      />
                      <YAxis tick={{ fontSize: 10, fill: "#5B6259" }} domain={["auto", "auto"]} />
                      <Tooltip
                        contentStyle={{ fontSize: 11, borderRadius: 0, border: "1px solid #D7D9D0" }}
                        formatter={(v: any) => Number(v).toFixed(4)}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <ReferenceLine y={1} stroke="#14181C" strokeDasharray="3 3" />
                      {curves.map(([name], i) => (
                        <Line
                          key={name}
                          type="monotone"
                          dataKey={name}
                          stroke={CHART_COLORS[i % CHART_COLORS.length]}
                          dot={false}
                          strokeWidth={name === comparison.best_strategy ? 2 : 1.2}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            );
          })()}

          {(() => {
            const best = comparison.best_strategy
              ? (comparison.strategies[comparison.best_strategy] as any)
              : null;
            if (!best || best.status !== "ok") return null;
            return (
              <>
                {best.drawdown_curve?.length > 0 && (
                  <Card title={`Drawdown — ${comparison.best_strategy}`}>
                    <div className="h-52 w-full">
                      <ResponsiveContainer>
                        <AreaChart data={best.drawdown_curve}>
                          <CartesianGrid strokeDasharray="2 2" stroke="#D7D9D0" />
                          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#5B6259" }} minTickGap={40} />
                          <YAxis tick={{ fontSize: 10, fill: "#5B6259" }} />
                          <Tooltip
                            contentStyle={{ fontSize: 11, borderRadius: 0, border: "1px solid #D7D9D0" }}
                            formatter={(v: any) => `${(Number(v) * 100).toFixed(2)}%`}
                          />
                          <Area type="monotone" dataKey="drawdown" stroke="#B23B2E" fill="#B23B2E" fillOpacity={0.15} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                )}

                <Card title="Cost model" subtitle={best.costs?.model}>
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                    <Stat label="Spread cost" value={num(best.costs?.total_spread_cost, 5)} />
                    <Stat label="Impact cost" value={num(best.costs?.total_impact_cost, 5)} tone="warn" />
                    <Stat label="Total drag" value={num(best.costs?.total_cost_drag_on_return, 5)} tone="loss" />
                    <Stat
                      label="Assumed AUM"
                      value={`$${(best.costs?.assumed_aum / 1e6).toFixed(0)}M`}
                      hint="Impact scales with size"
                    />
                  </div>
                  {best.significance?.sharpe_bootstrap_ci?.ci_low !== undefined && (
                    <div className="mt-3">
                      <Note>
                        Sharpe 95% bootstrap confidence interval:{" "}
                        <span className="font-mono">
                          [{num(best.significance.sharpe_bootstrap_ci.ci_low, 2)},{" "}
                          {num(best.significance.sharpe_bootstrap_ci.ci_high, 2)}]
                        </span>{" "}
                        from {best.significance.sharpe_bootstrap_ci.n_boot} stationary-block
                        resamples. An interval spanning zero means the point estimate is not
                        distinguishable from no edge.
                      </Note>
                    </div>
                  )}
                </Card>
              </>
            );
          })()}
        </div>
      )}
      {tab === "backtest" && comparison?.status !== "ok" && (
        <Empty>No backtest results for this run.</Empty>
      )}

      {/* ------------------------------------------------------------- risk */}
      {tab === "risk" && (
        <div className="space-y-5">
          {risk.status === "ok" ? (
            <>
              <Card title="Tail risk" subtitle={risk.var_interpretation}>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                  <Stat label="VaR 95% hist" value={pct(risk.value_at_risk_daily?.historical_95)} />
                  <Stat label="VaR 95% CF" value={pct(risk.value_at_risk_daily?.cornish_fisher_95)} tone="warn" />
                  <Stat label="VaR 99% CF" value={pct(risk.value_at_risk_daily?.cornish_fisher_99)} tone="loss" />
                  <Stat label="CVaR 95%" value={pct(risk.conditional_var_95)} tone="loss" />
                  <Stat label="CVaR 99%" value={pct(risk.conditional_var_99)} tone="loss" />
                </div>
              </Card>

              {risk.benchmark_relative?.beta !== null && risk.benchmark_relative && (
                <Card title="Benchmark relative">
                  <div className="grid gap-3 sm:grid-cols-4">
                    <Stat label="Beta" value={num(risk.benchmark_relative.beta, 3)} />
                    <Stat label="Alpha (ann.)" value={pct(risk.benchmark_relative.alpha_annualized)} />
                    <Stat label="Information ratio" value={num(risk.benchmark_relative.information_ratio, 2)} />
                    <Stat label="R² vs benchmark" value={num(risk.benchmark_relative.r_squared_vs_benchmark, 3)} />
                  </div>
                  <p className="mt-2 text-[11px] text-muted">
                    {risk.benchmark_relative.n_aligned_observations} date-aligned observations.
                  </p>
                </Card>
              )}

              {risk.factor_risk?.status === "ok" && (
                <Card title="Factor risk decomposition" subtitle={risk.factor_risk.method}>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <Stat label="Systematic share" value={pct(risk.factor_risk.systematic_variance_share)} />
                    <Stat label="Idiosyncratic share" value={pct(risk.factor_risk.idiosyncratic_variance_share)} tone="warn" />
                    <Stat label="Idio vol (ann.)" value={pct(risk.factor_risk.annualized_idiosyncratic_vol)} />
                  </div>
                  <p className="mt-3 text-xs">
                    <span className="text-muted">Dominant risk factors:</span>{" "}
                    <span className="font-mono">
                      {(risk.factor_risk.dominant_risk_factors ?? []).join(", ")}
                    </span>
                  </p>
                  <Table head={["Factor", "Exposure", "Variance share"]} dense>
                    {Object.entries(risk.factor_risk.factor_exposures ?? {})
                      .sort(
                        (a: any, b: any) =>
                          Math.abs(b[1].variance_share ?? 0) - Math.abs(a[1].variance_share ?? 0)
                      )
                      .map(([f, v]: [string, any]) => (
                        <tr key={f} className="border-b border-grid/60">
                          <Td mono>{f}</Td>
                          <Td mono>{num(v.exposure, 4)}</Td>
                          <Td mono>{pct(v.variance_share)}</Td>
                        </tr>
                      ))}
                  </Table>
                </Card>
              )}

              {risk.concentration?.status === "ok" && (
                <Card title="Concentration and liquidity" subtitle={risk.concentration.note}>
                  <div className="grid gap-3 sm:grid-cols-4">
                    <Stat label="Positions" value={risk.concentration.n_positions} />
                    <Stat label="Largest" value={pct(risk.concentration.largest_position)} />
                    <Stat
                      label="HHI"
                      value={num(risk.concentration.herfindahl_index, 3)}
                      tone={risk.concentration.concentration_flag ? "loss" : "neutral"}
                    />
                    <Stat label="Effective N" value={num(risk.concentration.effective_n_positions, 1)} />
                  </div>
                  {risk.concentration.concentration_flag && (
                    <div className="mt-3 border border-loss/40 bg-loss/5 px-3 py-2 text-xs text-loss">
                      Concentration flag: the book is dominated by a small number of positions, so
                      idiosyncratic risk exceeds factor risk. On a small universe the tercile legs
                      collapse to one or two names — this is a bet on those companies, not on a factor.
                    </div>
                  )}
                </Card>
              )}

              {risk.scenarios?.historical_scenarios && (
                <Card title="Historical scenario replay" subtitle={risk.scenarios.method}>
                  <Table head={["Scenario", "Benchmark", "Estimated strategy return"]} dense>
                    {Object.entries(risk.scenarios.historical_scenarios).map(
                      ([k, s]: [string, any]) => (
                        <tr key={k} className="border-b border-grid/60">
                          <Td>{s.description}</Td>
                          <Td mono className="text-loss">
                            {pct(s.benchmark_return)}
                          </Td>
                          <Td mono className={(s.estimated_strategy_return ?? 0) < 0 ? "text-loss" : "text-gain"}>
                            {pct(s.estimated_strategy_return)}
                          </Td>
                        </tr>
                      )
                    )}
                  </Table>
                </Card>
              )}
            </>
          ) : (
            <Empty>No risk analysis for this run.</Empty>
          )}
        </div>
      )}

      {/* -------------------------------------------------------- portfolio */}
      {tab === "portfolio" && (
        <div className="space-y-5">
          {portfolio.status === "ok" ? (
            <>
              <Card
                title="Allocation methods, walk-forward"
                subtitle="Covariance re-estimated on a trailing window at each rebalance — never full-sample"
              >
                <Table head={["Method", "OOS Sharpe", "OOS return", "OOS vol", "Max DD", "Turnover", "In-sample Sharpe"]}>
                  {Object.entries(portfolio.walk_forward ?? {}).map(([name, r]: [string, any]) => {
                    if (r.status !== "ok")
                      return (
                        <tr key={name} className="border-b border-grid/60">
                          <Td mono>{name}</Td>
                          <Td className="text-xs text-muted">{r.status}</Td>
                        </tr>
                      );
                    const o = r.out_of_sample;
                    const isBest = name === portfolio.best_method;
                    const isSharpe = portfolio.in_sample_reference?.[name]?.sharpe;
                    return (
                      <tr key={name} className={`border-b border-grid/60 ${isBest ? "bg-navy/[0.04]" : ""}`}>
                        <Td mono>
                          {name} {isBest && <Badge tone="info">best</Badge>}
                        </Td>
                        <Td mono className="text-gain">{num(o.sharpe, 2)}</Td>
                        <Td mono>{pct(o.annualized_return)}</Td>
                        <Td mono>{pct(o.annualized_volatility)}</Td>
                        <Td mono className="text-loss">{pct(o.max_drawdown)}</Td>
                        <Td mono>{num(r.avg_turnover_per_rebalance, 3)}</Td>
                        <Td mono className="text-muted">{num(isSharpe, 2)}</Td>
                      </tr>
                    );
                  })}
                </Table>
                {portfolio.in_sample_optimism !== null && portfolio.in_sample_optimism !== undefined && (
                  <div className="mt-3">
                    <Note>
                      <strong>In-sample optimism: {num(portfolio.in_sample_optimism, 3)} Sharpe.</strong>{" "}
                      {portfolio.optimism_note}
                    </Note>
                  </div>
                )}
                <div className="mt-2">
                  <Note>{portfolio.covariance_estimator}</Note>
                </div>
              </Card>

              {(() => {
                const best = portfolio.walk_forward?.[portfolio.best_method];
                const weights = best?.weights_latest;
                if (!weights) return null;
                const data = Object.entries(weights)
                  .map(([ticker, w]) => ({ ticker, weight: Number(w) }))
                  .sort((a, b) => b.weight - a.weight);
                return (
                  <Card title={`Latest weights — ${portfolio.best_method}`}>
                    <div className="h-64 w-full">
                      <ResponsiveContainer>
                        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                          <CartesianGrid strokeDasharray="2 2" stroke="#D7D9D0" />
                          <XAxis dataKey="ticker" angle={-45} textAnchor="end" interval={0} tick={{ fontSize: 10, fill: "#5B6259" }} />
                          <YAxis tick={{ fontSize: 10, fill: "#5B6259" }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                          <Tooltip
                            contentStyle={{ fontSize: 11, borderRadius: 0, border: "1px solid #D7D9D0" }}
                            formatter={(v: any) => `${(Number(v) * 100).toFixed(2)}%`}
                          />
                          <Bar dataKey="weight" fill="#1C2B45" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    {run.status === "approved" ? (
                      <Note>
                        This run is approved, so these weights can be adopted into a portfolio book
                        from the Portfolio screen.
                      </Note>
                    ) : (
                      <Note>
                        Weights can only be adopted into a live book after this run is approved.
                        Research output does not become a position without an explicit human decision.
                      </Note>
                    )}
                  </Card>
                );
              })()}
            </>
          ) : (
            <Empty>No portfolio construction results for this run.</Empty>
          )}
        </div>
      )}

      {/* ----------------------------------------------------------- critic */}
      {tab === "critic" && critic && (
        <div className="space-y-5">
          <Card
            title={critic.overall_verdict.replace(/_/g, " ")}
            subtitle={critic.verdict_note}
            right={<Badge tone={verdictTone(critic.overall_verdict)}>{critic.recommended_action}</Badge>}
          >
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat label="Checks run" value={critic.n_checks_run} />
              <Stat label="Errors" value={critic.n_errors} tone={critic.n_errors ? "loss" : "gain"} />
              <Stat label="Warnings" value={critic.n_warnings} tone={critic.n_warnings ? "warn" : "gain"} />
            </div>
          </Card>

          <div className="space-y-3">
            {critic.checks.map((c) => (
              <div
                key={c.check}
                className={`border px-4 py-3 ${
                  c.passed === false
                    ? c.severity === "error"
                      ? "border-loss/40 bg-loss/5"
                      : "border-warn/40 bg-warn/5"
                    : c.passed === true
                      ? "border-grid bg-white/70"
                      : "border-grid bg-paper/50"
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={c.passed === true ? "ok" : c.passed === false ? (c.severity === "error" ? "fail" : "warn") : "muted"}>
                    {c.passed === true ? "pass" : c.passed === false ? "flag" : "skip"}
                  </Badge>
                  <span className="font-mono text-xs">{c.check}</span>
                  <span className="text-[10px] uppercase tracking-wider text-muted">{c.severity}</span>
                </div>
                <div className="mt-2 space-y-1 text-xs leading-relaxed text-muted">
                  {Array.isArray(c.detail) ? (
                    c.detail.map((d, i) => <p key={i}>· {d}</p>)
                  ) : (
                    <p>{c.detail}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {tab === "critic" && !critic && <Empty>No critique for this run.</Empty>}

      {/* ------------------------------------------------------------ trace */}
      {tab === "trace" && (
        <Card title="Agent execution trace" subtitle="What ran, in what order, how long, and what degraded">
          {steps.length === 0 ? (
            <Empty>No steps recorded.</Empty>
          ) : (
            <>
              <Table head={["#", "Agent", "Status", "Seconds", "Summary"]}>
                {steps.map((s) => (
                  <tr key={s.sequence} className="border-b border-grid/60">
                    <Td mono>{s.sequence}</Td>
                    <Td mono>{s.agent}</Td>
                    <Td>
                      <Badge tone={statusTone(s.status)}>{s.status}</Badge>
                    </Td>
                    <Td mono>{s.seconds.toFixed(2)}</Td>
                    <Td>
                      {s.error ? (
                        <span className="text-[11px] text-loss">{s.error}</span>
                      ) : s.degraded_reason ? (
                        <span className="text-[11px] text-warn">{s.degraded_reason}</span>
                      ) : s.summary ? (
                        <span className="font-mono text-[10px] text-muted">
                          {JSON.stringify(s.summary).slice(0, 150)}
                        </span>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </Td>
                  </tr>
                ))}
              </Table>
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <Stat label="Total" value={`${steps.reduce((a, s) => a + s.seconds, 0).toFixed(1)}s`} />
                <Stat
                  label="Slowest agent"
                  value={
                    <span className="text-xs">
                      {[...steps].sort((a, b) => b.seconds - a.seconds)[0]?.agent ?? "—"}
                    </span>
                  }
                />
                <Stat label="Failed steps" value={steps.filter((s) => s.status === "failed").length} />
              </div>
            </>
          )}
        </Card>
      )}

      {/* ----------------------------------------------------------- report */}
      {tab === "report" && (
        <Card
          title="Research report"
          right={
            <Button
              variant="ghost"
              onClick={() => {
                const blob = new Blob([report], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `research-${run.id.slice(0, 8)}.md`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              Download .md
            </Button>
          }
        >
          {report ? (
            <article className="prose-quant max-w-none text-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
            </article>
          ) : (
            <Empty>No report yet.</Empty>
          )}
        </Card>
      )}
    </div>
  );
}
