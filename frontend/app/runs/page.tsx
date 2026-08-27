"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, listRuns } from "../lib/api";
import type { RunSummary } from "../lib/types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Table,
  Td,
  num,
  pct,
  relTime,
  statusTone,
  verdictTone,
} from "../components/ui";

const STATUSES = ["", "awaiting_approval", "approved", "rejected", "failed", "running"];

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    setLoading(true);
    listRuns(100, filter || undefined)
      .then((r) => {
        setRuns(r.runs);
        setError(null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.detail : String(e)))
      .finally(() => setLoading(false));
  }, [filter]);

  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-4)));

  const chosen = runs.filter((r) => selected.includes(r.id));

  return (
    <div className="space-y-6">
      <Card
        title="Run history"
        subtitle="Every run is persisted with its plan, provenance, metrics and critique — comparable across time."
        right={
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border border-grid bg-white px-2 py-1 text-xs"
          >
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {s === "" ? "all statuses" : s.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        }
      >
        {loading ? (
          <Spinner label="Loading runs…" />
        ) : error ? (
          <ErrorNote>{error}</ErrorNote>
        ) : runs.length === 0 ? (
          <Empty>No runs match this filter.</Empty>
        ) : (
          <Table
            head={["", "Question", "Status", "Verdict", "Sharpe", "CAGR", "Max DD", "Model", "Data", "When"]}
          >
            {runs.map((r) => (
              <tr key={r.id} className="border-b border-grid/60 hover:bg-navy/[0.03]">
                <Td>
                  <input
                    type="checkbox"
                    checked={selected.includes(r.id)}
                    onChange={() => toggle(r.id)}
                    className="accent-navy"
                    aria-label={`Select run ${r.id.slice(0, 8)} for comparison`}
                  />
                </Td>
                <Td>
                  <Link href={`/runs/${r.id}`} className="text-xs hover:underline">
                    {r.question.length > 78 ? `${r.question.slice(0, 78)}…` : r.question}
                  </Link>
                  <div className="mt-0.5 font-mono text-[10px] text-muted">
                    {r.id.slice(0, 8)} · {r.universe_size ?? "—"} names
                  </div>
                </Td>
                <Td>
                  <Badge tone={statusTone(r.status)}>{r.status.replace(/_/g, " ")}</Badge>
                </Td>
                <Td>
                  {r.critic_verdict ? (
                    <Badge tone={verdictTone(r.critic_verdict)} title={r.critic_verdict}>
                      {r.critic_verdict.split("_")[0]}
                    </Badge>
                  ) : (
                    <span className="text-xs text-muted">—</span>
                  )}
                  {r.n_critic_flags ? (
                    <span className="ml-1 text-[10px] text-muted">{r.n_critic_flags} flags</span>
                  ) : null}
                </Td>
                <Td mono>{num(r.sharpe, 2)}</Td>
                <Td mono>{pct(r.cagr)}</Td>
                <Td mono>{pct(r.max_drawdown)}</Td>
                <Td mono>{r.best_model ?? "—"}</Td>
                <Td>
                  {r.is_synthetic === true ? (
                    <Badge tone="warn">synthetic</Badge>
                  ) : r.is_synthetic === false ? (
                    <Badge tone="ok">live</Badge>
                  ) : (
                    <span className="text-xs text-muted">—</span>
                  )}
                </Td>
                <Td>
                  <span className="text-[11px] text-muted">{relTime(r.created_at)}</span>
                </Td>
              </tr>
            ))}
          </Table>
        )}
      </Card>

      {chosen.length >= 2 && (
        <Card
          title={`Comparing ${chosen.length} runs`}
          subtitle="Side by side, with the caveats that make the numbers comparable or not"
          right={
            <Button variant="ghost" onClick={() => setSelected([])}>
              Clear
            </Button>
          }
        >
          <Table head={["Metric", ...chosen.map((r) => r.id.slice(0, 8))]}>
            {[
              { label: "Question", get: (r: RunSummary) => r.question.slice(0, 40) + "…" },
              { label: "Status", get: (r: RunSummary) => r.status.replace(/_/g, " ") },
              { label: "Critic verdict", get: (r: RunSummary) => r.critic_verdict ?? "—" },
              { label: "Sharpe (net)", get: (r: RunSummary) => num(r.sharpe, 3) },
              { label: "CAGR", get: (r: RunSummary) => pct(r.cagr) },
              { label: "Max drawdown", get: (r: RunSummary) => pct(r.max_drawdown) },
              { label: "Best model", get: (r: RunSummary) => r.best_model ?? "—" },
              { label: "Universe size", get: (r: RunSummary) => String(r.universe_size ?? "—") },
              {
                label: "Data",
                get: (r: RunSummary) =>
                  r.is_synthetic === true ? "SYNTHETIC" : r.is_synthetic === false ? "live" : "—",
              },
              { label: "Runtime", get: (r: RunSummary) => (r.total_seconds ? `${r.total_seconds.toFixed(0)}s` : "—") },
            ].map((row) => (
              <tr key={row.label} className="border-b border-grid/60">
                <Td className="text-[11px] uppercase tracking-wider text-muted">{row.label}</Td>
                {chosen.map((r) => (
                  <Td key={r.id} mono>
                    {row.get(r)}
                  </Td>
                ))}
              </tr>
            ))}
          </Table>

          <div className="mt-4">
            <Note>
              Comparing Sharpe across runs is only meaningful when the runs share a universe, a
              period and a label horizon. Runs on different universes or windows are different
              experiments, and the higher number is not the better strategy. Check the plan on each
              run before drawing a conclusion — and note that each additional strategy compared
              raises the multiple-testing bar that any of them must clear.
            </Note>
          </div>
        </Card>
      )}

      {selected.length === 1 && (
        <Note>Select at least one more run to compare.</Note>
      )}
    </div>
  );
}
