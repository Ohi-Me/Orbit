"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { ApiError, adoptWeights, deleteBook, getUser, listBooks, listRuns } from "../lib/api";
import type { PortfolioBook, RunSummary } from "../lib/types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Stat,
  Table,
  Td,
  num,
  pct,
  shortDate,
} from "../components/ui";

const COLORS = ["#1C2B45", "#28405F", "#1F7A5C", "#C97A2B", "#B23B2E", "#5B6259", "#8FA3B8", "#7FA98F"];
const METHODS = ["risk_parity", "min_variance", "max_diversification", "mean_variance", "equal_weight"];

export default function PortfolioPage() {
  const [books, setBooks] = useState<PortfolioBook[]>([]);
  const [approved, setApproved] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState("");
  const [method, setMethod] = useState("risk_parity");
  const [bookName, setBookName] = useState("Live book");
  const [busy, setBusy] = useState(false);
  const signedIn = typeof window !== "undefined" && !!getUser();

  const load = useCallback(async () => {
    try {
      const [b, r] = await Promise.all([listBooks(), listRuns(50, "approved")]);
      setBooks(b.books);
      setApproved(r.runs);
      if (!runId && r.runs.length) setRunId(r.runs[0].id);
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  const adopt = async () => {
    setBusy(true);
    setError(null);
    try {
      await adoptWeights(runId, method, bookName);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (loading) return <Spinner label="Loading books…" />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl text-navy">Portfolio books</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-muted">
          A book is the standing allocation a new signal gets judged against. Weights can only be
          adopted from an <strong>approved</strong> run — research output does not become a
          position without an explicit human decision.
        </p>
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}

      <Card title="Adopt weights from an approved run">
        {approved.length === 0 ? (
          <Empty>
            No approved runs yet. Approve a run from the{" "}
            <Link href="/approvals" className="text-navy underline">
              approval queue
            </Link>{" "}
            before adopting weights.
          </Empty>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex-1 min-w-[240px]">
              <span className="mb-1 block text-[10px] uppercase tracking-wider text-muted">Run</span>
              <select
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
                className="w-full border border-grid bg-white px-2 py-1.5 text-xs"
              >
                {approved.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id.slice(0, 8)} — {r.question.slice(0, 60)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="mb-1 block text-[10px] uppercase tracking-wider text-muted">Method</span>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="border border-grid bg-white px-2 py-1.5 text-xs"
              >
                {METHODS.map((m) => (
                  <option key={m} value={m}>
                    {m.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="mb-1 block text-[10px] uppercase tracking-wider text-muted">Book name</span>
              <input
                value={bookName}
                onChange={(e) => setBookName(e.target.value)}
                className="border border-grid bg-white px-2 py-1.5 text-xs"
              />
            </label>
            <Button onClick={adopt} disabled={busy || !runId || !signedIn}>
              {busy ? "Adopting…" : "Adopt weights"}
            </Button>
          </div>
        )}
        {!signedIn && (
          <div className="mt-3">
            <Note>Sign in to adopt weights — the action is recorded against your account.</Note>
          </div>
        )}
      </Card>

      {books.length === 0 ? (
        <Empty>No books yet.</Empty>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {books.map((b) => {
            const data = Object.entries(b.positions)
              .map(([ticker, weight]) => ({ name: ticker, value: Math.abs(weight) }))
              .sort((a, b2) => b2.value - a.value);
            return (
              <Card
                key={b.id}
                title={b.name}
                subtitle={`${b.n_positions} positions · ${b.method ?? "manual"} · ${shortDate(b.updated_at)}`}
                right={
                  <button
                    onClick={async () => {
                      await deleteBook(b.id);
                      load();
                    }}
                    className="text-[10px] uppercase tracking-wider text-muted hover:text-loss"
                  >
                    delete
                  </button>
                }
              >
                <div className="grid gap-3 sm:grid-cols-3">
                  <Stat label="Notional" value={`$${(b.notional / 1e6).toFixed(1)}M`} />
                  <Stat label="Gross" value={num(b.gross_exposure, 3)} />
                  <Stat label="Net" value={num(b.net_exposure, 3)} />
                </div>

                {data.length > 0 && (
                  <div className="mt-3 h-56 w-full">
                    <ResponsiveContainer>
                      <PieChart>
                        <Pie
                          data={data}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={45}
                          outerRadius={80}
                          paddingAngle={1}
                          label={(e: any) => e.name}
                          labelLine={false}
                        >
                          {data.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ fontSize: 11, borderRadius: 0, border: "1px solid #D7D9D0" }}
                          formatter={(v: any) => `${(Number(v) * 100).toFixed(2)}%`}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                )}

                <Table head={["Ticker", "Weight"]} dense>
                  {data.map((d) => (
                    <tr key={d.name} className="border-b border-grid/60">
                      <Td mono>{d.name}</Td>
                      <Td mono>{pct(b.positions[d.name])}</Td>
                    </tr>
                  ))}
                </Table>

                {b.source_run_id && (
                  <p className="mt-2 text-[11px] text-muted">
                    Adopted from run{" "}
                    <Link href={`/runs/${b.source_run_id}`} className="text-navy underline">
                      {b.source_run_id.slice(0, 8)}
                    </Link>
                  </p>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
