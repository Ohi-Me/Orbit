"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ApiError, getAgentHealth, getQueue, getRunsSummary } from "../lib/api";
import type { AgentHealth } from "../lib/types";
import {
  Badge,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Stat,
  Table,
  Td,
  int,
  num,
  pct,
} from "../components/ui";

export default function MonitoringPage() {
  const [agents, setAgents] = useState<AgentHealth[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [queue, setQueue] = useState<any>(null);
  const [slowest, setSlowest] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [a, s, q] = await Promise.all([getAgentHealth(30), getRunsSummary(30), getQueue()]);
        setAgents(a.agents);
        setSlowest(a.slowest_agent);
        setSummary(s);
        setQueue(q);
        setError(null);
      } catch (e) {
        setError(e instanceof ApiError ? e.detail : String(e));
      } finally {
        setLoading(false);
      }
    };
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  if (loading) return <Spinner label="Loading monitoring…" />;
  if (error) return <ErrorNote>{error}</ErrorNote>;

  const totalTokens = agents.reduce((a, x) => a + x.llm_input_tokens + x.llm_output_tokens, 0);
  const totalCalls = agents.reduce((a, x) => a + x.llm_calls, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl text-navy">Monitoring</h1>
        <p className="mt-1 text-xs text-muted">
          Which agent is slow, which is failing, what runs cost, and how the queue is doing.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Stat label="Runs (30d)" value={int(summary?.n_runs)} />
        <Stat
          label="Approval rate"
          value={summary?.approval_rate !== null ? pct(summary?.approval_rate, 0) : "—"}
          hint="100% usually means rubber-stamping"
        />
        <Stat label="Median runtime" value={`${num(summary?.duration_seconds?.median, 0)}s`} />
        <Stat label="p90 runtime" value={`${num(summary?.duration_seconds?.p90, 0)}s`} tone="warn" />
        <Stat
          label="Queue"
          value={`${queue?.n_running ?? 0} / ${queue?.max_workers ?? 0}`}
          hint={`${queue?.n_queued ?? 0} queued`}
        />
      </div>

      {summary?.note && <Note>{summary.note}</Note>}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Run outcomes" subtitle="By status and by Critic verdict">
          <div className="space-y-3">
            <div>
              <p className="mb-1 text-[10px] uppercase tracking-wider text-muted">Status</p>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(summary?.by_status ?? {}).map(([k, v]: [string, any]) => (
                  <span key={k} className="border border-grid px-2 py-1 text-[11px]">
                    {k.replace(/_/g, " ")} <span className="font-mono">{v}</span>
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="mb-1 text-[10px] uppercase tracking-wider text-muted">Critic verdict</p>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(summary?.by_critic_verdict ?? {}).map(([k, v]: [string, any]) => (
                  <span key={k} className="border border-grid px-2 py-1 text-[11px]">
                    {k} <span className="font-mono">{v}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </Card>

        <Card title="LLM cost" subtitle="Tokens attributed to the agents that spent them">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Calls" value={int(totalCalls)} />
            <Stat label="Tokens" value={int(totalTokens)} />
          </div>
          <div className="mt-3">
            <Note>
              Most of this platform is deterministic computation, not generation — the LLM is used
              for research planning and optional document synthesis only. Model comparison,
              backtesting, risk and portfolio construction never call a language model, which is
              why token cost stays near zero for a full run.
            </Note>
          </div>
        </Card>
      </div>

      <Card title="Agent latency" subtitle={slowest ? `Slowest: ${slowest}` : undefined}>
        {agents.length === 0 ? (
          <Empty>No agent executions recorded yet.</Empty>
        ) : (
          <>
            <div className="h-64 w-full">
              <ResponsiveContainer>
                <BarChart data={agents} margin={{ top: 5, right: 10, left: 0, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="#D7D9D0" />
                  <XAxis
                    dataKey="agent"
                    angle={-45}
                    textAnchor="end"
                    interval={0}
                    tick={{ fontSize: 9, fill: "#5B6259" }}
                  />
                  <YAxis tick={{ fontSize: 10, fill: "#5B6259" }} unit="s" />
                  <Tooltip
                    contentStyle={{ fontSize: 11, borderRadius: 0, border: "1px solid #D7D9D0" }}
                    formatter={(v: any) => `${Number(v).toFixed(2)}s`}
                  />
                  <Bar dataKey="avg_seconds" name="Avg seconds">
                    {agents.map((a, i) => (
                      <Cell key={i} fill={a.failure_rate > 0 ? "#B23B2E" : "#1C2B45"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <Table head={["Agent", "Runs", "Avg", "Max", "Failure rate", "Degraded rate", "Statuses"]}>
              {agents.map((a) => (
                <tr key={a.agent} className="border-b border-grid/60">
                  <Td mono>{a.agent}</Td>
                  <Td mono>{a.n_executions}</Td>
                  <Td mono>{a.avg_seconds.toFixed(2)}s</Td>
                  <Td mono>{a.max_seconds.toFixed(2)}s</Td>
                  <Td mono className={a.failure_rate > 0 ? "text-loss" : "text-gain"}>
                    {pct(a.failure_rate, 1)}
                  </Td>
                  <Td mono className={a.degraded_rate > 0 ? "text-warn" : ""}>
                    {pct(a.degraded_rate, 1)}
                  </Td>
                  <Td>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(a.status_counts).map(([s, n]) => (
                        <Badge
                          key={s}
                          tone={s === "ok" ? "ok" : s === "failed" ? "fail" : s === "degraded" ? "warn" : "muted"}
                        >
                          {s} {n}
                        </Badge>
                      ))}
                    </div>
                  </Td>
                </tr>
              ))}
            </Table>
          </>
        )}
      </Card>

      {queue && (
        <Card title="Execution queue">
          <Note>{queue.note}</Note>
        </Card>
      )}
    </div>
  );
}
