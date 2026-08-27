"use client";

import { useEffect, useState } from "react";

import { ApiError, getAgentHealth, getPipelineTopology } from "../lib/api";
import type { AgentHealth } from "../lib/types";
import {
  Badge,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Stat,
  num,
  pct,
} from "../components/ui";

const GROUP_LABEL: Record<string, string> = {
  planning: "Planning",
  ingestion: "Data ingestion",
  features: "Feature engineering",
  modelling: "Modelling",
  evaluation: "Evaluation",
  validation: "Validation",
  reporting: "Reporting",
};

const GROUP_ORDER = [
  "planning",
  "ingestion",
  "features",
  "modelling",
  "evaluation",
  "validation",
  "reporting",
];

export default function AgentsPage() {
  const [topology, setTopology] = useState<any>(null);
  const [health, setHealth] = useState<AgentHealth[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getPipelineTopology(), getAgentHealth(30)])
      .then(([t, h]) => {
        setTopology(t);
        setHealth(h.agents);
      })
      .catch((e) => setError(e instanceof ApiError ? e.detail : String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner label="Loading agent registry…" />;
  if (error) return <ErrorNote>{error}</ErrorNote>;

  const stages: any[] = topology?.stages ?? [];
  const healthByAgent = Object.fromEntries(health.map((h) => [h.agent, h]));
  const llmStages = stages.filter((s) => s.uses_llm);

  return (
    <div className="space-y-6">
      <div className="max-w-3xl">
        <h1 className="font-display text-2xl text-navy">Agent registry</h1>
        <p className="mt-1.5 text-sm leading-relaxed text-muted">
          Every stage in the pipeline, what it is responsible for, what it guarantees, whether it
          calls a language model, and how it has actually behaved in production.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-4">
        <Stat label="Agents" value={stages.length} />
        <Stat label="Orchestrator" value={<span className="text-sm">{topology?.orchestrator}</span>} />
        <Stat
          label="LLM-backed"
          value={`${llmStages.length} / ${stages.length}`}
          hint="Everything else is deterministic"
        />
        <Stat
          label="Feedback loop"
          value={topology?.has_cycle ? "yes" : "no"}
          hint={`max ${topology?.max_revisions} revision`}
        />
      </div>

      <Card title="Control flow">
        <p className="text-xs leading-relaxed text-muted">{topology?.cycle_description}</p>
        <div className="mt-3 flex flex-wrap items-center gap-1">
          {stages.map((s, i) => (
            <span key={s.id} className="flex items-center gap-1">
              <span className="border border-grid bg-white/70 px-1.5 py-0.5 font-mono text-[10px] text-navy">
                {s.id}
              </span>
              {i < stages.length - 1 && <span className="text-muted">→</span>}
            </span>
          ))}
        </div>
        <div className="mt-3">
          <Note>{topology?.terminal_note}</Note>
        </div>
      </Card>

      {GROUP_ORDER.filter((g) => stages.some((s) => s.group === g)).map((group) => (
        <Card key={group} title={GROUP_LABEL[group] ?? group}>
          <div className="space-y-3">
            {stages
              .filter((s) => s.group === group)
              .map((s) => {
                const h = healthByAgent[s.agent];
                return (
                  <div key={s.id} className="border border-grid bg-white/60 px-3 py-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs text-navy">{s.agent}</span>
                      {s.uses_llm ? (
                        <Badge tone="warn">llm</Badge>
                      ) : (
                        <Badge tone="ok">deterministic</Badge>
                      )}
                      {h && (
                        <span className="ml-auto flex items-center gap-2 font-mono text-[10px] text-muted">
                          <span>{h.n_executions} runs</span>
                          <span>{h.avg_seconds.toFixed(1)}s avg</span>
                          {h.failure_rate > 0 ? (
                            <Badge tone="fail">{pct(h.failure_rate, 0)} fail</Badge>
                          ) : (
                            <Badge tone="ok">0% fail</Badge>
                          )}
                        </span>
                      )}
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-ink">{s.does}</p>
                    <p className="mt-1 border-l-2 border-navy/25 pl-2 text-[11px] leading-relaxed text-muted">
                      <strong className="text-ink">Guarantees:</strong> {s.guarantees}
                    </p>
                  </div>
                );
              })}
          </div>
        </Card>
      ))}

      <Card title="Where language models are and are not used">
        <div className="space-y-2 text-xs leading-relaxed text-muted">
          <p>
            <strong className="text-ink">
              {llmStages.length} of {stages.length} stages call a language model.
            </strong>{" "}
            The planner ({llmStages.map((s: any) => s.id).join(", ")}) — and nothing else.
          </p>
          <p>
            Model training, validation, backtesting, risk decomposition, portfolio construction and
            the critic are all deterministic computation. The report is assembled from structured
            upstream output by a template, so no language model ever writes a number.
          </p>
          <p>
            This is a deliberate boundary, not an incidental one. An LLM chooses{" "}
            <em>what to investigate</em>; it never decides whether a result is significant, never
            sizes a position, and never approves its own work. Those paths have fixed rules and a
            human gate.
          </p>
        </div>
      </Card>

      {health.length > 0 && (
        <Card title="Reliability (30 days)">
          <div className="space-y-1.5">
            {health.map((h) => (
              <div key={h.agent} className="flex items-center gap-3 text-xs">
                <span className="w-56 shrink-0 truncate font-mono text-[11px]">{h.agent}</span>
                <div className="h-2 flex-1 overflow-hidden border border-grid bg-paper">
                  <div
                    className="h-full bg-navy"
                    style={{
                      width: `${Math.min(100, (h.avg_seconds / Math.max(...health.map((x) => x.avg_seconds), 1)) * 100)}%`,
                    }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right font-mono text-[11px] text-muted">
                  {h.avg_seconds.toFixed(1)}s
                </span>
                <span className="w-14 shrink-0 text-right font-mono text-[11px]">
                  {h.n_executions}×
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {health.length === 0 && <Empty>No execution history yet — run a pipeline first.</Empty>}
    </div>
  );
}
