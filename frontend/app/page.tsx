"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  getPipelineTopology,
  getRun,
  getRunSteps,
  listRuns,
  submitRun,
} from "./lib/api";
import type { RunStep, RunSummary } from "./lib/types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  num,
  relTime,
  statusTone,
  verdictTone,
} from "./components/ui";

const EXAMPLES = [
  "Do value and quality features improve risk-adjusted returns in large-cap US equities?",
  "Does a sequence model beat gradient boosting on 21-day return direction?",
  "Do macro regime indicators improve model performance in financials?",
  "What risk factors does management discuss in recent filings?",
];

const GROUP_LABEL: Record<string, string> = {
  planning: "Planning",
  ingestion: "Data ingestion",
  features: "Feature engineering",
  modelling: "Modelling",
  evaluation: "Evaluation",
  validation: "Validation",
  reporting: "Reporting",
};

const GROUP_ORDER = ["planning", "ingestion", "features", "modelling", "evaluation", "validation", "reporting"];

export default function PipelinesPage() {
  const router = useRouter();
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [useLlm, setUseLlm] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeStatus, setActiveStatus] = useState<string>("");
  const [steps, setSteps] = useState<RunStep[]>([]);
  const [recent, setRecent] = useState<RunSummary[]>([]);
  const [topology, setTopology] = useState<any>(null);

  const refreshRecent = useCallback(async () => {
    try {
      setRecent((await listRuns(6)).runs);
    } catch {
      /* connectivity is surfaced by the shell badge */
    }
  }, []);

  useEffect(() => {
    refreshRecent();
    getPipelineTopology()
      .then(setTopology)
      .catch(() => undefined);
  }, [refreshRecent]);

  useEffect(() => {
    if (!activeRunId) return;
    let alive = true;
    const tick = async () => {
      try {
        const [run, st] = await Promise.all([getRun(activeRunId), getRunSteps(activeRunId)]);
        if (!alive) return;
        setActiveStatus(run.status);
        setSteps(st.steps);
        if (run.status === "awaiting_approval" || run.status === "failed") {
          refreshRecent();
          return true;
        }
      } catch {
        /* transient poll failures are expected during execution */
      }
      return false;
    };
    tick();
    const id = setInterval(async () => {
      if (await tick()) clearInterval(id);
    }, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [activeRunId, refreshRecent]);

  const onSubmit = async () => {
    setError(null);
    setSubmitting(true);
    setSteps([]);
    try {
      const res = await submitRun(question, {}, useLlm);
      setActiveRunId(res.run_id);
      setActiveStatus("queued");
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const executed = new Set(steps.map((s) => s.agent));
  const running = activeStatus === "queued" || activeStatus === "running";
  const stages: any[] = topology?.stages ?? [];

  return (
    <div className="space-y-6">
      <div className="max-w-3xl">
        <h1 className="font-display text-2xl leading-snug text-navy">
          An end-to-end ML pipeline you can point at a question
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Fourteen agents orchestrated as a graph with a validation feedback loop: ingest and
          validate data, engineer point-in-time features, train and compare four model families
          under purged walk-forward validation, evaluate with a real cost model, then criticise the
          result and stop at a human gate. The application domain here is financial markets.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <div className="space-y-6">
          <Card
            title="Run a pipeline"
            subtitle="A planning agent turns your question into a schema-validated configuration, then the graph executes it."
          >
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              className="w-full resize-y border border-grid bg-white px-3 py-2 font-body text-sm outline-none focus:border-navy"
            />

            <div className="mt-2 flex flex-wrap gap-1.5">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setQuestion(ex)}
                  className="border border-grid px-2 py-1 text-left text-[11px] text-muted transition-colors hover:border-navy hover:text-navy"
                >
                  {ex.length > 58 ? `${ex.slice(0, 58)}…` : ex}
                </button>
              ))}
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-xs text-muted">
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(e) => setUseLlm(e.target.checked)}
                  className="accent-navy"
                />
                LLM planner (falls back to deterministic routing without an API key)
              </label>
              <Button onClick={onSubmit} disabled={submitting || running || question.trim().length < 10}>
                {submitting ? "Submitting…" : running ? "Pipeline running" : "Execute pipeline"}
              </Button>
            </div>

            {error && (
              <div className="mt-3">
                <ErrorNote>{error}</ErrorNote>
              </div>
            )}

            <div className="mt-4">
              <Note>
                Execution takes several minutes and runs asynchronously on a worker pool — the
                request returns a job id immediately and every state transition is persisted, so
                nothing is lost to a closed tab or a restart.
              </Note>
            </div>
          </Card>

          {activeRunId && (
            <Card
              title="Execution trace"
              subtitle={`Job ${activeRunId.slice(0, 8)}`}
              right={
                <div className="flex items-center gap-2">
                  <Badge tone={statusTone(activeStatus)}>{activeStatus.replace(/_/g, " ")}</Badge>
                  {(activeStatus === "awaiting_approval" || activeStatus === "failed") && (
                    <Button variant="ghost" onClick={() => router.push(`/runs/${activeRunId}`)}>
                      Open
                    </Button>
                  )}
                </div>
              }
            >
              {running && steps.length === 0 && <Spinner label="Scheduling…" />}

              <ol className="space-y-1">
                {stages.map((stage) => {
                  const runs = steps.filter((s) => s.agent === stage.agent);
                  const last = runs[runs.length - 1];
                  const done = executed.has(stage.agent);
                  return (
                    <li
                      key={stage.id}
                      className={`flex items-center gap-3 border-l-2 py-1 pl-3 text-xs ${
                        done ? "border-navy" : "border-grid"
                      }`}
                      title={stage.does}
                    >
                      <span className="w-24 shrink-0 text-[10px] uppercase tracking-wider text-muted">
                        {GROUP_LABEL[stage.group] ?? stage.group}
                      </span>
                      <span className={`flex-1 font-mono ${done ? "text-ink" : "text-muted"}`}>
                        {stage.agent}
                        {runs.length > 1 && <span className="ml-1 text-muted">×{runs.length}</span>}
                      </span>
                      {last && (
                        <>
                          <span className="font-mono text-[11px] text-muted">
                            {last.seconds.toFixed(1)}s
                          </span>
                          <Badge tone={statusTone(last.status)}>{last.status}</Badge>
                        </>
                      )}
                    </li>
                  );
                })}
              </ol>

              {steps.some((s) => s.agent === "plan_revision") && (
                <div className="mt-3 border border-navy/30 bg-navy/5 px-3 py-2 text-[11px] text-navy">
                  The validation stage rejected the first attempt, revised the configuration, and
                  re-entered the graph. That cycle is why some stages appear more than once.
                </div>
              )}
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Card
            title="Recent experiments"
            right={
              <Link href="/runs" className="text-xs uppercase tracking-wider text-navy hover:underline">
                all
              </Link>
            }
          >
            {recent.length === 0 ? (
              <Empty>No experiments yet.</Empty>
            ) : (
              <ul className="divide-y divide-grid">
                {recent.map((r) => (
                  <li key={r.id} className="py-2">
                    <Link href={`/runs/${r.id}`} className="group block">
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-xs leading-snug text-ink group-hover:underline">
                          {r.question.length > 78 ? `${r.question.slice(0, 78)}…` : r.question}
                        </p>
                        <Badge tone={statusTone(r.status)}>{r.status.replace(/_/g, " ")}</Badge>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
                        <span>{relTime(r.created_at)}</span>
                        {r.best_model && <span className="font-mono">{r.best_model}</span>}
                        {r.critic_verdict && (
                          <Badge tone={verdictTone(r.critic_verdict)}>
                            {r.critic_verdict.split("_")[0]}
                          </Badge>
                        )}
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Engineering guarantees" subtitle="What the pipeline enforces, not what it hopes">
            <ul className="space-y-2 text-xs leading-relaxed text-muted">
              <li>
                <strong className="text-ink">No temporal leakage.</strong> Cross-validation purges
                training samples whose label window overlaps the test period and applies an
                embargo. Features are point-in-time: fundamentals key on filing date, macro series
                are shifted by their publication lag.
              </li>
              <li>
                <strong className="text-ink">Corrected inference.</strong> Overlapping label
                windows inflate significance, so every test uses HAC standard errors and every
                model-selection claim is deflated for the number of configurations tried.
              </li>
              <li>
                <strong className="text-ink">Fair model comparison.</strong> Linear, gradient
                boosting and two deep sequence architectures see identical folds, features and
                scaling, and are scored against the base rate rather than on raw accuracy.
              </li>
              <li>
                <strong className="text-ink">Grounded generation.</strong> Retrieval filters on
                metadata before ranking, reranks with a cross-encoder, and verifies every figure in
                an answer against the source text.
              </li>
              <li>
                <strong className="text-ink">Honest degradation.</strong> Capabilities are probed,
                not declared. A missing dependency falls back to a labelled substitute and the run
                records that it did.
              </li>
            </ul>
          </Card>

          {topology && (
            <Card title="Pipeline topology" subtitle={`${topology.orchestrator} · ${stages.length} stages`}>
              <div className="space-y-2">
                {GROUP_ORDER.filter((g) => stages.some((s) => s.group === g)).map((g) => (
                  <div key={g}>
                    <p className="text-[10px] uppercase tracking-wider text-muted">
                      {GROUP_LABEL[g] ?? g}
                    </p>
                    <div className="mt-0.5 flex flex-wrap gap-1">
                      {stages
                        .filter((s) => s.group === g)
                        .map((s) => (
                          <span
                            key={s.id}
                            title={`${s.does}\n\nGuarantee: ${s.guarantees}`}
                            className="border border-grid bg-white/60 px-1.5 py-0.5 font-mono text-[10px] text-navy"
                          >
                            {s.id}
                            {s.uses_llm && <span className="ml-1 text-warn">llm</span>}
                          </span>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3">
                <Note>{topology.terminal_note}</Note>
              </div>
              <div className="mt-2">
                <Link href="/agents" className="text-xs uppercase tracking-wider text-navy hover:underline">
                  full agent registry →
                </Link>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
