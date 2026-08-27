"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ApiError, getLeaderboard } from "../lib/api";
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
  num,
} from "../components/ui";

const ARCHITECTURE: Record<string, { family: string; note: string }> = {
  logistic_regression: {
    family: "Linear",
    note: "The baseline every other model must beat. Cheap, stable, and interpretable — if it wins, the added complexity was not worth its cost.",
  },
  xgboost: {
    family: "Gradient boosting",
    note: "Non-linear tabular learner. Sees one date at a time; captures interactions between features but not the path that produced them.",
  },
  lstm: {
    family: "Recurrent sequence",
    note: "Reads a 20-day window of feature history. Tests whether the trajectory of a feature carries information its current level does not.",
  },
  transformer: {
    family: "Attention sequence",
    note: "Small encoder with learned positional embeddings and mean pooling. Same hypothesis as the LSTM, different inductive bias.",
  },
};

export default function ModelsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getLeaderboard()
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.detail : String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner label="Loading leaderboard…" />;
  if (error) return <ErrorNote>{error}</ErrorNote>;

  const models = data?.models ?? [];
  const baseline = models.find((m: any) => m.model === "logistic_regression");

  return (
    <div className="space-y-6">
      <div className="max-w-3xl">
        <h1 className="font-display text-2xl text-navy">Model leaderboard</h1>
        <p className="mt-1.5 text-sm leading-relaxed text-muted">
          Aggregated across every fold of every experiment. The question this screen answers is not
          &ldquo;which model won once&rdquo; but &ldquo;does an architecture hold up across
          experiments, or did it get a lucky fold&rdquo;.
        </p>
      </div>

      {models.length === 0 ? (
        <Empty>No model results yet — run a pipeline first.</Empty>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Architectures compared" value={models.length} />
            <Stat label="Experiments" value={data.n_experiments} />
            <Stat
              label="Total folds"
              value={models.reduce((a: number, m: any) => a + m.n_folds, 0)}
            />
          </div>

          <Card
            title="Mean information coefficient by architecture"
            subtitle="IC measures ranking skill directly — the thing a cross-sectional strategy actually monetizes"
          >
            <div className="h-64 w-full">
              <ResponsiveContainer>
                <BarChart data={models} margin={{ top: 5, right: 10, left: 0, bottom: 50 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="#D7D9D0" />
                  <XAxis
                    dataKey="model"
                    angle={-30}
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
                  <Bar dataKey="mean_information_coefficient" name="Mean IC">
                    {models.map((m: any, i: number) => (
                      <Cell
                        key={i}
                        fill={(m.mean_information_coefficient ?? 0) > 0 ? "#1F7A5C" : "#B23B2E"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <Note>{data.note}</Note>
          </Card>

          <Card title="Detail">
            <Table
              head={["Model", "Family", "Runs", "Folds", "Mean IC", "Mean AUC", "Mean accuracy", "Accuracy range", "vs baseline"]}
            >
              {models.map((m: any) => {
                const arch = ARCHITECTURE[m.model];
                const delta =
                  baseline && m.model !== "logistic_regression"
                    ? (m.mean_information_coefficient ?? 0) -
                      (baseline.mean_information_coefficient ?? 0)
                    : null;
                const spread =
                  m.accuracy_range[0] !== null && m.accuracy_range[1] !== null
                    ? m.accuracy_range[1] - m.accuracy_range[0]
                    : null;
                return (
                  <tr key={m.model} className="border-b border-grid/60">
                    <Td mono>{m.model}</Td>
                    <Td>
                      <Badge tone="info">{arch?.family ?? "—"}</Badge>
                    </Td>
                    <Td mono>{m.n_runs}</Td>
                    <Td mono>{m.n_folds}</Td>
                    <Td
                      mono
                      className={
                        (m.mean_information_coefficient ?? 0) > 0 ? "text-gain" : "text-loss"
                      }
                    >
                      {num(m.mean_information_coefficient, 4)}
                    </Td>
                    <Td mono>{num(m.mean_auc, 4)}</Td>
                    <Td mono>{num(m.mean_accuracy, 4)}</Td>
                    <Td mono className={spread !== null && spread > 0.2 ? "text-warn" : "text-muted"}>
                      {num(m.accuracy_range[0], 3)}–{num(m.accuracy_range[1], 3)}
                    </Td>
                    <Td mono>
                      {delta === null ? (
                        <span className="text-muted">baseline</span>
                      ) : (
                        <span className={delta > 0 ? "text-gain" : "text-loss"}>
                          {delta > 0 ? "+" : ""}
                          {num(delta, 4)}
                        </span>
                      )}
                    </Td>
                  </tr>
                );
              })}
            </Table>
            <div className="mt-3">
              <Note>
                A wide accuracy range across folds is the signature of instability, not skill: a
                model that scores 0.42 on one fold and 0.67 on another has an unremarkable mean and
                no reliability. Read the range next to the mean, always.
              </Note>
            </div>
          </Card>

          <Card title="Architectures" subtitle="Why each one is in the comparison">
            <div className="grid gap-3 sm:grid-cols-2">
              {Object.entries(ARCHITECTURE).map(([key, a]) => {
                const m = models.find((x: any) => x.model === key);
                return (
                  <div key={key} className="border border-grid bg-white/60 px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs text-navy">{key}</span>
                      <Badge tone="muted">{a.family}</Badge>
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-muted">{a.note}</p>
                    {m && (
                      <p className="mt-1.5 font-mono text-[10px] text-muted">
                        IC {num(m.mean_information_coefficient, 4)} · {m.n_folds} folds ·{" "}
                        {m.n_runs} runs
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
            <div className="mt-3">
              <Note>
                Every architecture sees identical folds, identical features and identical scaling,
                so a difference between them is attributable to the model rather than to its
                preprocessing. The honest prior on a few thousand noisy samples is that the deep
                models will <em>not</em> beat gradient boosting — and reporting that clearly is a
                result, not a failure.
              </Note>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
