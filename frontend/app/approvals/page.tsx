"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { ApiError, decideRun, getApprovals, getUser } from "../lib/api";
import type { ApprovalItem } from "../lib/types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Tabs,
  num,
  pct,
  relTime,
  verdictTone,
} from "../components/ui";

export default function ApprovalsPage() {
  const [items, setItems] = useState<ApprovalItem[]>([]);
  const [status, setStatus] = useState("pending");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, string>>({});
  const signedIn = typeof window !== "undefined" && !!getUser();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await getApprovals(status);
      setItems(r.items);
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setLoading(false);
    }
  }, [status]);

  useEffect(() => {
    load();
  }, [load]);

  const decide = async (runId: string, decision: "approved" | "rejected") => {
    setBusy(runId);
    try {
      await decideRun(runId, decision, feedback[runId] ?? "");
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-display text-2xl text-navy">Approval queue</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-muted">
          Every completed run stops here. A research result becomes decision-grade only when a
          named person accepts it — the Critic can recommend, but it cannot approve, and neither
          can the model that produced the work.
        </p>
      </div>

      {!signedIn && (
        <Note>
          You are not signed in. You can review the queue, but recording a decision requires an
          account so the approval is attributable to a person.
        </Note>
      )}

      <Tabs
        tabs={[
          { id: "pending", label: "Pending" },
          { id: "approved", label: "Approved" },
          { id: "rejected", label: "Rejected" },
        ]}
        active={status}
        onChange={setStatus}
      />

      {loading ? (
        <Spinner label="Loading queue…" />
      ) : error ? (
        <ErrorNote>{error}</ErrorNote>
      ) : items.length === 0 ? (
        <Empty>Nothing {status}.</Empty>
      ) : (
        <div className="space-y-4">
          {items.map((a) => (
            <Card
              key={a.approval_id}
              title={
                <Link href={`/runs/${a.run_id}`} className="hover:underline">
                  {a.question}
                </Link>
              }
              subtitle={`Run ${a.run_id.slice(0, 8)} · ${relTime(a.created_at)}`}
              right={
                a.critic_verdict ? (
                  <Badge tone={verdictTone(a.critic_verdict)}>{a.critic_verdict}</Badge>
                ) : undefined
              }
            >
              <div className="grid gap-3 sm:grid-cols-4">
                <div className="border border-grid bg-paper/60 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted">Sharpe</div>
                  <div className="font-mono text-lg">{num(a.sharpe, 2)}</div>
                </div>
                <div className="border border-grid bg-paper/60 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted">CAGR</div>
                  <div className="font-mono text-lg">{pct(a.cagr)}</div>
                </div>
                <div className="border border-grid bg-paper/60 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted">Max DD</div>
                  <div className="font-mono text-lg text-loss">{pct(a.max_drawdown)}</div>
                </div>
                <div className="border border-grid bg-paper/60 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted">Best model</div>
                  <div className="font-mono text-sm">{a.best_model ?? "—"}</div>
                </div>
              </div>

              {a.critic_summary && (
                <div className="mt-3 text-xs text-muted">
                  <span className="text-loss">{a.critic_summary.n_errors ?? 0} errors</span> ·{" "}
                  <span className="text-warn">{a.critic_summary.n_warnings ?? 0} warnings</span>
                  {a.critic_summary.recommended_revisions?.length > 0 && (
                    <span> · recommends: {a.critic_summary.recommended_revisions.join(", ")}</span>
                  )}
                </div>
              )}

              {status === "pending" ? (
                <div className="mt-4 space-y-2">
                  <textarea
                    value={feedback[a.run_id] ?? ""}
                    onChange={(e) => setFeedback((f) => ({ ...f, [a.run_id]: e.target.value }))}
                    rows={2}
                    placeholder="Reviewer notes — what you checked and what you are accepting."
                    className="w-full border border-grid bg-white px-3 py-2 text-xs outline-none focus:border-navy"
                  />
                  <div className="flex gap-2">
                    <Button
                      onClick={() => decide(a.run_id, "approved")}
                      disabled={busy === a.run_id || !signedIn}
                    >
                      Approve
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => decide(a.run_id, "rejected")}
                      disabled={busy === a.run_id || !signedIn}
                    >
                      Reject
                    </Button>
                    <Link
                      href={`/runs/${a.run_id}`}
                      className="inline-flex items-center border border-grid px-3 py-1.5 text-xs uppercase tracking-wider text-navy hover:bg-navy/5"
                    >
                      Review in full
                    </Link>
                  </div>
                </div>
              ) : (
                <p className="mt-3 text-xs text-muted">
                  {a.decided_by ? (
                    <>
                      {status} by <strong>{a.decided_by}</strong> {relTime(a.decided_at)}
                    </>
                  ) : (
                    "—"
                  )}
                </p>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
