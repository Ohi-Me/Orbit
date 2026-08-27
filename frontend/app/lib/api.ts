"use client";

import type {
  AgentHealth,
  ApprovalItem,
  DocumentRow,
  Health,
  ModelFold,
  PortfolioBook,
  RetrievalHit,
  RunDetail,
  RunStep,
  RunSummary,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

const TOKEN_KEY = "quant_platform_token";
const USER_KEY = "quant_platform_user";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setSession(token: string, user: unknown) {
  window.localStorage.setItem(TOKEN_KEY, token);
  window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  window.dispatchEvent(new Event("quant-auth-change"));
}

export function clearSession() {
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event("quant-auth-change"));
}

export function getUser(): { id: string; email: string; display_name: string } | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init.headers as Record<string, string>) || {}),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    // A network failure here has two very different causes, and the wrong
    // message sends you looking in the wrong place for an hour.
    //
    // NEXT_PUBLIC_* values are inlined by Next at BUILD time, not read at
    // runtime. So a deployed site built without NEXT_PUBLIC_API_BASE ships
    // the localhost default baked into its JavaScript, and every visitor's
    // browser then tries to reach a backend on their OWN machine. The symptom
    // is identical to "the server isn't running", but the fix is completely
    // different: set the variable and REBUILD.
    const pageIsLocal =
      typeof window !== "undefined" &&
      /^(localhost|127\.0\.0\.1|\[::1\])$/i.test(window.location.hostname);
    const apiIsLocal = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(API_BASE);

    if (!pageIsLocal && apiIsLocal) {
      throw new ApiError(
        0,
        `This site is trying to reach ${API_BASE} — a backend on your own computer, ` +
          `not a deployed one. It was built without NEXT_PUBLIC_API_BASE set. ` +
          `Next inlines NEXT_PUBLIC_* at build time, so set it to the API's public ` +
          `URL and rebuild — changing it after the build has no effect.`
      );
    }

    throw new ApiError(
      0,
      `Cannot reach the backend at ${API_BASE}. Start it with: uvicorn app.main:app --port 8000`
    );
  }

  if (res.status === 204) return undefined as T;

  const text = await res.text();
  if (!res.ok) {
    let detail = text;
    try {
      detail = JSON.parse(text).detail ?? text;
    } catch {
      /* keep raw text */
    }
    if (res.status === 401) clearSession();
    throw new ApiError(res.status, String(detail));
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return JSON.parse(text) as T;
  return text as unknown as T;
}

// ---------------------------------------------------------------- system
export const getHealth = () => request<Health>("/api/health");
export const getPresets = () =>
  request<{
    universe_presets: Record<string, { tickers: string[]; size: number }>;
    factor_families: Record<string, string[]>;
  }>("/api/config/presets");

// ------------------------------------------------------------------ auth
export const signup = (email: string, password: string, display_name = "") =>
  request<{ access_token: string; user: any }>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, display_name }),
  });

export const login = (email: string, password: string) =>
  request<{ access_token: string; user: any }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });

export const me = () => request<{ authenticated: boolean; user: any }>("/api/auth/me");

// ------------------------------------------------------------------ runs
export const submitRun = (question: string, overrides: Record<string, unknown> = {}, useLlm = true) =>
  request<{ run_id: string; status: string }>("/api/runs", {
    method: "POST",
    body: JSON.stringify({ question, overrides, use_llm_planner: useLlm }),
  });

export const listRuns = (limit = 30, status?: string) =>
  request<{ runs: RunSummary[] }>(
    `/api/runs?limit=${limit}${status ? `&status=${status}` : ""}`
  );

export const getRun = (id: string) => request<RunDetail>(`/api/runs/${id}`);
export const getRunSteps = (id: string) =>
  request<{ steps: RunStep[]; total_seconds: number; n_failed: number; n_degraded: number }>(
    `/api/runs/${id}/steps`
  );
export const getRunModels = (id: string) =>
  request<{ folds: ModelFold[] }>(`/api/runs/${id}/models`);
export const getRunLineage = (id: string) =>
  request<{ capabilities_at_execution: any; snapshots: any[] }>(`/api/runs/${id}/lineage`);
export const getRunReport = (id: string) => request<string>(`/api/runs/${id}/report`);
export const deleteRun = (id: string) => request<void>(`/api/runs/${id}`, { method: "DELETE" });
export const getQueue = () => request<any>("/api/runs/queue");

export const decideRun = (id: string, decision: "approved" | "rejected", feedback = "") =>
  request<any>(`/api/runs/${id}/approve`, {
    method: "POST",
    body: JSON.stringify({ decision, feedback }),
  });

// ------------------------------------------------------------- documents
export const listDocuments = (ticker?: string) =>
  request<{ documents: DocumentRow[]; total: number }>(
    `/api/documents${ticker ? `?ticker=${ticker}` : ""}`
  );

export const corpusStats = () => request<any>("/api/documents/stats");

export const ingestEdgar = (ticker: string, limit = 2) =>
  request<any>("/api/documents/ingest/edgar", {
    method: "POST",
    body: JSON.stringify({ ticker, forms: ["10-K", "10-Q"], limit }),
  });

export const searchDocuments = (payload: {
  query: string;
  tickers?: string[];
  doc_types?: string[];
  top_k?: number;
}) =>
  request<{ status: string; hits: RetrievalHit[]; retrieval_mode: string; n_candidates: number }>(
    "/api/documents/search",
    { method: "POST", body: JSON.stringify(payload) }
  );

export const askDocuments = (payload: { query: string; tickers?: string[]; top_k?: number }) =>
  request<any>("/api/documents/ask", { method: "POST", body: JSON.stringify(payload) });

export const deleteDocument = (id: string) =>
  request<void>(`/api/documents/${id}`, { method: "DELETE" });

// ------------------------------------------------------------ monitoring
export const getApprovals = (status = "pending") =>
  request<{ items: ApprovalItem[]; n_pending: number }>(
    `/api/monitoring/approvals?status=${status}`
  );

export const getAgentHealth = (days = 7) =>
  request<{ agents: AgentHealth[]; slowest_agent: string | null; most_failure_prone: string | null }>(
    `/api/monitoring/agents?days=${days}`
  );

export const getRunsSummary = (days = 30) => request<any>(`/api/monitoring/runs/summary?days=${days}`);

// ------------------------------------------------------------- portfolio
export const listBooks = () => request<{ books: PortfolioBook[] }>("/api/portfolio/books");

export const createBook = (name: string, notional: number, positions: Record<string, number>) =>
  request<PortfolioBook>("/api/portfolio/books", {
    method: "POST",
    body: JSON.stringify({ name, notional, positions }),
  });

export const adoptWeights = (run_id: string, method: string, book_name: string) =>
  request<PortfolioBook>("/api/portfolio/books/adopt", {
    method: "POST",
    body: JSON.stringify({ run_id, method, book_name }),
  });

export const deleteBook = (id: string) =>
  request<void>(`/api/portfolio/books/${id}`, { method: "DELETE" });

// -------------------------------------------------------------- ml platform
export const getLeaderboard = () =>
  request<{
    models: {
      model: string;
      n_folds: number;
      n_runs: number;
      mean_accuracy: number | null;
      mean_auc: number | null;
      mean_information_coefficient: number | null;
      mean_signal_return: number | null;
      accuracy_range: [number | null, number | null];
    }[];
    n_experiments: number;
    ranked_by: string;
    note: string;
  }>("/api/ml/leaderboard");

export const getPipelineTopology = () =>
  request<{
    orchestrator: string;
    has_cycle: boolean;
    cycle_description: string;
    max_revisions: number;
    stages: {
      id: string;
      agent: string;
      group: string;
      does: string;
      guarantees: string;
      uses_llm: boolean;
    }[];
    edges: [string, string][];
    terminal_state: string;
    terminal_note: string;
  }>("/api/ml/pipeline");
