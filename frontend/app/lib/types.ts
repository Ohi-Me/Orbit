export type RunStatus =
  | "queued"
  | "running"
  | "awaiting_approval"
  | "approved"
  | "rejected"
  | "failed";

export interface Capabilities {
  database: string;
  live_market_data: boolean;
  live_filings: boolean;
  live_macro: boolean;
  llm: boolean;
  neural_embeddings: boolean;
  reranker: boolean;
  finbert_sentiment: boolean;
  deep_learning: boolean;
  experiment_tracking: boolean;
  graph_orchestration: boolean;
  jwt_secret_is_default: boolean;
}

export interface Health {
  status: string;
  version: string;
  capabilities: Capabilities;
  degraded_capabilities: string[];
  fidelity: "full" | "degraded";
  warnings: string[];
  note: string;
}

export interface RunSummary {
  id: string;
  question: string;
  status: RunStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  total_seconds?: number | null;
  sharpe?: number | null;
  cagr?: number | null;
  max_drawdown?: number | null;
  critic_verdict?: string | null;
  n_critic_flags?: number | null;
  best_model?: string | null;
  is_synthetic?: boolean | null;
  universe_size?: number;
  error?: string | null;
}

export interface RunStep {
  sequence: number;
  agent: string;
  status: "ok" | "failed" | "degraded" | "skipped";
  seconds: number;
  summary?: Record<string, unknown> | null;
  error?: string | null;
  degraded_reason?: string | null;
  llm_calls: number;
  llm_input_tokens: number;
  llm_output_tokens: number;
}

export interface ModelFold {
  model: string;
  fold: number;
  train: [string, string];
  test: [string, string];
  n_train: number;
  n_test: number;
  accuracy: number | null;
  auc: number | null;
  information_coefficient: number | null;
  signal_mean_return: number | null;
  signal_t_stat: number | null;
  signal_p_value: number | null;
}

export interface RunDetail extends RunSummary {
  plan?: ResearchPlan | null;
  capabilities?: Capabilities;
  data_provenance?: Record<string, any>;
  result?: RunResult | null;
  approval?: {
    id: string;
    status: string;
    decided_by?: string | null;
    decided_at?: string | null;
    feedback?: string | null;
    critic_summary?: Record<string, any> | null;
  } | null;
}

export interface ResearchPlan {
  question: string;
  universe: string[];
  universe_rationale: string;
  n_days: number;
  label_horizon: number;
  n_folds: number;
  factor_families: string[];
  include_deep_learning: boolean;
  ingest_filings: boolean;
  filing_tickers: string[];
  document_question?: string | null;
  max_weight: number;
  rebalance_days: number;
  planner_backend: string;
  notes: string[];
}

export interface CriticCheck {
  check: string;
  passed: boolean | null;
  severity: string;
  detail: string | string[];
}

export interface CriticResult {
  checks: CriticCheck[];
  n_checks_run: number;
  n_checks_failed: number;
  n_errors: number;
  n_warnings: number;
  overall_verdict: string;
  recommended_action: string;
  recommended_revisions: string[];
  evaluated_strategy?: string | null;
  verdict_note: string;
}

export interface StrategyResult {
  status: string;
  signal_name?: string;
  n_trading_days?: number;
  n_rebalances?: number;
  period?: [string, string];
  equity_curve?: { date: string; equity: number }[];
  drawdown_curve?: { date: string; drawdown: number }[];
  metrics?: Record<string, number | null>;
  costs?: Record<string, any>;
  significance?: Record<string, any>;
  note?: string;
}

export interface RunResult {
  question?: string;
  plan?: ResearchPlan;
  selected_factors?: string[];
  factor_diagnostics?: {
    coverage: Record<string, { coverage: number; n_present: number; usable: boolean }>;
    rejected: Record<string, string>;
    min_coverage: number;
  };
  factor_ic?: Record<string, any>;
  fama_macbeth?: {
    status: string;
    mode?: string;
    n_names?: number;
    n_factors?: number;
    caveat?: string;
    per_factor?: Record<string, any>;
  };
  ml_result?: Record<string, any>;
  backtest_comparison?: {
    status: string;
    strategies: Record<string, StrategyResult>;
    n_strategies_tested: number;
    best_strategy?: string | null;
    best_sharpe?: number | null;
    baseline_sharpe?: number | null;
    model_beats_baseline?: boolean | null;
    multiple_testing_note?: string;
  };
  risk_result?: Record<string, any>;
  portfolio_result?: Record<string, any>;
  critic_result?: CriticResult;
  documents?: Record<string, any>;
  data_summary?: Record<string, any>;
  price_series?: Record<string, { dates: string[]; close: number[] }>;
  steps?: RunStep[];
  revisions_applied?: string[];
  revision_count?: number;
  report_markdown?: string;
  provenance?: Record<string, any>;
}

export interface DocumentRow {
  id: string;
  ticker: string | null;
  company_name: string | null;
  doc_type: string;
  title: string;
  filing_date: string | null;
  fiscal_period: string | null;
  source: string;
  source_url: string | null;
  char_count: number;
  n_chunks: number;
  ingested_at: string;
}

export interface RetrievalHit {
  chunk_id: string;
  text: string;
  section: string | null;
  ticker: string | null;
  doc_type: string;
  filing_date: string | null;
  source_url: string | null;
  scores: {
    score: number;
    dense_score: number | null;
    lexical_score: number | null;
    rerank_score: number | null;
    fusion_score: number;
  };
  retrieval_stage: string;
}

export interface ApprovalItem {
  approval_id: string;
  run_id: string;
  question: string;
  critic_verdict: string | null;
  critic_summary: Record<string, any> | null;
  sharpe: number | null;
  cagr: number | null;
  max_drawdown: number | null;
  best_model: string | null;
  created_at: string;
  decided_by: string | null;
  decided_at: string | null;
}

export interface AgentHealth {
  agent: string;
  n_executions: number;
  avg_seconds: number;
  max_seconds: number;
  status_counts: Record<string, number>;
  failure_rate: number;
  degraded_rate: number;
  llm_calls: number;
  llm_input_tokens: number;
  llm_output_tokens: number;
}

export interface PortfolioBook {
  id: string;
  name: string;
  notional: number;
  base_currency: string;
  is_active: boolean;
  source_run_id: string | null;
  method: string | null;
  created_at: string;
  updated_at: string;
  positions: Record<string, number>;
  n_positions: number;
  gross_exposure: number;
  net_exposure: number;
}
