# reference.md — Orbit

Single source of truth for architecture decisions, known limits, and what is
genuinely built versus described. Update after every meaningful change.

**Status:** built, tested and verified end-to-end against live external data.
109 backend tests passing. Frontend production build compiles clean with full
type checking. Full pipeline verified through the HTTP API with real market,
filing, macro and news data.

---

## 1. What is real

Every one of these was exercised end to end during the build, not just written:

| Capability | Provider | Verified |
|---|---|---|
| Daily OHLCV, split/dividend adjusted | Yahoo Finance (`yfinance`) | 756 aligned trading days × 20 tickers |
| Point-in-time fundamentals | SEC EDGAR XBRL company-facts API | 665 facts for AAPL across 10 metrics |
| Macro series with publication lags | FRED CSV endpoint | 7 series, 2,013 rows, regimes correctly identified |
| Filing text for retrieval | SEC EDGAR archives | 2 Apple 10-Qs, 134 embedded chunks |
| Dated news | Yahoo Finance news | 30 items across 3 tickers |
| Financial sentiment | FinBERT (`ProsusAI/finbert`) | Live, negation handled correctly |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | Live |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Live, MRR 1.0 on a labelled set |
| Deep sequence models | PyTorch 2.13 CPU | LSTM + Transformer trained across folds |
| Orchestration | LangGraph 1.2 | Cycle verified — critic rejected, plan revised, re-ran |
| Experiment tracking | MLflow 3.15 | Params, metrics and artefacts logged |

---

## 2. Architecture decisions, and why

### Orchestration: LangGraph — *reversing an earlier decision*

The first version used a plain sequential function and documented, correctly, that
a straight-line DAG gains nothing from a graph framework. That reasoning was sound
for what it described and **stopped being true** once the critic became
load-bearing: it can now reject a result, the planner revises the configuration,
and the graph re-enters at ingestion. That is a cycle with a conditional edge and
bounded iteration — precisely the control flow a hand-rolled pipeline expresses
badly. The framework is here because the topology changed, not because it is
fashionable.

### Async execution: thread pool + database, not Celery/Redis

A run takes minutes. Executing inside the request meant any client timeout
destroyed the result and a restart wiped every completed run. The fix does not
need a broker: runs execute on a bounded `ThreadPoolExecutor` and **every state
transition is persisted**, so the `runs` table *is* the queue. Clients poll. This
survives restarts (stale `running` rows are reaped on boot), adds no fourth
service, and is honest about its one limit — single-node.

Redis + a broker earns its place when runs must survive process death mid-flight
or fan out across machines. `submit_run` is the only function that changes.

### Vector store: brute-force cosine in NumPy, not Qdrant/pgvector

Embeddings live as raw `float32` bytes on the `chunks` table. At hundreds to low
thousands of chunks, an exact scan is *faster* than building and maintaining an
ANN index and returns exact neighbours. `core/vectorstore.py::search` is the single
swap point if the corpus outgrows it. Adding a vector database here would be
infrastructure without a benefit.

### Database: SQLAlchemy over SQLite, targeting Postgres

One ORM, two dialects, two differences handled in `core/db.py`
(`check_same_thread` for SQLite, a real pool for Postgres). The platform runs with
zero infrastructure by default and moves to Postgres by changing one env var.

### Retrieval: hybrid + rerank, not dense-only

Dense embeddings are excellent at paraphrase and poor at exact tokens — and
financial queries are full of exact tokens (`10-K`, `FY2024`, `$1.2 billion`,
`EBITDA`). Retrieval fuses dense and BM25 by reciprocal rank (chosen over weighted
score-sum because the two scores live on incomparable scales), then reranks the
top candidates with a cross-encoder, because precision at rank 1 is what matters
when a number is about to be quoted.

### Report: template, never generative

The one place the temptation to use an LLM is strongest and the cost highest. A
language model writing a research summary will round, infer and smooth numbers.
The report is assembled from structured upstream output, so it *cannot* state a
figure no stage computed.

### Password hashing: `bcrypt` directly, not passlib

passlib 1.7.x probes `bcrypt.__about__.__version__`, which bcrypt 4+ removed. Its
backend detection fails and it mishandles the 72-byte limit — observed as a real
failure during the build. The direct API is smaller and maintained.

---

## 3. Bugs found and fixed during the build

Kept because each one is a real methodological lesson.

1. **Outlier detection judged a point against a window containing itself.**
   The rolling z-score included the outlier, inflating the window's standard
   deviation so large outliers escaped detection entirely. Fixed by shifting the
   window to prior data only (`log_ret.shift(1).rolling(...)`). *Do not
   "simplify" the shift away.*

2. **Deflated Sharpe had a units bug.** It was fed an annualized Sharpe against a
   count of monthly observations, overstating significance by √12 and reporting
   near-certainty for a coin-flip result. Now takes `periods_per_year` and
   de-annualizes internally. Regression test in `test_statistics.py`.

3. **Fama-MacBeth was unidentified and returned silently empty.** With 6 names
   and 14 factors the cross-sectional regression has no unique solution. It now
   selects multivariate or univariate mode from the data and states the
   limitation of the univariate estimate explicitly.

4. **Sequence models were scored on a different test set.** Sequences were built
   from the test slice alone, silently discarding the first 19 days of each test
   window — so the deep models were evaluated on a smaller, later subset than the
   tabular ones. Now built over contiguous history and sliced by end date.

5. **Accuracy was reported without a base rate.** An LSTM showing 0.668 accuracy
   looked strong until the base rate turned out to be 0.618. Accuracy, base rate,
   lift and a degenerate-predictor flag are now reported together.

6. **Coverage was reported for factors that were never candidates.** The panel was
   sliced before measurement, so unrequested factors showed "0.0% coverage,
   rejected" — a measured finding about something never measured.

7. **The document lookup was keyed by chunk id instead of document id.** Caught
   immediately by an integration check against real filings.

---

## 4. Statistical guarantees

| Guarantee | Mechanism | Test |
|---|---|---|
| No fold-boundary label leakage | Purge = label horizon, plus embargo | `test_purge_gap_covers_the_label_horizon` |
| No publication leakage | EDGAR `filed` date; macro publication lags | `build_pit_fundamentals`, `fetch_macro` |
| No constancy leakage | Time-constant features rejected; empirical critic check | `test_constant_factor_is_rejected` |
| Autocorrelation-corrected inference | Newey-West HAC, naive shown beside it | `test_no_false_significance_on_zero_edge_data` |
| Multiple-testing correction | Deflated Sharpe against trial count | `test_more_trials_raises_the_bar` |
| Dependence-aware sample size | Effective n discounts overlap and cross-correlation | `test_overlap_and_correlation_shrink_n` |
| Interval estimates | Stationary block bootstrap | `test_zero_edge_interval_spans_zero` |
| Grounded numbers in generated text | Verbatim re-scan with unit normalisation | `test_catches_a_fabricated_figure` |

---

## 5. Deliberately not built

- **Live trading, order management, execution.** Out of scope by design.
- **Intraday or tick data.** Daily bars only.
- **Options, futures, fixed income.** Equity cross-section only.
- **Multi-tenancy beyond per-user run ownership.** No orgs, roles or sharing.
- **Refresh-token rotation.** Adding it without a revocation store would be
  security theatre; short-lived JWTs are the honest position at this scope.
- **Kubernetes manifests.** Docker Compose only.

---

## 6. Known limitations

Also stated in the README and printed in every generated report:

- **Survivorship bias** — universes are built from currently-listed tickers.
- **Costs are modelled, not measured** — square-root impact with k=1.0 is a
  literature estimate; understated for thin names and at larger AUM.
- **Borrow is assumed available at a flat 50 bps** — hard-to-borrow names cost
  far more and can be recalled.
- **News history is shallow** — the sentiment feature usually fails the coverage
  gate and is dropped rather than backfilled. Correct, but it means sentiment
  rarely contributes.
- **No market-impact feedback** — the strategy's own trading is assumed not to
  move prices.
- **Small cross-sections are weak** — below ~15 names the tercile legs collapse
  to one or two positions and idiosyncratic risk dominates. The critic raises this
  as an error-level finding.

---

## 7. Next steps

1. Broker-backed execution (Celery or Arq + Redis) for multi-node runs.
2. A survivorship-bias-free universe via historical index constituents.
3. Ablation studies: which feature families actually contribute, measured by
   retraining without each.
4. Model registry with promotion gates, so a model must clear the critic before
   being marked servable.
5. Scheduled re-runs with drift-triggered retraining, using the PSI monitor
   already implemented.
