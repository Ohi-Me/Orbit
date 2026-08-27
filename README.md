# Orbit

**An end-to-end applied machine learning platform, demonstrated on financial data.**

Orbit takes a research question in plain language, plans an experiment, ingests and validates
real data from four external sources, engineers point-in-time features, trains and compares four
model families under leakage-proof cross-validation, evaluates the result with a realistic cost
model, criticises its own conclusions, and stops at a human approval gate.

It is built around one idea: **a machine learning result is only as good as the discipline that
produced it.** Most of the engineering here exists to stop the platform from fooling itself.

---

## What it actually does

| Layer | Implementation |
|---|---|
| **Orchestration** | LangGraph state machine, 14 agents, conditional cycle — the validation stage can reject a result and re-run with a revised configuration |
| **Data ingestion** | Yahoo Finance (prices), SEC EDGAR XBRL (point-in-time fundamentals), FRED (macro), news — each with validation, provenance and honest fallbacks |
| **Feature engineering** | 17 features across momentum, volatility, value, quality, growth, sentiment and macro, admitted on measured coverage |
| **Models** | Logistic regression · XGBoost · LSTM · Transformer encoder, all on identical folds |
| **Validation** | Purged, embargoed, expanding walk-forward CV; HAC-corrected inference; deflated Sharpe for multiple testing |
| **Evaluation** | Cost-aware backtest (spread + square-root market impact + borrow), factor risk decomposition, walk-forward portfolio construction |
| **RAG** | Hybrid dense + BM25 retrieval over SEC filings, cross-encoder reranking, citation validation, numeric grounding verification |
| **NLP** | FinBERT for financial sentiment, management-tone analysis, risk-factor extraction |
| **MLOps** | MLflow experiment tracking, drift detection (PSI), structured logging, agent-level latency and failure metrics |
| **Serving** | FastAPI, JWT auth, async job execution, SQLAlchemy (SQLite → Postgres), Next.js 14 dashboard |

---

## The engineering that matters

Anyone can wire an LLM to a chart. These are the parts that took real work, and the reasons they
exist.

### 1. Leakage prevention, tested rather than assumed

The label is a 21-day forward return. That single fact creates three separate leaks, and the
platform closes all three:

- **Fold-boundary leakage.** A training sample on the last day before the test window has a label
  that resolves *inside* it. Cross-validation therefore **purges** training samples whose label
  window overlaps the test period and applies an **embargo** on top. `test_leakage.py` asserts the
  gap covers the full horizon.
- **Publication leakage.** Apple's FY2023 revenue was not knowable in June 2023 — the 10-K was
  filed in November. Fundamentals key on EDGAR's `filed` date, never the period end, and the
  first-reported value wins over later restatements. Macro series are shifted by their real
  publication lag.
- **Constancy leakage.** The subtlest one. Computing sentiment today and writing it into every
  historical row makes it a constant *derived from the present* — every past date silently
  contains future knowledge. Features that are constant within a ticker are rejected
  automatically, and the critic tests for this empirically rather than by inspecting column names.

### 2. Inference that survives contact with reality

Overlapping return windows make consecutive observations share 20 of 21 days of price path. A
plain t-test on that data is badly wrong. On simulated data with **zero true edge**, the platform's
own test suite shows:

```
naive  t = -5.77   (spuriously "significant")
HAC    t = -1.51   (correctly not significant)
inflation factor = 3.8x
```

Every significance claim uses Newey-West HAC standard errors, and the naive number is displayed
next to it so the gap is visible. Model selection across N strategies is corrected with a
**deflated Sharpe ratio** — the best of 12 backtests is judged against what the luckiest of 12
worthless strategies would produce.

### 3. The model and the strategy are the same object

An earlier design validated two models, then backtested an unrelated hand-weighted formula. The ML
table and the equity curve described different things, so the platform could not answer its own
question. The ML stage now emits its **out-of-sample predictions**, and the backtest trades those
directly. The hand-weighted blend survives as an explicitly labelled baseline to beat.

### 4. Honest reporting of negative results

Accuracy on a directional label is nearly meaningless without the base rate. In a bull sample where
62% of 21-day forward returns are positive, a model scoring 0.62 has learned nothing. Orbit reports
**accuracy, base rate, and lift** side by side, flags folds where a model predicted a single class
for >95% of samples, and will state plainly that no model beat the baseline when that is the case.

The report leads with limitations rather than burying them, and is assembled by a **template from
structured upstream output** — no language model writes the numbers, so it cannot state a figure no
stage computed.

### 5. Grounded retrieval

Financial RAG fails in specific ways, and each has a specific control:

- Metadata is a **hard pre-filter**, not a soft signal — a semantically perfect passage from the
  wrong company or quarter is a wrong answer, not a near miss.
- Retrieval is **hybrid**: dense embeddings find paraphrase, BM25 finds exact tokens like `10-K`,
  `FY2024`, `$1.2 billion`. Fused by reciprocal rank, then reranked by a cross-encoder.
- **Every number in an answer is verified against the source text**, normalising formatting so
  `$1.2 billion` and `1200000000` match. A figure that is not present verbatim is flagged
  `UNGROUNDED` and the answer is marked unverified.

### 6. Human-in-the-loop, structurally

The pipeline has no terminal "done" state. It ends at `awaiting_approval`, and a result becomes
decision-grade only when a **named, authenticated person** approves it. Portfolio weights cannot be
adopted into a book from an unapproved run — the API returns 409. An LLM chooses what to
*investigate*; it never decides significance, never sizes a position, and never approves its own
work.

---

## Architecture

```
                       ┌──────────────┐
                       │   Planner    │  natural language → validated config
                       └──────┬───────┘
                              ▼
   ┌──────────── Data ingestion (4 sources, validated, provenance-tracked) ─────────┐
   │  market data · macro · fundamentals (point-in-time) · news · filings (RAG)     │
   └──────────────────────────────┬────────────────────────────────────────────────┘
                                  ▼
                       ┌──────────────────┐
                       │ Feature panel    │  coverage-gated, leakage-checked
                       └──────┬───────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  Model comparison (purged walk-forward) │  linear · GBM · LSTM · Transformer
        └──────┬──────────────────────────────────┘
               ▼  out-of-sample predictions
        ┌──────────────┬──────────────┬────────────────────┐
        │  Backtest    │    Risk      │    Portfolio       │
        │ (cost model) │ (decomposed) │  (walk-forward)    │
        └──────┬───────┴──────┬───────┴─────────┬──────────┘
               └──────────────┴─────────────────┘
                              ▼
                       ┌──────────────┐
                       │   Critic     │──── reject ──┐
                       └──────┬───────┘              │
                              │ pass                 ▼
                              ▼               ┌──────────────┐
                       ┌──────────────┐       │   Revise     │
                       │   Report     │       └──────┬───────┘
                       └──────┬───────┘              │
                              ▼                      └──► back to ingestion
                    ┌───────────────────┐
                    │ Human approval    │  ← the only exit
                    └───────────────────┘
```

---

## Quick start

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Check what the deployment can actually do — every capability is probed, not declared:

```bash
curl http://localhost:8000/api/health
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>.

### Tests

```bash
cd backend
pytest -q                      # 109 tests
```

---

## Configuration

Everything optional degrades honestly and reports that it did.

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./quant_platform.db` | Point at Postgres for deployment; ORM is identical |
| `JWT_SECRET` | dev default | **Set this before deploying** — health endpoint warns if unset |
| `AUTH_REQUIRED` | `false` | Require a token for every route |
| `ANTHROPIC_API_KEY` | unset | Enables the LLM planner and RAG synthesis; falls back to deterministic routing and extractive evidence |
| `ALLOW_LIVE_MARKET_DATA` | `true` | Falls back to a labelled synthetic generator when unreachable |
| `ALLOW_MODEL_DOWNLOAD` | `true` | Enables FinBERT, embeddings and the reranker; falls back to BM25 and a keyword baseline |
| `SEC_USER_AGENT` | generic | SEC requires a self-identifying UA; set your own contact |
| `ENABLE_MLFLOW` | `true` | Experiment tracking; no-ops cleanly if MLflow is absent |
| `MAX_CONCURRENT_RUNS` | `2` | Worker pool size |

---

## Repository layout

```
backend/
  app/
    agents/         one module per pipeline stage
    api/            auth · runs · documents · portfolio · monitoring · ml
    core/           config · db · models · security · stats · vectorstore · observability
    data/           providers (prices · edgar · macro · news) + validation
    models_dl/      PyTorch LSTM and Transformer
    orchestration/  LangGraph state machine
    services/       async run execution and persistence
  tests/            109 tests
frontend/
  app/
    components/     shared UI kit and shell
    lib/            typed API client
    runs/[id]/      the main research screen (9 tabs)
    models/ agents/ documents/ portfolio/ monitoring/ approvals/
```

---

## Known limitations

Stated here rather than discovered later:

- **Survivorship bias.** The universe is defined from currently-listed tickers, so companies that
  delisted during the window are absent. This biases historical results upward by construction.
- **Costs are modelled, not measured.** The square-root impact law with k=1.0 is a literature
  estimate. Real fills depend on venue, urgency and book state.
- **Short borrow is assumed available at a flat 50 bps.** Hard-to-borrow names cost far more and
  can be recalled.
- **News history is shallow.** Free news endpoints return recent items only, so the sentiment
  feature usually fails the coverage gate and is dropped rather than backfilled — which is the
  correct behaviour, but it means sentiment rarely contributes.
- **Single-node execution.** Runs execute on an in-process thread pool. A restart marks in-flight
  runs as interrupted rather than resuming them. A broker (Celery/Arq + Redis) is the documented
  upgrade path; `submit_run` is the only function that changes.
- **No live trading.** Orbit produces research artefacts. It has no broker integration, no order
  management, and no execution layer — by design.

---

## License

MIT
