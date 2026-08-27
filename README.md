# Orbit

### Evidence-Driven Quantitative Research Platform

**Orbit turns financial research questions into reproducible, evidence-backed quantitative experiments.**

It combines point-in-time financial data, quantitative factors, financial NLP/RAG, machine learning, walk-forward validation, cost-aware backtesting, portfolio construction, risk analysis, and AI-assisted research orchestration.

> **Research Question → Evidence → Features → Models → OOS Signals → Backtest → Risk → Critic → Report → Human Approval**

The goal is not to predict stock prices.
The goal is to test whether combining different information sources produces **statistically meaningful and economically useful improvements in risk-adjusted performance**.

---

## What Orbit Does

| Area            | Capability                                                                          |
| --------------- | ----------------------------------------------------------------------------------- |
| **Research**    | Natural-language research questions → structured experiments                        |
| **Data**        | Market prices, SEC/XBRL fundamentals, macro data, filings, news                     |
| **Features**    | Momentum, value, quality, growth, volatility, sentiment, macro                      |
| **ML**          | Logistic Regression, XGBoost, LSTM, Transformer                                     |
| **Validation**  | Purged + embargoed expanding walk-forward CV                                        |
| **Statistics**  | HAC/Newey-West inference, multiple-testing controls                                 |
| **Backtesting** | OOS signals, transaction costs, slippage, market impact, borrow                     |
| **Risk**        | Sharpe, Sortino, drawdown, beta, turnover, factor exposure                          |
| **Portfolio**   | Risk-aware portfolio construction and constraints                                   |
| **RAG**         | Hybrid vector + BM25 retrieval, reranking, citations, grounding                     |
| **NLP**         | Financial sentiment, management tone, events, risk factors                          |
| **Agents**      | Research planning, data, RAG, NLP, quant, ML, backtest, risk, validation, reporting |
| **MLOps**       | MLflow, model/dataset tracking, drift and system monitoring                         |
| **Serving**     | FastAPI, async jobs, authentication, Next.js                                        |
| **Governance**  | Human approval before research results become decision-ready                        |

---

## Research Pipeline

```text
Research Question
       ↓
Research Planner
       ↓
Data + Documents + News + Macro
       ↓
Point-in-Time Features
       ↓
Leakage / Coverage Validation
       ↓
Model Comparison
       ↓
Out-of-Sample Predictions
       ↓
Signal → Portfolio
       ↓
Cost-Aware Backtest
       ↓
Risk + Robustness
       ↓
Research Critic
   ↙          ↘
Reject        Pass
  ↓             ↓
Revise        Report
  └──────→      ↓
           Human Approval
```

The critical design principle is that **model predictions flow directly into the backtest**. The system does not validate one model and then evaluate an unrelated hand-built strategy.

---

## Quantitative Rigor

Orbit is designed around common failure modes in financial ML.

### Point-in-time research

Fundamentals use filing availability rather than simply fiscal-period dates. Macro and document features respect publication timing.

### Leakage prevention

The validation framework addresses:

* forward-label overlap
* fold-boundary leakage
* embargo periods
* publication leakage
* improperly backfilled historical features
* model-selection leakage

### Time-series validation

Models are evaluated using expanding walk-forward folds with purging and embargo rather than random train/test splits.

### Statistical evaluation

Because financial observations are dependent, Orbit uses HAC/Newey-West inference where appropriate and accounts for multiple strategy selection when evaluating Sharpe ratios.

### Honest model comparison

Models are compared against strong baselines. Complexity only earns its place when it improves **out-of-sample** performance.

---

## Models & Signals

Orbit compares:

```text
Baseline / Factor Models
        │
        ├── Logistic Regression
        ├── XGBoost
        ├── LSTM
        └── Transformer
```

Across feature families including:

* technical signals
* value
* quality
* growth
* momentum
* volatility
* fundamentals
* financial sentiment
* macro regime

The research question is:

> **Does additional information or model complexity provide incremental predictive and portfolio value after costs and risk?**

---

## Financial RAG & NLP

Orbit's document research pipeline is built for financial evidence rather than generic chatbot retrieval.

```text
SEC / Financial Documents
        ↓
Parse + Chunk + Metadata
        ↓
Embeddings + BM25
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Evidence
        ↓
LLM Synthesis
        ↓
Citation + Numerical Grounding
```

It can investigate:

* SEC filings
* earnings information
* management tone
* sentiment
* risk factors
* corporate events
* financial news

Important answers are linked back to source evidence, while unsupported numerical claims are flagged.

---

## Multi-Agent Architecture

Agents have defined responsibilities rather than acting as independent chatbots:

```text
Planner
  ↓
Market Data
  ↓
Financial Documents / RAG
  ↓
NLP / News
  ↓
Quant Research
  ↓
ML Modeling
  ↓
Backtesting
  ↓
Risk
  ↓
Portfolio
  ↓
Validation / Critic
  ↓
Research Report
```

The orchestration layer manages structured state, tools, permissions, validation, retries, and failures.

**LLMs assist investigation and synthesis; they do not determine statistical significance, position sizing, or approve investment decisions.**

---

## Architecture

```text
                    Next.js
                       │
                       ▼
                    FastAPI
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Research API       Document API
              │                 │
              ▼                 ▼
          LangGraph       RAG / NLP
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
    Data     ML     Quant Engine
      │       │        │
      └───────┼────────┘
              ▼
        Risk / Portfolio
              │
              ▼
       Validation / Critic
              │
              ▼
          Report
              │
              ▼
       Human Approval

PostgreSQL + pgvector
Redis + Arq
MLflow
Object Storage
```

---

## Technology Stack

| Layer         | Technology                        |
| ------------- | --------------------------------- |
| Frontend      | Next.js, TypeScript               |
| API           | FastAPI                           |
| Database      | PostgreSQL, SQLAlchemy            |
| Vector Search | pgvector                          |
| Async Jobs    | Redis + Arq                       |
| Quant         | NumPy, pandas, SciPy, statsmodels |
| ML            | scikit-learn, XGBoost             |
| Deep Learning | PyTorch                           |
| NLP           | Transformers / FinBERT            |
| RAG           | pgvector + BM25 + reranker        |
| Agents        | LangGraph                         |
| LLM           | Claude                            |
| MLOps         | MLflow                            |
| Storage       | S3-compatible object storage      |
| Deployment    | Docker + Vercel                   |
| CI/CD         | GitHub Actions                    |

The architecture intentionally avoids unnecessary microservices, multiple agent frameworks, multiple vector databases, and infrastructure added only for buzzword value.

---

## Reproducibility

Every research run should be traceable to:

```text
Research Run
   ↓
Dataset Version
   ↓
Feature Version
   ↓
Model Version
   ↓
Experiment Configuration
   ↓
Backtest Configuration
   ↓
Research Report
```

Tracked configuration includes universe, dates, features, model parameters, random seeds, validation windows, costs, portfolio constraints, and code version.

---

## Frontend

The research workstation focuses on decisions a researcher actually needs to make:

* **Research Workspace** — question, methodology, evidence, findings
* **Model Comparison** — baselines, ML models, fold performance
* **Backtest** — equity curve, benchmark, drawdown, turnover, costs
* **Portfolio & Risk** — weights, exposures, concentration, risk
* **Document Research** — search, evidence, citations
* **Agent Trace** — pipeline state, tools, failures, latency
* **Research Report** — methodology, results, evidence, limitations

---

## Production & MLOps

Orbit is designed to evolve from research prototype to production system with:

* PostgreSQL persistence
* asynchronous workers
* experiment/model tracking
* dataset lineage
* model registry
* structured logging
* data/model drift monitoring
* inference and agent latency monitoring
* LLM usage/cost tracking
* authentication and RBAC
* CI/CD
* Dockerized deployment

---

## Quick Start

### Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open:

```text
http://localhost:3000
```

### Tests

```bash
cd backend
pytest -q
```

---

## Repository Structure

```text
Orbit/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── api/
│   │   ├── core/
│   │   ├── data/
│   │   ├── models_dl/
│   │   ├── orchestration/
│   │   └── services/
│   └── tests/
│
├── frontend/
│   └── app/
│       ├── components/
│       ├── lib/
│       ├── runs/
│       ├── models/
│       ├── agents/
│       ├── documents/
│       ├── portfolio/
│       ├── monitoring/
│       └── approvals/
│
├── docs/
├── .github/
└── README.md
```

---

## Known Limitations

Orbit is a research platform, not a live trading system.

Current limitations include:

* survivorship bias in the current universe construction
* modelled rather than observed transaction costs
* simplified short-borrow assumptions
* limited historical news coverage
* external data-source availability and rate limits
* single-node execution in development
* no broker or order-management integration

These limitations are intentionally documented rather than hidden.

---

## Research Philosophy

**Evidence over narrative.**
**Out-of-sample over in-sample.**
**Simple baselines must be beaten.**
**Negative results are valid results.**
**AI assists research; humans remain accountable.**

---

## Disclaimer

Orbit is a quantitative research software project.

It is **not investment advice, a broker, a trading system, or a guarantee of future investment performance**.
