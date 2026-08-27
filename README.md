# Orbit

### Evidence-driven quantitative research infrastructure

**Orbit is an end-to-end financial research platform for investigating whether market, fundamental, sentiment, and macro signals produce statistically meaningful, risk-adjusted investment signals.**

Instead of asking an LLM to predict a stock price, Orbit turns a research question into a **reproducible quantitative experiment**:

> **Question → Evidence → Features → Models → Out-of-sample Signals → Backtest → Risk → Criticism → Research Report**

The platform combines point-in-time financial data, quantitative factors, financial NLP, grounded RAG, machine learning, walk-forward validation, cost-aware backtesting, portfolio construction, risk analysis, and human review.

The central design principle is simple:

> **A model is only useful when the research process around it is rigorous enough to prevent the model from fooling you.**

---

## Why Orbit exists

Financial ML projects often demonstrate a model without demonstrating that the model produces a useful investment result.

A high accuracy score does not imply alpha.

A high backtest return does not imply robustness.

A good-looking LLM answer does not imply factual correctness.

And a sophisticated neural network does not compensate for look-ahead bias, survivorship bias, leakage, unrealistic transaction costs, or multiple testing.

Orbit is designed around those problems.

It asks a more useful question:

> **Does combining fundamental, technical, sentiment, and macro information improve out-of-sample risk-adjusted performance relative to strong baselines, after realistic trading costs and risk constraints?**

---

# What Orbit can do

| Capability                 | Purpose                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| **Research orchestration** | Convert a natural-language research question into a structured experiment |
| **Market data**            | Prices, returns, volatility, liquidity and market-derived features        |
| **Fundamentals**           | Point-in-time company financial information from SEC EDGAR/XBRL           |
| **Macro data**             | Economic indicators and publication-aware macro features                  |
| **Financial documents**    | SEC filings and research evidence with citations                          |
| **Financial NLP**          | Sentiment, management tone, events and risk-factor extraction             |
| **Quantitative factors**   | Momentum, value, quality, growth, volatility and other systematic signals |
| **Machine learning**       | Logistic regression, XGBoost, LSTM and Transformer models                 |
| **Time-series validation** | Purged, embargoed, expanding walk-forward evaluation                      |
| **Backtesting**            | Out-of-sample signal evaluation with transaction costs and market impact  |
| **Risk analytics**         | Drawdown, volatility, Sharpe, Sortino, beta and factor exposures          |
| **Portfolio construction** | Risk-aware portfolio allocation using model outputs                       |
| **RAG**                    | Hybrid retrieval, reranking, citation validation and numerical grounding  |
| **Research critic**        | Challenges assumptions, leakage, statistical validity and robustness      |
| **Research reports**       | Reproducible reports generated from structured research outputs           |
| **Human approval**         | Prevents an AI system from autonomously approving investment decisions    |
| **Experiment tracking**    | Dataset, feature, model and experiment provenance                         |
| **Monitoring**             | Data quality, drift, latency, failures and system health                  |

---

# Research workflow

A typical Orbit research run looks like this:

```text
Research Question
       │
       ▼
Research Planner
       │
       ▼
Data Discovery
       │
       ├── Market Data
       ├── Fundamentals
       ├── SEC Filings
       ├── Financial News
       └── Macro Data
       │
       ▼
Point-in-Time Feature Engineering
       │
       ▼
Leakage & Coverage Validation
       │
       ▼
Model Benchmarking
       │
       ├── Naive / Statistical Baselines
       ├── Logistic Regression
       ├── XGBoost
       ├── LSTM
       └── Transformer
       │
       ▼
Out-of-Sample Predictions
       │
       ▼
Strategy Construction
       │
       ▼
Cost-Aware Backtest
       │
       ├── Transaction Costs
       ├── Slippage
       ├── Market Impact
       └── Borrow Costs
       │
       ▼
Risk & Portfolio Analysis
       │
       ▼
Robustness & Statistical Tests
       │
       ▼
Research Critic
       │
       ├── Reject → Revise → Re-run
       │
       ▼
Research Report
       │
       ▼
Human Approval
```

The important property is that **the model and strategy are connected**.

The backtest uses the model's out-of-sample predictions rather than evaluating an unrelated hand-built score.

---

# Quantitative research methodology

## Point-in-time data

Historical research must use information that was actually available at the time.

Orbit therefore distinguishes between:

```text
period_end
    ↓
when the economic period occurred

filed_at / published_at
    ↓
when the information became observable

available_at
    ↓
when the feature is allowed to enter the research dataset
```

This prevents a model from learning information that would not have been available to an investor at the time.

Fundamental data is keyed by filing availability rather than simply by fiscal period.

Macro features account for publication timing.

---

# Leakage prevention

Orbit explicitly tests for several classes of leakage.

### Fold-boundary leakage

Forward-return labels overlap future periods.

Training observations whose label windows overlap a test window are therefore purged, with an embargo applied around the test period.

### Publication leakage

A company's financial statement cannot become a feature before the filing was publicly available.

### Derived-feature leakage

Historical rows cannot contain features computed using information from the present.

Orbit tests for suspicious constant or improperly time-aligned derived features.

### Selection leakage

Model and strategy selection must not use future test-period performance.

The final evaluation is reserved for genuinely out-of-sample observations.

---

# Validation methodology

Orbit uses time-aware evaluation rather than random train/test splitting.

### Walk-forward validation

```text
Train ────────► Test
Train ───────────────► Test
Train ─────────────────────► Test
Train ───────────────────────────► Test
```

Training expands through time while evaluation remains strictly forward-looking.

### Purging

Observations whose forward-return labels overlap the test period are removed.

### Embargo

A gap is maintained around test periods to reduce contamination from overlapping observations.

### Statistical inference

Financial observations are temporally dependent.

Orbit therefore uses HAC/Newey-West inference where appropriate instead of treating every observation as independent.

### Multiple testing

When many strategies are evaluated, the best observed Sharpe ratio can be inflated simply by selection.

Orbit incorporates deflated-Sharpe-style adjustment to account for multiple strategy trials.

---

# Models

Orbit intentionally compares models rather than assuming that a more complex model is better.

## Baselines

Strong baselines are essential.

The platform compares ML models against simpler approaches such as:

* majority-class / naive baseline
* linear models
* factor-based signals
* hand-weighted composite baseline

## XGBoost

XGBoost provides a strong nonlinear tabular baseline for:

* fundamentals
* technical indicators
* factor exposures
* sentiment
* macro variables

## LSTM

LSTM models are used to investigate sequential representations of market and feature history.

## Transformer

Transformer encoders provide a higher-capacity sequence model for temporal feature representations.

### Important principle

All candidate models are evaluated using the **same research folds and evaluation protocol**.

The goal is not:

> “Which model has the highest training accuracy?”

The goal is:

> “Does additional model complexity produce a statistically and economically meaningful improvement out of sample?”

---

# Features and factors

Orbit treats feature engineering as a research problem rather than a preprocessing step.

Current feature families include:

### Momentum

* trailing returns
* multi-horizon momentum
* trend measures

### Risk

* realized volatility
* downside volatility
* drawdown-related features

### Value

* valuation ratios
* earnings-based measures
* fundamental relative-value features

### Quality

* profitability
* balance-sheet characteristics
* operating quality

### Growth

* revenue growth
* earnings growth
* fundamental acceleration

### Sentiment

* financial sentiment
* management tone
* earnings-call language
* news signals

### Macro

* rates
* inflation
* economic activity
* market regime variables

Features are admitted based on **coverage, availability, and leakage checks** rather than being blindly backfilled.

---

# Financial NLP

Orbit uses financial NLP to transform unstructured information into researchable signals.

The NLP pipeline can investigate:

* earnings-call sentiment
* management confidence
* uncertainty
* risk language
* guidance changes
* corporate events
* financial news sentiment
* risk-factor disclosures
* changes in management tone

The objective is not simply to label text as positive or negative.

The objective is to determine whether language-derived information provides **incremental predictive information after controlling for existing quantitative factors**.

---

# Financial RAG

Orbit's RAG system is designed for research rather than generic question answering.

```text
Document
   ↓
Parsing
   ↓
Chunking
   ↓
Metadata
   ↓
Embedding
   ↓
Vector Index
   +
BM25 Index
   ↓
Hybrid Retrieval
   ↓
Reciprocal-Rank Fusion
   ↓
Cross-Encoder Reranking
   ↓
Evidence Selection
   ↓
LLM Synthesis
   ↓
Citation Validation
   ↓
Numerical Grounding
```

### Metadata filtering

Documents can be filtered by information such as:

* company
* filing type
* filing date
* fiscal period
* document section
* publication date

Metadata is treated as a hard constraint where required.

### Hybrid retrieval

Dense retrieval handles semantic similarity.

BM25 handles exact financial terminology, identifiers, dates and numerical expressions.

The results are fused before reranking.

### Reranking

A cross-encoder evaluates retrieved passages more precisely before they are passed to the language model.

### Citation validation

Generated research answers must point back to source evidence.

### Numerical grounding

Financial numbers in generated answers are checked against retrieved source text.

Unsupported figures are flagged rather than silently accepted.

---

# Multi-agent research system

Orbit uses agents for **workflow orchestration**, not as decorative chatbots.

Each agent has a defined responsibility and structured state.

```text
Research Planner
       ↓
Market Data Agent
       ↓
Financial Documents / RAG Agent
       ↓
NLP / News Agent
       ↓
Feature Research Agent
       ↓
Quant Research Agent
       ↓
ML Modeling Agent
       ↓
Backtesting Agent
       ↓
Risk Agent
       ↓
Portfolio Agent
       ↓
Validation / Critic Agent
       ↓
Research Reporting Agent
```

Agents operate through explicit tools and structured inputs/outputs.

The orchestration layer controls:

* permissions
* state transitions
* validation
* retries
* failures
* evidence requirements
* run provenance
* approval gates

The LLM can decide **what should be investigated**.

It does not independently decide statistical significance, portfolio sizing, or investment approval.

---

# Human approval

Orbit deliberately stops before autonomous investment decisions.

A completed research run reaches:

```text
awaiting_approval
```

rather than an autonomous "execute trade" state.

A human reviewer can inspect:

* data sources
* assumptions
* features
* model comparison
* backtest
* risk
* citations
* limitations
* robustness tests

Only an authenticated human can approve the research output.

Orbit is a **research platform, not a trading system or financial adviser**.

---

# Backtesting

The backtest engine consumes the model's **out-of-sample predictions**.

This distinction matters.

The research pipeline is:

```text
Features
   ↓
Train model
   ↓
Generate OOS predictions
   ↓
Convert predictions → signal
   ↓
Portfolio / position construction
   ↓
Backtest
```

Not:

```text
Train model
   ↓
Look at historical predictions
   ↓
Invent strategy
```

The backtester incorporates:

* commissions / spread assumptions
* slippage
* market impact
* turnover
* borrow assumptions
* portfolio constraints
* benchmark comparison
* exposure analysis

---

# Portfolio and risk

Orbit separates prediction quality from portfolio quality.

A model can predict returns reasonably well and still produce a poor portfolio.

The portfolio layer therefore evaluates:

### Performance

* cumulative return
* annualized return
* volatility
* Sharpe ratio
* Sortino ratio
* maximum drawdown
* hit rate
* turnover

### Risk

* beta
* volatility
* downside risk
* factor exposures
* sector concentration
* position concentration
* drawdown behavior

### Portfolio construction

Model signals are translated into portfolio weights through explicit optimization constraints rather than arbitrary position sizing.

---

# Research reports

Each research run produces a structured research artifact containing:

1. Research question
2. Hypothesis
3. Dataset definition
4. Data sources
5. Feature definitions
6. Availability assumptions
7. Validation methodology
8. Model configuration
9. Out-of-sample results
10. Backtest results
11. Transaction-cost assumptions
12. Risk analysis
13. Robustness tests
14. RAG evidence and citations
15. Critic findings
16. Limitations
17. Reproducibility metadata
18. Human approval state

The report is generated from structured upstream results.

The LLM does not invent the numerical results.

---

# Architecture

```text
                           ┌──────────────────────┐
                           │      Next.js UI      │
                           │ Research Workspace   │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │       FastAPI        │
                           │ REST API + Auth      │
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
              Research Runs       Documents        Portfolio
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                           ┌──────────────────────┐
                           │     LangGraph        │
                           │ Research Workflow    │
                           └──────────┬───────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
      Data & Features            RAG / NLP                ML Research
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      ▼
                           ┌──────────────────────┐
                           │   Quant Engine       │
                           │ Backtest + Risk      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Validation / Critic  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Research Report      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Human Approval       │
                           └──────────────────────┘


 Data Layer
 ─────────────────────────────────────────────────────────────

 Yahoo Finance ─┐
 SEC EDGAR ─────┤
 FRED ──────────┼──► Validation ─► Canonical Data ─► Features
 News ──────────┘                              │
                                               ▼
                                         PostgreSQL
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              Research Data          pgvector
                                                         │
                                                         ▼
                                                   RAG Retrieval


 MLOps
 ─────────────────────────────────────────────────────────────

 Dataset
    ↓
Feature Version
    ↓
Experiment
    ↓
Model
    ↓
Evaluation
    ↓
MLflow Registry
    ↓
Deployment
    ↓
Monitoring
```

---

# Technology stack

Orbit deliberately avoids unnecessary infrastructure.

| Layer         | Technology                           | Purpose                                     |
| ------------- | ------------------------------------ | ------------------------------------------- |
| Frontend      | Next.js + TypeScript                 | Research workspace                          |
| API           | FastAPI                              | Typed backend API                           |
| Database      | PostgreSQL                           | Durable application and research data       |
| Vector search | pgvector                             | Financial document embeddings               |
| ORM           | SQLAlchemy                           | Database access                             |
| Async jobs    | Arq + Redis                          | Long-running research/data jobs             |
| Quant         | NumPy / pandas / SciPy / statsmodels | Statistical research                        |
| ML            | scikit-learn / XGBoost               | Tabular modeling                            |
| Deep learning | PyTorch                              | LSTM / Transformer research                 |
| NLP           | Transformers / FinBERT               | Financial language modeling                 |
| RAG           | pgvector + BM25 + reranker           | Evidence retrieval                          |
| Agents        | LangGraph                            | Stateful orchestration                      |
| LLM           | Claude                               | Planning, synthesis and research assistance |
| Experiments   | MLflow                               | Experiment and model tracking               |
| Storage       | S3-compatible object storage         | Reports and large artifacts                 |
| Containers    | Docker                               | Reproducible deployment                     |
| CI/CD         | GitHub Actions                       | Automated testing and deployment            |
| Monitoring    | OpenTelemetry + structured logs      | Production observability                    |

The architecture intentionally avoids Kubernetes, multiple agent frameworks, multiple vector databases, and unnecessary microservices.

---

# Data architecture

The data lifecycle is:

```text
External Source
      ↓
Raw Ingestion
      ↓
Validation
      ↓
Provenance
      ↓
Canonical Dataset
      ↓
Point-in-Time Alignment
      ↓
Feature Engineering
      ↓
Feature Validation
      ↓
Research Dataset Version
      ↓
Training / Evaluation
      ↓
Model Artifact
      ↓
Out-of-Sample Predictions
      ↓
Backtest
      ↓
Research Result
```

Every research result should be traceable back to:

```text
source
→ dataset version
→ feature version
→ model version
→ experiment configuration
→ backtest configuration
→ final report
```

This is the foundation of reproducibility.

---

# Data sources

Orbit currently integrates financial and economic sources including:

* Yahoo Finance
* SEC EDGAR / XBRL
* FRED
* financial news

Data ingestion is validated and provenance-aware.

When an external source is unavailable, Orbit uses clearly labelled fallbacks rather than silently pretending that synthetic or stale data is real.

---

# API surface

The backend is organized around research resources rather than exposing internal implementation details.

Representative API domains include:

```text
/api/auth
/api/runs
/api/runs/{id}
/api/research
/api/data
/api/features
/api/models
/api/backtests
/api/portfolio
/api/risk
/api/documents
/api/rag
/api/agents
/api/monitoring
/api/approvals
```

Long-running operations such as ingestion, training and backtesting should execute asynchronously and expose job status through the API.

---

# Frontend

The frontend is designed as a research workstation rather than a generic analytics dashboard.

Core research views include:

### Research Workspace

* research question
* experiment configuration
* pipeline status
* evidence
* findings
* limitations

### Model Comparison

* baseline performance
* XGBoost
* LSTM
* Transformer
* fold-by-fold metrics
* calibration
* stability
* statistical significance

### Backtest

* equity curve
* benchmark comparison
* drawdowns
* turnover
* costs
* rolling Sharpe
* exposure

### Portfolio & Risk

* weights
* concentration
* factor exposures
* beta
* volatility
* drawdown
* risk contribution

### Document Research

* search
* source metadata
* retrieved passages
* citations
* document sections
* grounded answers

### Agent Activity

* current stage
* tool execution
* latency
* validation
* failures
* retries

### Research Reports

* methodology
* evidence
* model results
* backtest
* risk
* limitations
* reproducibility information
* approval state

---

# Reproducibility

A research result should not depend on somebody remembering which settings were used.

Orbit tracks research configuration such as:

```text
Research Run ID
Dataset Version
Feature Version
Model Version
Random Seed
Training Window
Validation Windows
Test Window
Universe
Transaction Costs
Slippage
Portfolio Constraints
LLM Configuration
Prompt Version
Code Version
```

The goal is:

> **Same inputs + same configuration + same code version → reproducible research result.**

---

# MLOps

Orbit treats ML as a production lifecycle.

```text
Dataset
   ↓
Experiment
   ↓
Training
   ↓
Evaluation
   ↓
Model Registry
   ↓
Validation
   ↓
Serving
   ↓
Monitoring
```

Tracked metrics include:

* model performance
* validation performance
* prediction distribution
* feature drift
* dataset drift
* inference latency
* model failures
* agent latency
* agent failures
* LLM usage/cost

MLflow is used for experiment tracking and model lifecycle management.

---

# Security

Production deployments should include:

* JWT authentication
* role-based permissions
* secret management
* API rate limiting
* input validation
* audit logging
* restricted agent tools
* human approval for sensitive actions
* no credentials stored in source control

Financial research systems should assume that model outputs are **untrusted data** until validated.

---

# Testing

Orbit tests the system at multiple levels.

### Unit tests

Individual:

* feature calculations
* statistical functions
* validators
* data transformations
* portfolio calculations

### Data tests

* schema validation
* missingness
* duplicate observations
* timestamp ordering
* point-in-time availability
* source provenance

### Quantitative tests

* leakage tests
* fold boundaries
* forward-label construction
* transaction-cost calculations
* portfolio constraints
* statistical inference

### Integration tests

* API
* database
* research orchestration
* RAG retrieval
* model pipeline

### End-to-end tests

```text
Research Question
      ↓
Data
      ↓
Features
      ↓
Models
      ↓
Backtest
      ↓
Risk
      ↓
Report
      ↓
Approval
```

The objective is not simply high test coverage.

The objective is to test the **failure modes that could make financial research misleading**.

---

# Quick start

## Backend

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

## Frontend

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

## Tests

```bash
cd backend
pytest -q
```

---

# Configuration

Important environment variables include:

| Variable                 | Purpose                                |
| ------------------------ | -------------------------------------- |
| `DATABASE_URL`           | PostgreSQL / development database      |
| `JWT_SECRET`             | Authentication secret                  |
| `AUTH_REQUIRED`          | Enable authentication                  |
| `ANTHROPIC_API_KEY`      | LLM research capabilities              |
| `SEC_USER_AGENT`         | SEC API identification                 |
| `ALLOW_LIVE_MARKET_DATA` | Enable live external market data       |
| `ALLOW_MODEL_DOWNLOAD`   | Enable pretrained NLP/embedding models |
| `ENABLE_MLFLOW`          | Enable experiment tracking             |
| `MAX_CONCURRENT_RUNS`    | Research worker concurrency            |

Secrets must never be committed to Git.

---

# Repository structure

```text
Orbit/
│
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── api/
│   │   ├── core/
│   │   ├── data/
│   │   ├── models_dl/
│   │   ├── orchestration/
│   │   └── services/
│   │
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
│   ├── architecture/
│   ├── research-methodology/
│   ├── api/
│   └── reproducibility/
│
├── .github/
│   └── workflows/
│
├── docker/
│
└── README.md
```

---

# Example research question

A representative Orbit experiment could ask:

> **Does combining value, momentum, earnings sentiment and macro-regime features improve risk-adjusted performance for large-cap equities compared with a traditional factor baseline?**

Orbit should then:

1. Define the universe and research period.
2. Collect point-in-time market and fundamental data.
3. Retrieve relevant filings and financial documents.
4. Extract NLP-derived signals.
5. Construct leakage-safe features.
6. Validate data coverage.
7. Train baseline and ML models.
8. Generate out-of-sample predictions.
9. Convert predictions into portfolio signals.
10. Run a cost-aware backtest.
11. Compare risk-adjusted performance.
12. Run robustness and statistical tests.
13. Ask the critic to challenge the result.
14. Generate a cited research report.
15. Present the result for human approval.

The important outcome may be:

> **No statistically significant improvement was found.**

That is a valid research result.

Orbit is designed to make that conclusion visible rather than optimize for an impressive-looking backtest.

---

# What makes Orbit different

### Not an LLM wrapper

LLMs are used for planning, evidence synthesis and research assistance—not for fabricating quantitative conclusions.

### Not a stock-price predictor

The system evaluates whether information produces **incremental, out-of-sample investment signal**.

### Not a backtest toy

Backtests incorporate explicit assumptions for costs, slippage, impact, turnover and risk.

### Not a generic RAG chatbot

Financial retrieval uses metadata constraints, hybrid retrieval, reranking, citations and numerical grounding.

### Not an autonomous trading bot

Orbit stops at human approval and does not place trades.

### Not "more models = better"

Complex models must beat simpler baselines under the same evaluation protocol.

---

# Known limitations

Orbit is a research platform and its results should not be interpreted as evidence of guaranteed investment performance.

Current limitations include:

### Survivorship bias

The historical universe currently relies on currently available securities, meaning delisted companies may be absent from historical experiments.

### Transaction-cost assumptions

Market impact and trading costs are modelled assumptions rather than observed execution costs.

### Borrow assumptions

Short borrow availability and cost are simplified and may materially underestimate costs for hard-to-borrow securities.

### News history

Free news sources may have limited historical depth. Orbit prefers dropping a poorly covered feature over silently backfilling it.

### Data quality

External financial APIs can change availability, schemas, rate limits and historical coverage.

### Execution

Orbit does not provide broker connectivity, order management or live trading.

### Infrastructure

Development can run locally with a lightweight database, while production deployments should use PostgreSQL and asynchronous workers.

---

# Research philosophy

Orbit follows five principles:

### 1. Evidence over narrative

Every important research conclusion should trace back to data or source evidence.

### 2. Out-of-sample over in-sample

A model's historical fit is not evidence of predictive value.

### 3. Simplicity must be beaten

Complex models earn their place by improving robustly over strong baselines.

### 4. Negative results are valuable

A failed hypothesis is useful if the experiment was well designed.

### 5. AI assists research; it does not replace judgment

The system can investigate, retrieve, synthesize and challenge.

A human remains responsible for approval.

---

# Project status

Orbit is being developed incrementally from research prototype toward production-grade quantitative research infrastructure.

Current implementation areas include:

* research orchestration
* financial data ingestion
* point-in-time feature engineering
* ML model comparison
* time-series validation
* cost-aware backtesting
* RAG
* financial NLP
* portfolio analytics
* risk analysis
* experiment tracking
* monitoring
* authenticated human approval

Production hardening continues across persistence, asynchronous execution, data versioning, observability, security and deployment.

---

# Roadmap

## Phase 1 — Quantitative research foundation

* [x] Point-in-time feature framework
* [x] Leakage-aware validation
* [x] Model comparison
* [x] Out-of-sample prediction pipeline
* [x] Cost-aware backtesting
* [x] Statistical evaluation
* [ ] Full dataset versioning
* [ ] Persistent research-run history
* [ ] Survivorship-bias-aware universe construction

## Phase 2 — Financial intelligence

* [x] SEC document retrieval
* [x] Hybrid RAG
* [x] Financial NLP
* [ ] Historical news pipeline
* [ ] Event extraction
* [ ] Improved citation/evidence graph

## Phase 3 — Production research platform

* [ ] PostgreSQL production deployment
* [ ] Redis + asynchronous workers
* [ ] MLflow model registry
* [ ] Dataset lineage
* [ ] Model serving
* [ ] Drift monitoring
* [ ] Production authentication/RBAC
* [ ] CI/CD hardening

## Phase 4 — Research workstation

* [ ] Research workspace
* [ ] Experiment comparison
* [ ] Interactive factor analysis
* [ ] Backtest comparison
* [ ] Portfolio/risk dashboard
* [ ] Document research workspace
* [ ] Agent execution trace
* [ ] Reproducible research reports

---

# Disclaimer

Orbit is a software and quantitative research project.

It is **not investment advice, a broker, a trading system, or a guarantee of future investment performance**.

Historical backtests are subject to assumptions, data limitations, model risk and statistical uncertainty.
