"""
Research Planner Agent
======================
Turns a research question into a validated, executable plan.

WHY A PLANNER RATHER THAN A FIXED PIPELINE
-------------------------------------------
The original build had no planner -- the "plan" was the hardcoded call order
in the orchestrator, and the user's question was never read by anything. That
made the multi-agent framing decorative: ten functions called in a fixed
sequence is a pipeline, not an agent system.

The planner reads the question and decides what the run should actually do:
which universe, how much history, which factor families are relevant, whether
documents need ingesting, which models are worth training. Its output is a
STRUCTURED, VALIDATED plan -- never free text -- and every field is bounded,
so a bad plan is rejected before anything expensive runs.

TWO BACKENDS, same contract:
  * RuleBasedPlanner -- deterministic keyword routing. Always available.
  * LLMPlanner       -- Anthropic, returns JSON matched to the same schema and
                        validated against it. If validation fails, the rule-
                        based plan is used and the failure is recorded.

THE LLM NEVER MAKES A FINANCIAL DECISION HERE. It chooses what to *research*
-- universe, factor families, horizon. It cannot size a position, approve a
strategy, or alter a statistical threshold; those are code paths with fixed
rules and a human approval gate.
"""

from __future__ import annotations

import json
import os
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

# Small, curated universes so a question can name a theme rather than tickers.
UNIVERSE_PRESETS = {
    "mega_cap_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"],
    "large_cap_diversified": [
        "AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "UNH", "HD", "CVX", "KO",
        "PFE", "CSCO", "VZ", "WMT", "MRK", "BAC", "ABT", "CRM", "TMO", "COST",
    ],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "USB"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "HAL"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "LLY", "BMY", "AMGN", "GILD"],
    "consumer": ["PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "TGT", "HD"],
}

FACTOR_FAMILIES = {
    "momentum": ["momentum_12_1", "momentum_3m"],
    "volatility": ["volatility_20d", "volatility_60d", "downside_beta_proxy"],
    "mean_reversion": ["mean_reversion_z"],
    "liquidity": ["liquidity_log_dollar_vol"],
    "value": ["earnings_yield", "book_to_price"],
    "quality": ["return_on_equity", "operating_margin", "leverage_ratio"],
    "growth": ["asset_growth", "earnings_surprise"],
    "sentiment": ["sentiment_score"],
    "macro": ["vix_percentile", "term_spread", "credit_spread"],
}


class ResearchPlan(BaseModel):
    """The validated contract every downstream agent reads.

    Bounds are enforced here rather than defensively re-checked in each agent:
    an unbounded n_days or a 500-name universe would be an accidental
    denial-of-service on a synchronous research run.
    """

    question: str
    universe: list[str] = Field(min_length=2, max_length=60)
    universe_rationale: str = ""
    n_days: int = Field(default=1008, ge=252, le=3024)
    label_horizon: int = Field(default=21, ge=5, le=63)
    n_folds: int = Field(default=4, ge=2, le=8)
    factor_families: list[str] = Field(min_length=1)
    include_deep_learning: bool = True
    ingest_filings: bool = False
    filing_tickers: list[str] = Field(default_factory=list, max_length=8)
    document_question: str | None = None
    max_weight: float = Field(default=0.35, gt=0.05, le=1.0)
    rebalance_days: int = Field(default=21, ge=5, le=63)
    planner_backend: str = "rule_based"
    notes: list[str] = Field(default_factory=list)

    @field_validator("universe")
    @classmethod
    def _upper_unique(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for t in v:
            t = t.strip().upper()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        if len(out) < 2:
            raise ValueError("universe needs at least 2 distinct tickers")
        return out

    @field_validator("factor_families")
    @classmethod
    def _known_families(cls, v: list[str]) -> list[str]:
        unknown = [f for f in v if f not in FACTOR_FAMILIES]
        if unknown:
            raise ValueError(f"unknown factor families: {unknown}")
        return v

    def resolved_factors(self) -> list[str]:
        out: list[str] = []
        for fam in self.factor_families:
            out.extend(FACTOR_FAMILIES[fam])
        return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Rule-based planner
# ---------------------------------------------------------------------------
TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_STOPWORDS = {
    "A", "I", "THE", "AND", "OR", "IS", "IT", "TO", "IN", "ON", "OF", "FOR",
    "DO", "DOES", "ML", "AI", "US", "VS", "AN", "BE", "BY", "AT", "IF", "SO",
}


def _detect_universe(question: str) -> tuple[list[str], str]:
    q = question.lower()
    for name, tickers in UNIVERSE_PRESETS.items():
        if name.replace("_", " ") in q:
            return tickers, f"Matched the '{name}' preset from the question."

    keyword_map = {
        "tech": "mega_cap_tech", "technology": "mega_cap_tech", "software": "mega_cap_tech",
        "bank": "financials", "financial": "financials",
        "energy": "energy", "oil": "energy",
        "health": "healthcare", "pharma": "healthcare", "biotech": "healthcare",
        "consumer": "consumer", "retail": "consumer",
    }
    for kw, preset in keyword_map.items():
        if kw in q:
            return UNIVERSE_PRESETS[preset], f"Question mentions '{kw}'; used the '{preset}' preset."

    explicit = [t for t in TICKER_RE.findall(question) if t not in _STOPWORDS and len(t) >= 2]
    if len(explicit) >= 2:
        return explicit[:60], "Tickers named explicitly in the question."

    return (
        UNIVERSE_PRESETS["large_cap_diversified"],
        "No universe specified. Defaulted to a 20-name diversified large-cap set, "
        "which is wide enough for a cross-sectional strategy to mean something.",
    )


def _detect_factor_families(question: str) -> tuple[list[str], list[str]]:
    q = question.lower()
    notes = []
    families = []

    triggers = {
        "momentum": ["momentum", "trend", "price action"],
        "value": ["value", "cheap", "valuation", "p/e", "earnings yield", "book"],
        "quality": ["quality", "profitab", "margin", "roe", "balance sheet", "leverage"],
        "volatility": ["volatil", "risk", "beta", "drawdown"],
        "mean_reversion": ["mean revers", "reversal", "oversold", "contrarian"],
        "growth": ["growth", "surprise", "expansion"],
        "sentiment": ["sentiment", "news", "tone"],
        "macro": ["macro", "regime", "rates", "inflation", "vix", "spread"],
        "liquidity": ["liquid", "volume", "turnover"],
    }
    for fam, words in triggers.items():
        if any(w in q for w in words):
            families.append(fam)

    if not families:
        families = ["momentum", "value", "quality", "volatility", "macro"]
        notes.append(
            "Question named no specific factor family. Used a standard multi-factor "
            "set (momentum, value, quality, volatility, macro) so the run tests a "
            "recognised hypothesis rather than an arbitrary one."
        )
    return families, notes


def rule_based_plan(question: str, overrides: dict | None = None) -> ResearchPlan:
    overrides = overrides or {}
    universe, uni_note = _detect_universe(question)
    families, notes = _detect_factor_families(question)
    q = question.lower()

    if "sentiment" in families:
        notes.append(
            "Sentiment was requested, but free news sources cover only recent weeks. "
            "The factor-coverage gate will drop it if it cannot span the backtest "
            "window rather than broadcasting recent sentiment across history."
        )

    wants_docs = any(w in q for w in ["filing", "10-k", "10-q", "risk factor", "management", "disclosure", "transcript"])

    plan_kwargs = {
        "question": question,
        "universe": universe,
        "universe_rationale": uni_note,
        "factor_families": families,
        "ingest_filings": wants_docs,
        "filing_tickers": universe[:3] if wants_docs else [],
        "document_question": question if wants_docs else None,
        "include_deep_learning": "deep learning" in q or "lstm" in q or "transformer" in q or "neural" in q or True,
        "notes": notes,
        "planner_backend": "rule_based",
    }

    if "short" in q and "history" in q:
        plan_kwargs["n_days"] = 504
    if "weekly" in q:
        plan_kwargs["label_horizon"] = 5
        plan_kwargs["rebalance_days"] = 5
    elif "quarterly" in q:
        plan_kwargs["label_horizon"] = 63
        plan_kwargs["rebalance_days"] = 63

    plan_kwargs.update(overrides)
    return ResearchPlan(**plan_kwargs)


# ---------------------------------------------------------------------------
# LLM planner
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You plan quantitative equity research runs. You choose WHAT TO \
RESEARCH. You never make investment decisions, never size positions, and never set \
statistical thresholds.

Return ONLY a JSON object with these keys:
  universe            array of 2-60 US equity tickers
  universe_rationale  one sentence on why this universe fits the question
  n_days              integer 252-3024, trading days of history
  label_horizon       integer 5-63, forward-return horizon in trading days
  n_folds             integer 2-8, walk-forward folds
  factor_families     array from: momentum, volatility, mean_reversion, liquidity, value, quality, growth, sentiment, macro
  include_deep_learning  boolean
  ingest_filings      boolean, true only if the question needs filing TEXT
  filing_tickers      array of at most 8 tickers, only if ingest_filings
  document_question   string or null
  max_weight          float 0.05-1.0
  rebalance_days      integer 5-63
  notes               array of strings: caveats a researcher should know

Rules you must follow:
- A cross-sectional long/short strategy needs breadth. Never propose fewer than 15 \
tickers unless the question names specific companies.
- Match label_horizon to the question's stated horizon; default 21.
- Only set ingest_filings when the question is about document CONTENT (risk factors, \
management discussion, disclosures), not about prices or factors.
No prose, no markdown fences."""


def llm_plan(question: str, overrides: dict | None = None) -> tuple[ResearchPlan, dict]:
    """Ask the LLM for a plan, then validate it against the schema.

    Validation is the point: an LLM plan that violates a bound is discarded in
    favour of the rule-based plan rather than being partially repaired, so a
    malformed plan can never half-configure a run.
    """
    import anthropic

    from app.core.config import get_settings

    client = anthropic.Anthropic()
    settings = get_settings()
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=1200,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": f"Research question: {question}"}],
    )
    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", 0),
        "output_tokens": getattr(resp.usage, "output_tokens", 0),
        "calls": 1,
    }
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()

    data = json.loads(text)
    data["question"] = question
    data["planner_backend"] = f"llm:{settings.llm_model}"
    data.update(overrides or {})
    return ResearchPlan(**data), usage


def run_planner(question: str, overrides: dict | None = None, prefer_llm: bool = True) -> dict:
    """Produce a validated plan and report which backend produced it."""
    fallback_reason = None
    usage = None

    if prefer_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            plan, usage = llm_plan(question, overrides)
            return {
                "status": "ok",
                "plan": plan.model_dump(),
                "resolved_factors": plan.resolved_factors(),
                "backend": plan.planner_backend,
                "llm_usage": usage,
            }
        except ValidationError as e:
            fallback_reason = f"LLM plan failed schema validation: {e.error_count()} error(s). {str(e)[:200]}"
        except Exception as e:
            fallback_reason = f"LLM planner unavailable: {type(e).__name__}: {str(e)[:150]}"

    plan = rule_based_plan(question, overrides)
    if fallback_reason:
        plan.notes.append(fallback_reason)
    return {
        "status": "ok",
        "plan": plan.model_dump(),
        "resolved_factors": plan.resolved_factors(),
        "backend": plan.planner_backend,
        "fallback_reason": fallback_reason,
        "llm_usage": usage,
    }


def revise_plan(plan_dict: dict, revisions: list[str]) -> tuple[dict, list[str]]:
    """Apply the Critic's recommended revisions to a plan.

    This is what makes the Critic's verdict load-bearing: the graph feeds these
    revisions back into the plan and re-runs, rather than emitting a label and
    proceeding identically. Each revision is a bounded, explicit edit.
    """
    plan = ResearchPlan(**plan_dict)
    applied = []

    for rev in revisions:
        if rev == "widen_universe" and len(plan.universe) < 20:
            merged = list(dict.fromkeys(plan.universe + UNIVERSE_PRESETS["large_cap_diversified"]))
            plan.universe = merged[:30]
            applied.append(
                f"Widened the universe from {len(plan_dict['universe'])} to {len(plan.universe)} "
                "names -- a cross-sectional strategy on a handful of names is a bet on "
                "those companies, not on a factor."
            )
        elif rev == "lengthen_history" and plan.n_days < 3024:
            plan.n_days = min(int(plan.n_days * 1.5), 3024)
            applied.append(f"Lengthened history to {plan.n_days} trading days for more folds.")
        elif rev == "drop_leaking_factors" and "sentiment" in plan.factor_families:
            plan.factor_families = [f for f in plan.factor_families if f != "sentiment"]
            applied.append(
                "Dropped the sentiment family: its coverage could not span the backtest "
                "window, which is the condition under which it leaks present-day "
                "information into historical rows."
            )
        elif rev == "use_live_data":
            applied.append(
                "Live market data requested. This is an environment setting "
                "(ALLOW_LIVE_MARKET_DATA), not a plan field -- flagged for the operator."
            )
        elif rev == "reduce_strategy_count":
            plan.include_deep_learning = False
            applied.append(
                "Disabled the deep-learning tier to cut the number of strategies tested, "
                "which lowers the multiple-testing threshold the result must clear."
            )

    plan.notes.extend(applied)
    return plan.model_dump(), applied
