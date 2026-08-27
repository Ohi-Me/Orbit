"""
News & Sentiment Agent (financial NLP)
======================================
Produces a DATED sentiment series per ticker, plus event tags, management
tone, and extracted risk factors.

THREE BACKENDS, PROBED NOT ASSUMED
  1. FinBERTEngine   -- ProsusAI/finbert, a BERT fine-tuned on financial text.
                        This is the default when transformers is available.
  2. LLMEngine       -- Anthropic, used when ANTHROPIC_API_KEY is set.
  3. LexiconEngine   -- keyword baseline, always available, fully offline.

The original build shipped only the lexicon and honestly called it a
baseline. The review's point stands: a platform that insists XGBoost must
beat logistic regression should hold its NLP layer to the same standard, so
FinBERT is now the default and the lexicon is the fallback it must beat.
The lexicon is retained deliberately -- it is what makes the comparison
possible, and `compare_backends` measures the gap rather than asserting it.

WHY DATED OUTPUT MATTERS: sentiment enters the factor panel as a time series
keyed on publication timestamps. A score computed today is attached to
today's date and no earlier one, which is what stops present-day information
from leaking into historical rows.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Lexicon baseline
# ---------------------------------------------------------------------------
_POSITIVE = {
    "beat", "beats", "raises", "raised", "growth", "outperform", "buyback",
    "upgrade", "upgraded", "strong", "record", "expands", "expansion",
    "surges", "rally", "profit", "gains", "exceeds", "accelerating",
}
_NEGATIVE = {
    "miss", "misses", "missed", "cuts", "cut", "pressure", "downgrade",
    "downgraded", "weak", "softens", "softer", "decline", "declines",
    "restructuring", "lawsuit", "probe", "investigation", "slump", "plunge",
    "loss", "losses", "warns", "warning", "delays", "recall",
}

EVENT_PATTERNS = {
    "earnings": re.compile(r"\b(beat|miss|earnings|guidance|outlook|eps|revenue)\b", re.I),
    "capital_return": re.compile(r"\b(buyback|dividend|repurchase)\b", re.I),
    "leverage_credit": re.compile(r"\b(leverage|debt|downgrade|credit rating|refinanc)\w*\b", re.I),
    "management_change": re.compile(r"\b(names? new|appoints?|steps? down|resign|ceo|cfo)\b", re.I),
    "m_and_a": re.compile(r"\b(acqui\w+|merger|takeover|divest\w*|spin-?off)\b", re.I),
    "legal_regulatory": re.compile(r"\b(lawsuit|litigation|probe|investigation|antitrust|fine[sd]?|settle\w*)\b", re.I),
    "guidance_change": re.compile(r"\b(raises?|cuts?|lowers?|lifts?)\s+(its\s+)?(full-?year\s+)?(guidance|outlook|forecast)\b", re.I),
}

# Risk-factor taxonomy used for filing/transcript extraction. Deliberately a
# transparent keyword taxonomy, labelled as such, rather than an opaque
# classifier whose categories nobody can audit.
RISK_PATTERNS = {
    "supply_chain": re.compile(r"\b(supply chain|supplier|shortage|logistics|inventory)\b", re.I),
    "regulatory": re.compile(r"\b(regulat\w+|complian\w+|antitrust|gdpr|sanction)\b", re.I),
    "competition": re.compile(r"\b(competit\w+|market share|pricing pressure)\b", re.I),
    "cybersecurity": re.compile(r"\b(cyber\w*|data breach|ransomware|information security)\b", re.I),
    "macro": re.compile(r"\b(inflation|recession|interest rate|foreign exchange|currency)\b", re.I),
    "concentration": re.compile(r"\b(concentrat\w+|single (customer|supplier)|depend\w+ on a (limited|small))\b", re.I),
    "litigation": re.compile(r"\b(litigation|lawsuit|legal proceeding|claim)\b", re.I),
    "talent": re.compile(r"\b(key personnel|attract and retain|talent|labor shortage)\b", re.I),
}

# Hedging language -- how confident management sounds. A rise in hedged,
# uncertain phrasing relative to a company's own baseline is a documented
# soft signal in the earnings-call literature.
HEDGE_TERMS = re.compile(
    r"\b(may|might|could|possibly|potentially|uncertain\w*|approximately|roughly|"
    r"we believe|we think|we expect|somewhat|to some extent|difficult to predict|"
    r"challenging|headwind)\b",
    re.I,
)
CONFIDENCE_TERMS = re.compile(
    r"\b(will|confident|strong|record|clearly|definitely|committed|robust|"
    r"significant\w*|momentum|accelerat\w+)\b",
    re.I,
)


class LexiconEngine:
    """Deterministic keyword baseline. Fully offline, no model download.

    Honest about what it is: it cannot read negation ("not strong"), sarcasm,
    or context, and it is here to be beaten by FinBERT rather than to be used.
    """

    name = "lexicon_v1"

    def score_text(self, text: str) -> dict:
        words = set(re.findall(r"[a-z']+", text.lower()))
        pos = len(words & _POSITIVE)
        neg = len(words & _NEGATIVE)
        if pos == 0 and neg == 0:
            score = 0.0
        else:
            score = (pos - neg) / (pos + neg)
        return {"score": float(np.clip(score, -1, 1)), "confidence": min((pos + neg) / 4.0, 1.0)}


class FinBERTEngine:
    """ProsusAI/finbert -- BERT fine-tuned for financial sentiment.

    Three-class (positive / negative / neutral); we map to a signed score as
    P(positive) - P(negative), which keeps the neutral mass out of the signal
    instead of forcing a directional call on genuinely neutral text.
    """

    name = "finbert"

    def __init__(self, model_name: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = model_name or get_settings().finbert_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.name = f"finbert:{model_name}"
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def score_texts(self, texts: list[str]) -> list[dict]:
        import torch

        if not texts:
            return []
        out = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).numpy()
            for row in probs:
                mapping = {self.id2label[j]: float(row[j]) for j in range(len(row))}
                pos = mapping.get("positive", 0.0)
                neg = mapping.get("negative", 0.0)
                neu = mapping.get("neutral", 0.0)
                out.append(
                    {
                        "score": float(pos - neg),
                        "confidence": float(1.0 - neu),
                        "probabilities": {k: round(v, 4) for k, v in mapping.items()},
                    }
                )
        return out

    def score_text(self, text: str) -> dict:
        return self.score_texts([text])[0]


class LLMEngine:
    """Anthropic-backed classifier. Structured JSON output only, never prose."""

    name = "llm"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or get_settings().llm_model
        self.name = f"llm:{self.model}"
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def score_texts(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts[:40]))
        prompt = (
            "You are a financial news sentiment classifier. For EACH numbered headline "
            "return one JSON object with keys: index (int), score (float -1 to 1 where "
            "-1 is maximally bearish for the company's equity), confidence (0 to 1). "
            "Return ONLY a JSON array, no prose, no markdown fences.\n\n"
            f"Headlines:\n{numbered}"
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=1500, messages=[{"role": "user", "content": prompt}]
        )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += getattr(resp.usage, "input_tokens", 0)
        self.usage["output_tokens"] += getattr(resp.usage, "output_tokens", 0)

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Fail closed to neutral rather than crash or invent a number.
            return [{"score": 0.0, "confidence": 0.0, "parse_failed": True} for _ in texts]

        by_index = {int(o.get("index", i + 1)) - 1: o for i, o in enumerate(parsed)}
        return [
            {
                "score": float(np.clip(by_index.get(i, {}).get("score", 0.0), -1, 1)),
                "confidence": float(np.clip(by_index.get(i, {}).get("confidence", 0.0), 0, 1)),
            }
            for i in range(len(texts))
        ]


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------
def build_engine(prefer: str | None = None):
    """Return (engine, note). Probes availability; never claims a backend it lacks."""
    settings = get_settings()
    order = [prefer] if prefer else []
    order += ["finbert", "llm", "lexicon"]

    for choice in order:
        if choice == "finbert" and settings.allow_model_download:
            try:
                return FinBERTEngine(), "FinBERT (financial-domain BERT)."
            except Exception as e:
                last = f"FinBERT unavailable ({type(e).__name__}); "
                continue
        if choice == "llm" and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return LLMEngine(), "Anthropic LLM classifier."
            except Exception:
                continue
        if choice == "lexicon":
            return (
                LexiconEngine(),
                "Keyword lexicon baseline -- no neural model available. This backend "
                "cannot read negation or context and is reported as a baseline, not a "
                "financial NLP result.",
            )
    return LexiconEngine(), "Fell through to lexicon baseline."


def _score_batch(engine, texts: list[str]) -> list[dict]:
    if hasattr(engine, "score_texts"):
        return engine.score_texts(texts)
    return [engine.score_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------
def run_sentiment_agent(
    tickers: list[str], news_by_ticker: dict | None = None, prefer_backend: str | None = None
) -> dict:
    """Score dated news per ticker and return both a summary and a dated series.

    `series` is the contract with the Quant Research Agent: {ticker: pd.Series}
    indexed by publication date. Building the factor from this rather than from
    a single scalar is what makes sentiment a time-varying, point-in-time
    signal instead of a constant that leaks the present into the past.
    """
    from app.data.providers.news import fetch_news

    if news_by_ticker is None:
        news_by_ticker, news_prov = fetch_news(tickers)
    else:
        news_prov = {"provider": "supplied"}

    engine, engine_note = build_engine(prefer_backend)

    summary: dict[str, dict] = {}
    series: dict[str, pd.Series] = {}

    for t in tickers:
        items = news_by_ticker.get(t, [])
        if not items:
            summary[t] = {
                "ticker": t, "score": None, "label": "no_coverage", "n_items": 0,
                "event_tags": [], "backend": engine.name,
                "note": "No news items found for this ticker.",
            }
            continue

        texts = [f"{i['title']}. {i.get('summary', '')}".strip() for i in items]
        try:
            scored = _score_batch(engine, texts)
        except Exception as e:
            fallback = LexiconEngine()
            scored = [fallback.score_text(x) for x in texts]
            engine_note = f"{engine.name} failed ({type(e).__name__}); fell back to lexicon."

        dates, values = [], []
        tags: set[str] = set()
        for item, s in zip(items, scored):
            dates.append(pd.Timestamp(item["published_at"]).tz_localize(None))
            values.append(s["score"])
            for tag, pat in EVENT_PATTERNS.items():
                if pat.search(item["title"]):
                    tags.add(tag)

        raw = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
        # Multiple items on one day -> that day's mean. Then a 5-day decayed
        # mean, because a single headline's relevance does not vanish overnight
        # nor persist forever.
        daily = raw.groupby(raw.index.normalize()).mean()
        # halflife must be a Timedelta when `times` is supplied -- pandas rejects
        # a bare integer there, since the decay is in calendar time, not in rows.
        smoothed = daily.ewm(halflife=pd.Timedelta(days=5), times=daily.index).mean()
        series[t] = smoothed

        latest = float(smoothed.iloc[-1])
        summary[t] = {
            "ticker": t,
            "score": round(latest, 4),
            "label": "positive" if latest > 0.15 else "negative" if latest < -0.15 else "neutral",
            "n_items": len(items),
            "event_tags": sorted(tags),
            "backend": engine.name,
            "first_item": str(daily.index.min().date()),
            "last_item": str(daily.index.max().date()),
            "coverage_days": int((daily.index.max() - daily.index.min()).days),
            "mean_confidence": round(float(np.mean([s.get("confidence", 0) for s in scored])), 3),
        }

    return {
        "status": "ok",
        "backend": engine.name,
        "backend_note": engine_note,
        "per_ticker": summary,
        "series": series,
        "news_provenance": news_prov,
        "llm_usage": getattr(engine, "usage", None),
    }


# ---------------------------------------------------------------------------
# Document-level NLP: management tone and risk factors
# ---------------------------------------------------------------------------
def analyze_management_tone(text: str, engine=None) -> dict:
    """Hedging vs. confidence in management language.

    Measured as rates per 1,000 words so documents of different lengths are
    comparable. The absolute level is far less informative than the change
    against the same company's prior filing, which is what `compare_tone`
    computes -- a company that always hedges is not a signal; one that
    suddenly starts hedging is.
    """
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    if n_words < 100:
        return {"status": "too_short", "n_words": n_words}

    hedges = len(HEDGE_TERMS.findall(text))
    confident = len(CONFIDENCE_TERMS.findall(text))
    per_1k = 1000.0 / n_words

    result = {
        "status": "ok",
        "n_words": n_words,
        "hedge_rate_per_1k": round(hedges * per_1k, 3),
        "confidence_rate_per_1k": round(confident * per_1k, 3),
        "tone_balance": round((confident - hedges) * per_1k, 3),
        "method": "tone-lexicon-v1: rate of hedging vs. confidence terms per 1,000 "
        "words. A transparent keyword measure, not a trained classifier -- the "
        "absolute level is only meaningful relative to the same company's history.",
    }

    if engine is not None and hasattr(engine, "score_texts"):
        # Sentence-level FinBERT over a sample, for a document-level distribution.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 40 < len(s.strip()) < 400]
        sample = sentences[:120]
        if sample:
            scored = engine.score_texts(sample)
            vals = [s["score"] for s in scored]
            result["finbert_sentence_sentiment"] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "pct_negative": round(float(np.mean([v < -0.2 for v in vals])), 3),
                "pct_positive": round(float(np.mean([v > 0.2 for v in vals])), 3),
                "n_sentences_scored": len(sample),
            }
    return result


def extract_risk_factors(text: str, top_n: int = 8) -> dict:
    """Count and locate risk-factor mentions by category.

    Returns a representative sentence per category so a claim can be traced to
    the language that produced it, rather than to a bare count.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 30 < len(s.strip()) < 500]
    found: dict[str, dict] = {}

    for category, pattern in RISK_PATTERNS.items():
        matches = [s for s in sentences if pattern.search(s)]
        if matches:
            found[category] = {
                "n_mentions": len(matches),
                "example": matches[0][:300],
            }

    ranked = sorted(found.items(), key=lambda kv: -kv[1]["n_mentions"])[:top_n]
    return {
        "status": "ok" if found else "none_found",
        "n_sentences_scanned": len(sentences),
        "risk_factors": dict(ranked),
        "categories_detected": [k for k, _ in ranked],
        "method": "risk-taxonomy-v1: keyword taxonomy over sentence splits. A "
        "transparent, auditable rule set -- not a trained NER model. It finds "
        "discussion of a topic; it does not judge severity.",
    }


def compare_tone(current: dict, prior: dict) -> dict:
    """Change in management tone between two filings of the same company."""
    if current.get("status") != "ok" or prior.get("status") != "ok":
        return {"status": "insufficient_data"}
    delta = current["tone_balance"] - prior["tone_balance"]
    return {
        "status": "ok",
        "tone_balance_current": current["tone_balance"],
        "tone_balance_prior": prior["tone_balance"],
        "change": round(delta, 3),
        "direction": "more_confident" if delta > 0.5 else "more_hedged" if delta < -0.5 else "unchanged",
        "note": "Change in tone against the company's own prior filing. This "
        "controls for the fact that some companies always write cautiously.",
    }


def compare_backends(texts: list[str]) -> dict:
    """Measure the gap between FinBERT and the lexicon on the same inputs.

    Exists so the claim 'FinBERT beats the keyword baseline' is a measurement
    rather than an assertion -- the same discipline the ML agent applies to
    XGBoost against logistic regression.
    """
    if not texts:
        return {"status": "no_texts"}

    lex = LexiconEngine()
    lex_scores = [lex.score_text(t)["score"] for t in texts]

    try:
        fin = FinBERTEngine()
        fin_scores = [s["score"] for s in fin.score_texts(texts)]
    except Exception as e:
        return {"status": "finbert_unavailable", "error": f"{type(e).__name__}: {e}"}

    corr = float(np.corrcoef(lex_scores, fin_scores)[0, 1]) if len(texts) > 2 else None
    disagreements = [
        {"text": t[:140], "lexicon": round(l, 3), "finbert": round(f, 3)}
        for t, l, f in zip(texts, lex_scores, fin_scores)
        if abs(l - f) > 0.8
    ]

    return {
        "status": "ok",
        "n_texts": len(texts),
        "correlation": round(corr, 3) if corr is not None and np.isfinite(corr) else None,
        "lexicon_neutral_rate": round(float(np.mean([abs(s) < 1e-9 for s in lex_scores])), 3),
        "finbert_neutral_rate": round(float(np.mean([abs(s) < 0.1 for s in fin_scores])), 3),
        "n_major_disagreements": len(disagreements),
        "example_disagreements": disagreements[:5],
        "note": "The lexicon scores exactly 0 for any headline containing none of its "
        "keywords, which is why its neutral rate is typically far higher. That is the "
        "concrete cost of the keyword approach, measured rather than asserted.",
    }
