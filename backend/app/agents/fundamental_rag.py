"""
Fundamental Analysis Agent (RAG over financial documents)
=========================================================
Retrieval-augmented extraction from filings, with metadata filtering, hybrid
retrieval, cross-encoder reranking, citation validation, and -- the part most
RAG implementations skip -- NUMERIC GROUNDING VERIFICATION.

HALLUCINATION CONTROL, IN FOUR LAYERS
-------------------------------------
1. RETRIEVE NARROWLY. Metadata filters (ticker, form type, fiscal period,
   date range) are a hard pre-filter. Semantic similarity ranks only within
   documents that are already the right company and period, because a
   beautifully similar sentence from the wrong issuer is a wrong answer.

2. CITE STRUCTURALLY. Every claim must carry a chunk_id. Citations are
   validated against the set actually retrieved, so a fabricated reference is
   detected rather than trusted.

3. VERIFY EVERY NUMBER. This is the layer that matters most for financial
   text and the one the original build lacked. Any figure appearing in the
   answer is re-scanned against the cited chunks; a number that is not present
   verbatim (allowing for formatting variants like 1,234.5 / $1.2 billion) is
   flagged as UNGROUNDED and the answer is marked unverified. An LLM asked
   about revenue will happily produce a plausible number that appears nowhere
   in the source, and nothing upstream of this check would catch it.

4. FALL BACK TO EVIDENCE, NOT PROSE. With no LLM configured the agent returns
   the retrieved passages themselves rather than a synthesized answer. Showing
   the evidence is honest; inventing a summary is not.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.models import Chunk, Document
from app.core.vectorstore import (
    chunk_text,
    content_hash,
    embed_texts,
    from_bytes,
    search,
    to_bytes,
)

# Matches figures the way filings write them: $1.2 billion, 1,234.5, 12.3%, 3.4x
NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\$\s?)?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|bn|mm|%|x))?(?![\w])",
    re.I,
)
CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+::\d+)\]")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_document(
    db: Session,
    text: str,
    title: str,
    ticker: str | None = None,
    doc_type: str = "other",
    fiscal_period: str | None = None,
    filing_date: str | None = None,
    source_url: str | None = None,
    source: str = "upload",
    company_name: str | None = None,
) -> dict:
    """Chunk, embed, and persist a document. Idempotent on content hash."""
    digest = content_hash(text)
    existing = db.execute(select(Document).where(Document.content_hash == digest)).scalar_one_or_none()
    if existing:
        return {
            "status": "already_ingested",
            "document_id": existing.id,
            "n_chunks": len(existing.chunks),
            "title": existing.title,
        }

    doc = Document(
        ticker=(ticker or "").upper() or None,
        company_name=company_name,
        doc_type=doc_type,
        fiscal_period=fiscal_period,
        filing_date=filing_date,
        title=title[:512],
        source_url=source_url,
        source=source,
        char_count=len(text),
        content_hash=digest,
    )
    db.add(doc)
    db.flush()

    pieces = chunk_text(text, doc_id=doc.id[:8])
    vectors, model_name = embed_texts([p["text"] for p in pieces])

    for i, piece in enumerate(pieces):
        vec = vectors[i] if vectors is not None else None
        db.add(
            Chunk(
                id=f"{doc.id[:8]}::{piece['chunk_index']}",
                document_id=doc.id,
                chunk_index=piece["chunk_index"],
                text=piece["text"],
                char_count=piece["char_count"],
                section=piece["section"],
                embedding=to_bytes(vec) if vec is not None else None,
                embedding_model=model_name,
                embedding_dim=int(len(vec)) if vec is not None else None,
            )
        )

    db.commit()
    return {
        "status": "ingested",
        "document_id": doc.id,
        "n_chunks": len(pieces),
        "embedded": vectors is not None,
        "embedding_model": model_name,
        "title": doc.title,
    }


def ingest_edgar_filings(
    db: Session, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 2
) -> dict:
    """Pull real filings from SEC EDGAR into the corpus."""
    from app.data.providers.edgar import fetch_filing_text, list_filings

    results, errors = [], []
    try:
        filings = list_filings(ticker, forms=forms, limit=limit)
    except Exception as e:
        return {"status": "failed", "ticker": ticker, "error": f"{type(e).__name__}: {e}"}

    for f in filings:
        try:
            text = fetch_filing_text(f)
            if len(text) < 2000:
                errors.append({"url": f["url"], "error": "document too short after HTML stripping"})
                continue
            results.append(
                ingest_document(
                    db,
                    text=text,
                    title=f"{f['ticker']} {f['form']} filed {f['filing_date']}",
                    ticker=f["ticker"],
                    doc_type=f["form"],
                    fiscal_period=f.get("report_date"),
                    filing_date=f["filing_date"],
                    source_url=f["url"],
                    source="sec_edgar",
                    company_name=f.get("company_name"),
                )
            )
        except Exception as e:
            errors.append({"url": f.get("url"), "error": f"{type(e).__name__}: {str(e)[:120]}"})

    return {
        "status": "ok" if results else "failed",
        "ticker": ticker,
        "n_documents": len(results),
        "documents": results,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    db: Session,
    query: str,
    tickers: list[str] | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 6,
    use_reranker: bool = True,
) -> dict:
    """Metadata-filtered hybrid retrieval with reranking."""
    stmt = select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
    if tickers:
        stmt = stmt.where(Document.ticker.in_([t.upper() for t in tickers]))
    if doc_types:
        stmt = stmt.where(Document.doc_type.in_(doc_types))
    if date_from:
        stmt = stmt.where(Document.filing_date >= date_from)
    if date_to:
        stmt = stmt.where(Document.filing_date <= date_to)

    rows = db.execute(stmt).all()
    if not rows:
        return {
            "status": "no_documents",
            "filters": {"tickers": tickers, "doc_types": doc_types, "date_from": date_from, "date_to": date_to},
            "hits": [],
            "note": "No documents match these filters. Retrieval was not attempted -- "
            "returning nothing is correct here; widening the filter silently would "
            "answer about a different company or period.",
        }

    chunks = [r[0] for r in rows]
    docs = {r[1].id: r[1] for r in rows}
    texts = [c.text for c in chunks]

    embeddings = None
    dims = {c.embedding_dim for c in chunks if c.embedding is not None and c.embedding_dim}
    if len(dims) == 1:
        dim = dims.pop()
        vecs = [from_bytes(c.embedding, dim) if c.embedding else np.zeros(dim, dtype=np.float32) for c in chunks]
        embeddings = np.vstack(vecs)

    hits = search(query, texts, embeddings, top_k=top_k, use_reranker=use_reranker)

    enriched = []
    for h in hits:
        c = chunks[h["index"]]
        d = docs[c.document_id]
        enriched.append(
            {
                "chunk_id": c.id,
                "text": c.text,
                "section": c.section,
                "document_id": d.id,
                "ticker": d.ticker,
                "company_name": d.company_name,
                "doc_type": d.doc_type,
                "filing_date": d.filing_date,
                "fiscal_period": d.fiscal_period,
                "source_url": d.source_url,
                "scores": {k: h[k] for k in ("score", "dense_score", "lexical_score", "rerank_score", "fusion_score")},
                "retrieval_stage": h["stage"],
            }
        )

    return {
        "status": "ok",
        "n_candidates": len(chunks),
        "n_returned": len(enriched),
        "retrieval_mode": "dense+bm25 fused" + (", cross-encoder reranked" if use_reranker else ""),
        "embeddings_available": embeddings is not None,
        "filters": {"tickers": tickers, "doc_types": doc_types, "date_from": date_from, "date_to": date_to},
        "hits": enriched,
    }


# ---------------------------------------------------------------------------
# Numeric grounding
# ---------------------------------------------------------------------------
def _normalize_number(token: str) -> str | None:
    """Reduce a figure to a comparable canonical form.

    '$1,234.50' and '1234.5' are the same number written two ways, and a
    verifier that treats them as different produces false alarms that train
    the reader to ignore it.
    """
    t = token.strip().lower().replace("$", "").replace(",", "").strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(million|billion|trillion|bn|mm|%|x)?$", t)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").strip()
    multiplier = {"million": 1e6, "mm": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}.get(unit)
    if multiplier:
        value *= multiplier
        unit = ""
    if value == int(value):
        return f"{int(value)}{unit}"
    return f"{value:g}{unit}"


def verify_numeric_grounding(answer: str, cited_chunks: list[dict]) -> dict:
    """Check that every number in the answer appears in the cited source text.

    This is the strongest single guard against a confidently wrong figure, and
    it is cheap: extract the numbers from the answer, extract them from the
    evidence, and compare canonical forms.
    """
    source_text = " ".join(c["text"] for c in cited_chunks)
    source_numbers = {
        n for n in (_normalize_number(t) for t in NUMBER_RE.findall(source_text)) if n
    }

    answer_tokens = NUMBER_RE.findall(answer)
    grounded, ungrounded = [], []
    for token in answer_tokens:
        norm = _normalize_number(token)
        if norm is None:
            continue
        # Ignore small integers -- they are usually years, list indices, or
        # quarter numbers rather than claims about financial magnitudes.
        bare = norm.rstrip("%x")
        try:
            if bare and abs(float(bare)) < 13 and "." not in bare:
                continue
        except ValueError:
            pass
        (grounded if norm in source_numbers else ungrounded).append(token.strip())

    return {
        "n_numbers_in_answer": len(grounded) + len(ungrounded),
        "n_grounded": len(grounded),
        "n_ungrounded": len(ungrounded),
        "ungrounded_values": sorted(set(ungrounded))[:12],
        "fully_grounded": len(ungrounded) == 0,
        "method": "Every figure in the answer is re-scanned against the cited chunk "
        "text after normalizing formatting ($1.2 billion == 1200000000). A figure "
        "not present verbatim in the evidence is reported as ungrounded.",
    }


def validate_citations(answer: str, retrieved: list[dict]) -> dict:
    valid_ids = {c["chunk_id"] for c in retrieved}
    cited = set(CITATION_RE.findall(answer))
    return {
        "cited_chunk_ids": sorted(cited & valid_ids),
        "fabricated_citations": sorted(cited - valid_ids),
        "n_claims_cited": len(cited),
        "all_citations_valid": len(cited - valid_ids) == 0,
        "uncited_answer": len(cited) == 0,
    }


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def _extractive_answer(hits: list[dict]) -> dict:
    return {
        "mode": "extractive_evidence",
        "answer": None,
        "note": "No LLM configured. Returning the retrieved evidence verbatim rather "
        "than a synthesized answer, so no number is produced that did not come from a "
        "source document.",
        "evidence": [
            {
                "chunk_id": h["chunk_id"],
                "ticker": h["ticker"],
                "doc_type": h["doc_type"],
                "filing_date": h["filing_date"],
                "section": h["section"],
                "source_url": h["source_url"],
                "relevance": h["scores"]["score"],
                "text": h["text"][:900],
                "numbers_found": sorted({t.strip() for t in NUMBER_RE.findall(h["text"])})[:10],
            }
            for h in hits
        ],
    }


def _llm_answer(query: str, hits: list[dict]) -> dict:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic()

    context = "\n\n".join(
        f"[{h['chunk_id']}] (source: {h['ticker']} {h['doc_type']} filed {h['filing_date']}"
        + (f", section: {h['section']}" if h.get("section") else "")
        + f")\n{h['text']}"
        for h in hits
    )
    prompt = (
        "You are a financial research assistant. Answer the question using ONLY facts "
        "and figures that appear verbatim in the context below.\n\n"
        "RULES:\n"
        "1. Cite the chunk_id in brackets after every factual claim, e.g. [a1b2c3d4::7].\n"
        "2. Never state a number that does not appear in the context. Do not compute, "
        "estimate, annualize, or infer figures.\n"
        '3. If the answer is not in the context, say exactly: "Not found in the '
        'provided documents." Do not speculate.\n'
        "4. Note the fiscal period each figure belongs to -- a revenue number is "
        "meaningless without knowing which quarter it describes.\n\n"
        f"Question: {query}\n\nContext:\n{context}"
    )

    resp = client.messages.create(
        model=settings.llm_model, max_tokens=900, messages=[{"role": "user", "content": prompt}]
    )
    answer = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    return {
        "mode": "llm_extraction",
        "answer": answer,
        "model": settings.llm_model,
        "usage": {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        },
    }


def run_fundamental_agent(
    db: Session,
    query: str,
    tickers: list[str] | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 6,
) -> dict:
    """Answer a question over the document corpus with verified citations."""
    retrieval = retrieve(
        db, query, tickers=tickers, doc_types=doc_types, date_from=date_from, date_to=date_to, top_k=top_k
    )
    if retrieval["status"] != "ok" or not retrieval["hits"]:
        return {
            "status": "no_evidence",
            "query": query,
            "retrieval": retrieval,
            "answer": None,
            "note": "No relevant passages found. The agent does not answer without evidence.",
        }

    hits = retrieval["hits"]
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if use_llm:
        try:
            generated = _llm_answer(query, hits)
        except Exception as e:
            generated = _extractive_answer(hits)
            generated["note"] += f" (LLM call failed: {type(e).__name__}: {str(e)[:120]})"
    else:
        generated = _extractive_answer(hits)

    verification = {"citations": None, "numeric_grounding": None, "verdict": "evidence_only"}
    if generated.get("answer"):
        citations = validate_citations(generated["answer"], hits)
        cited = [h for h in hits if h["chunk_id"] in set(citations["cited_chunk_ids"])]
        numeric = verify_numeric_grounding(generated["answer"], cited or hits)

        if not citations["all_citations_valid"]:
            verdict = "REJECTED_FABRICATED_CITATION"
        elif not numeric["fully_grounded"]:
            verdict = "UNVERIFIED_UNGROUNDED_NUMBERS"
        elif citations["uncited_answer"]:
            verdict = "UNVERIFIED_NO_CITATIONS"
        else:
            verdict = "VERIFIED"

        verification = {"citations": citations, "numeric_grounding": numeric, "verdict": verdict}

    return {
        "status": "ok",
        "query": query,
        "filters_applied": retrieval["filters"],
        "retrieval_mode": retrieval["retrieval_mode"],
        "n_candidates_considered": retrieval["n_candidates"],
        "generation": generated,
        "verification": verification,
        "sources": [
            {
                "chunk_id": h["chunk_id"],
                "ticker": h["ticker"],
                "doc_type": h["doc_type"],
                "filing_date": h["filing_date"],
                "section": h["section"],
                "source_url": h["source_url"],
                "relevance": h["scores"]["score"],
                "excerpt": h["text"][:400],
            }
            for h in hits
        ],
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------
def evaluate_retrieval(db: Session, cases: list[dict], top_k: int = 6) -> dict:
    """Measure retrieval quality on labelled cases.

    Each case: {query, expect_substring, tickers?, doc_types?}. Reports hit
    rate and mean reciprocal rank. Without this, "we added a reranker" is an
    assertion; with it, it is a measurement -- the same standard the ML agent
    is held to.
    """
    results = []
    for case in cases:
        r = retrieve(
            db,
            case["query"],
            tickers=case.get("tickers"),
            doc_types=case.get("doc_types"),
            top_k=top_k,
        )
        hits = r.get("hits", [])
        needle = case["expect_substring"].lower()
        rank = next((i + 1 for i, h in enumerate(hits) if needle in h["text"].lower()), None)
        results.append(
            {
                "query": case["query"],
                "found": rank is not None,
                "rank": rank,
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
                "n_hits": len(hits),
            }
        )

    n = len(results)
    return {
        "status": "ok",
        "n_cases": n,
        "hit_rate_at_k": round(sum(r["found"] for r in results) / n, 3) if n else None,
        "mean_reciprocal_rank": round(sum(r["reciprocal_rank"] for r in results) / n, 3) if n else None,
        "top_k": top_k,
        "cases": results,
    }
