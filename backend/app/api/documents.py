"""Document corpus and RAG routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.agents import fundamental_rag as rag
from app.core.db import get_db
from app.core.models import Chunk, Document
from app.schemas import DocumentQuery, IngestFilingsRequest, IngestTextRequest

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def list_documents(
    ticker: str | None = None,
    doc_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    stmt = select(Document).order_by(desc(Document.ingested_at))
    if ticker:
        stmt = stmt.where(Document.ticker == ticker.upper())
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    docs = db.execute(stmt.limit(limit)).scalars().all()

    counts = dict(
        db.execute(select(Chunk.document_id, func.count(Chunk.id)).group_by(Chunk.document_id)).all()
    )
    return {
        "documents": [
            {
                "id": d.id,
                "ticker": d.ticker,
                "company_name": d.company_name,
                "doc_type": d.doc_type,
                "title": d.title,
                "filing_date": d.filing_date,
                "fiscal_period": d.fiscal_period,
                "source": d.source,
                "source_url": d.source_url,
                "char_count": d.char_count,
                "n_chunks": counts.get(d.id, 0),
                "ingested_at": d.ingested_at,
            }
            for d in docs
        ],
        "total": db.execute(select(func.count(Document.id))).scalar_one(),
    }


@router.get("/stats")
def corpus_stats(db: Session = Depends(get_db)):
    """Corpus composition -- what the RAG layer can actually answer about."""
    n_docs = db.execute(select(func.count(Document.id))).scalar_one()
    n_chunks = db.execute(select(func.count(Chunk.id))).scalar_one()
    n_embedded = db.execute(
        select(func.count(Chunk.id)).where(Chunk.embedding.is_not(None))
    ).scalar_one()
    by_type = dict(
        db.execute(select(Document.doc_type, func.count(Document.id)).group_by(Document.doc_type)).all()
    )
    by_ticker = dict(
        db.execute(
            select(Document.ticker, func.count(Document.id))
            .group_by(Document.ticker)
            .order_by(desc(func.count(Document.id)))
            .limit(25)
        ).all()
    )
    models = [
        m for m in db.execute(select(Chunk.embedding_model).distinct()).scalars().all() if m
    ]
    return {
        "n_documents": n_docs,
        "n_chunks": n_chunks,
        "n_chunks_embedded": n_embedded,
        "embedding_coverage": round(n_embedded / n_chunks, 3) if n_chunks else 0.0,
        "embedding_models": models,
        "by_doc_type": by_type,
        "by_ticker": by_ticker,
        "note": "Chunks without embeddings are still retrievable by BM25 but not by "
        "semantic similarity. Mixed embedding models mean vectors from different "
        "spaces are being compared -- re-embed if more than one appears here.",
    }


@router.post("/ingest/edgar")
def ingest_edgar(req: IngestFilingsRequest, db: Session = Depends(get_db)):
    """Pull real filings from SEC EDGAR into the corpus."""
    result = rag.ingest_edgar_filings(db, req.ticker, forms=tuple(req.forms), limit=req.limit)
    if result["status"] == "failed" and not result.get("n_documents"):
        raise HTTPException(
            status_code=502,
            detail=f"Could not ingest filings for {req.ticker}: {result.get('error') or result.get('errors')}",
        )
    return result


@router.post("/ingest/text")
def ingest_text(req: IngestTextRequest, db: Session = Depends(get_db)):
    return rag.ingest_document(
        db,
        text=req.text,
        title=req.title,
        ticker=req.ticker,
        doc_type=req.doc_type,
        filing_date=req.filing_date,
        source="upload",
    )


@router.post("/search")
def search(req: DocumentQuery, db: Session = Depends(get_db)):
    """Retrieval only -- ranked passages with their scores, no generation."""
    return rag.retrieve(
        db,
        req.query,
        tickers=req.tickers,
        doc_types=req.doc_types,
        date_from=req.date_from,
        date_to=req.date_to,
        top_k=req.top_k,
    )


@router.post("/ask")
def ask(req: DocumentQuery, db: Session = Depends(get_db)):
    """Retrieval plus a cited, numerically-verified answer."""
    return rag.run_fundamental_agent(
        db,
        req.query,
        tickers=req.tickers,
        doc_types=req.doc_types,
        date_from=req.date_from,
        date_to=req.date_to,
        top_k=req.top_k,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    db.delete(doc)
    db.commit()
