"""Portfolio book routes -- the standing allocation a new signal is judged against."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import PortfolioBook, Position, Run, User
from app.core.security import get_current_user, require_user
from app.schemas import AdoptWeightsRequest, BookCreate

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def _book_dict(b: PortfolioBook) -> dict:
    return {
        "id": b.id,
        "name": b.name,
        "notional": b.notional,
        "base_currency": b.base_currency,
        "is_active": b.is_active,
        "source_run_id": b.source_run_id,
        "method": b.method,
        "created_at": b.created_at,
        "updated_at": b.updated_at,
        "positions": {p.ticker: round(p.weight, 6) for p in b.positions},
        "n_positions": len(b.positions),
        "gross_exposure": round(sum(abs(p.weight) for p in b.positions), 6),
        "net_exposure": round(sum(p.weight for p in b.positions), 6),
    }


@router.get("/books")
def list_books(db: Session = Depends(get_db), user: User | None = Depends(get_current_user)):
    stmt = select(PortfolioBook).order_by(desc(PortfolioBook.updated_at))
    stmt = stmt.where(
        PortfolioBook.user_id == user.id if user else PortfolioBook.user_id.is_(None)
    )
    return {"books": [_book_dict(b) for b in db.execute(stmt).scalars().all()]}


@router.post("/books", status_code=201)
def create_book(
    req: BookCreate, db: Session = Depends(get_db), user: User | None = Depends(get_current_user)
):
    book = PortfolioBook(
        user_id=user.id if user else None, name=req.name, notional=req.notional
    )
    db.add(book)
    db.flush()
    for ticker, weight in req.positions.items():
        db.add(Position(book_id=book.id, ticker=ticker.upper(), weight=float(weight)))
    db.commit()
    db.refresh(book)
    return _book_dict(book)


@router.post("/books/adopt")
def adopt_weights(
    req: AdoptWeightsRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Adopt a run's proposed allocation into a live book.

    Gated on approval by design: a run that a human has not approved cannot
    become the standing book. This is the one place research output turns into
    something that looks like a position, and it is exactly where the
    human-in-the-loop requirement has to bite.
    """
    run = db.get(Run, req.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run.status != "approved":
        raise HTTPException(
            status_code=409,
            detail=f"Run status is '{run.status}'. Only an approved run's weights can be "
            "adopted into a book -- research output does not become a position without "
            "an explicit human approval.",
        )

    portfolio = ((run.result or {}).get("portfolio_result") or {}).get("walk_forward", {})
    method_result = portfolio.get(req.method)
    if not method_result or method_result.get("status") != "ok":
        available = [k for k, v in portfolio.items() if v.get("status") == "ok"]
        raise HTTPException(
            status_code=400,
            detail=f"Method '{req.method}' has no successful allocation in this run. Available: {available}",
        )

    weights = method_result.get("weights_latest") or {}
    if not weights:
        raise HTTPException(status_code=400, detail="That allocation produced no weights.")

    book = PortfolioBook(
        user_id=user.id,
        name=req.book_name,
        source_run_id=req.run_id,
        method=req.method,
    )
    db.add(book)
    db.flush()
    for ticker, weight in weights.items():
        db.add(Position(book_id=book.id, ticker=ticker.upper(), weight=float(weight)))
    db.commit()
    db.refresh(book)
    return _book_dict(book)


@router.get("/books/{book_id}")
def get_book(book_id: str, db: Session = Depends(get_db)):
    book = db.get(PortfolioBook, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found.")
    return _book_dict(book)


@router.delete("/books/{book_id}", status_code=204)
def delete_book(book_id: str, db: Session = Depends(get_db), user: User = Depends(require_user)):
    book = db.get(PortfolioBook, book_id)
    if book is None or (book.user_id is not None and book.user_id != user.id):
        raise HTTPException(status_code=404, detail="Book not found.")
    db.delete(book)
    db.commit()
