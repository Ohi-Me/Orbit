"""
Database engine and session management.

DIALECT NOTE: this layer runs unchanged on SQLite (the zero-infrastructure
default) and PostgreSQL (set DATABASE_URL). Only two things differ and both
are handled here rather than leaking into the models:

  1. SQLite needs check_same_thread=False because run execution happens on a
     background thread while the request thread may still be reading.
  2. Postgres gets a real connection pool; SQLite does not pool meaningfully.

Vector search: embeddings are stored as raw float32 bytes in a portable
LargeBinary column and searched with brute-force cosine in NumPy. At this
corpus size (hundreds to low thousands of chunks) an exact brute-force scan
is faster than building an approximate index and returns exact neighbours,
so pgvector/Qdrant would be infrastructure without a benefit today. The swap
point is a single function -- core/vectorstore.py::search -- if the corpus
outgrows it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    s = get_settings()
    url = s.database_url
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables if they do not exist. Idempotent, safe to call on boot."""
    from app.core import models  # noqa: F401  (registers mappers on Base)

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
