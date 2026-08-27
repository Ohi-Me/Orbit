"""
ORM models -- the system of record.

The design goal is that a run is *reproducible and comparable*: every run
stores the exact parameters it was given, the capability set that was
actually available when it ran (so a synthetic-data run can never be
mistaken for a live-data one later), a per-agent trace, and the model-level
fold metrics broken out into their own rows so "compare these two runs" is
a SQL query rather than a JSON diff in the client.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(120), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    runs: Mapped[list["Run"]] = relationship(back_populates="user")


# --------------------------------------------------------------------------
# Research runs
# --------------------------------------------------------------------------
class Run(Base):
    """One execution of the research graph.

    status lifecycle:
        queued -> running -> awaiting_approval -> approved / rejected
        queued -> running -> failed

    A run only reaches 'approved' through an explicit human decision (see
    Approval); nothing in the system can self-approve.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)

    question: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # What the deployment could actually do at execution time. Frozen here so
    # a run's fidelity is auditable forever, not inferred from today's config.
    capabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    data_provenance: Mapped[dict] = mapped_column(JSON, default=dict)

    # Full agent outputs. Headline metrics are ALSO denormalized into columns
    # below so history and comparison views are indexable queries.
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)

    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    cagr: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    critic_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    n_critic_flags: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    user: Mapped["User | None"] = relationship(back_populates="runs")
    steps: Mapped[list["RunStep"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunStep.sequence"
    )
    model_results: Mapped[list["ModelResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    approvals: Mapped[list["Approval"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RunStep(Base):
    """One agent's execution within a run -- the agent trace the UI renders.

    Stores retries and LLM cost so an operator can see which agent is slow,
    which is failing, and what the run cost in tokens.
    """

    __tablename__ = "run_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)

    agent: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="ok")  # ok|failed|degraded|skipped
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    attempt: Mapped[int] = mapped_column(Integer, default=1)

    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    degraded_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    run: Mapped["Run"] = relationship(back_populates="steps")


class ModelResult(Base):
    """Per-model, per-fold metrics -- broken out of the result JSON so the
    model-comparison screen and cross-run queries are real SQL."""

    __tablename__ = "model_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    fold: Mapped[int] = mapped_column(Integer)

    train_start: Mapped[str] = mapped_column(String(10), default="")
    train_end: Mapped[str] = mapped_column(String(10), default="")
    test_start: Mapped[str] = mapped_column(String(10), default="")
    test_end: Mapped[str] = mapped_column(String(10), default="")

    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    information_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_mean_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_t_stat: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="model_results")


Index("ix_model_results_run_model", ModelResult.run_id, ModelResult.model_name)


# --------------------------------------------------------------------------
# Human-in-the-loop
# --------------------------------------------------------------------------
class Approval(Base):
    """An explicit human decision on a run's conclusions.

    Nothing in this platform treats a result as decision-grade without a row
    here whose status is 'approved'. The Critic agent can only recommend.
    """

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)

    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    critic_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    critic_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    decided_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    run: Mapped["Run"] = relationship(back_populates="approvals")


# --------------------------------------------------------------------------
# Documents / RAG corpus
# --------------------------------------------------------------------------
class Document(Base):
    """A source document -- a filing, transcript, or uploaded note.

    The metadata columns exist because financial retrieval is almost always
    filtered before it is ranked: an analyst asking about Q1 revenue means
    *this* ticker's *this* period, and a semantically similar sentence from
    the wrong company or wrong year is a wrong answer, not a near miss.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    doc_type: Mapped[str] = mapped_column(String(32), default="other", index=True)
    fiscal_period: Mapped[str | None] = mapped_column(String(16), nullable=True)
    filing_date: Mapped[str | None] = mapped_column(String(10), index=True, nullable=True)

    title: Mapped[str] = mapped_column(String(512), default="")
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="local")  # local|sec_edgar|upload
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class Chunk(Base):
    """A retrievable passage with its embedding.

    'embedding' is raw float32 bytes (portable across SQLite and Postgres);
    'embedding_model' records which model produced it so a model change
    invalidates the right rows instead of silently mixing vector spaces.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)

    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    section: Mapped[str | None] = mapped_column(String(128), nullable=True)

    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")


# --------------------------------------------------------------------------
# Persistent portfolio book
# --------------------------------------------------------------------------
class PortfolioBook(Base):
    """The standing allocation a new signal gets evaluated *against*.

    This is what makes the platform a tool rather than a calculator: a
    proposed strategy is interesting relative to what you already hold.
    """

    __tablename__ = "portfolio_books"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="Default book")
    base_currency: Mapped[str] = mapped_column(String(8), default="USD")
    notional: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    source_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    method: Mapped[str | None] = mapped_column(String(48), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    positions: Mapped[list["Position"]] = relationship(
        back_populates="book", cascade="all, delete-orphan"
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("portfolio_books.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    book: Mapped["PortfolioBook"] = relationship(back_populates="positions")


# --------------------------------------------------------------------------
# Data lineage
# --------------------------------------------------------------------------
class DataSnapshot(Base):
    """Records what data a run actually consumed.

    Reproducibility means being able to say not just "we ran momentum on
    AAPL" but "on prices fetched at this timestamp from this provider,
    covering this window, with these validation warnings".
    """

    __tablename__ = "data_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    dataset: Mapped[str] = mapped_column(String(48))  # prices|fundamentals|macro|news|filings
    provider: Mapped[str] = mapped_column(String(48))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    universe: Mapped[list | None] = mapped_column(JSON, nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
