"""Request/response schemas for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    display_name: str = Field(default="", max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: dict


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
class ResearchRunRequest(BaseModel):
    """A research request is a QUESTION, not a parameter dump.

    The planner turns it into a validated plan. `overrides` exists for the
    cases where a researcher already knows what they want and should not have
    to phrase it as prose -- it is merged into the plan after validation, so an
    override still cannot violate a bound.
    """

    question: str = Field(
        min_length=10,
        max_length=800,
        examples=[
            "Do value and quality factors improve risk-adjusted returns in large-cap US equities?"
        ],
    )
    overrides: dict = Field(default_factory=dict)
    use_llm_planner: bool = True


class RunSummary(BaseModel):
    id: str
    question: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    total_seconds: float | None = None
    sharpe: float | None = None
    cagr: float | None = None
    max_drawdown: float | None = None
    critic_verdict: str | None = None
    n_critic_flags: int | None = None
    best_model: str | None = None
    is_synthetic: bool | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------
class ApprovalDecision(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    feedback: str = Field(default="", max_length=4000)


# ---------------------------------------------------------------------------
# Documents / RAG
# ---------------------------------------------------------------------------
class DocumentQuery(BaseModel):
    query: str = Field(min_length=3, max_length=600)
    tickers: list[str] | None = None
    doc_types: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    top_k: int = Field(default=6, ge=1, le=25)


class IngestFilingsRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=8)
    forms: list[str] = Field(default=["10-K", "10-Q"], max_length=4)
    limit: int = Field(default=2, ge=1, le=6)


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=400)
    text: str = Field(min_length=200)
    ticker: str | None = None
    doc_type: str = "other"
    filing_date: str | None = None


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class BookCreate(BaseModel):
    name: str = Field(default="Default book", max_length=120)
    notional: float = Field(default=1_000_000.0, gt=0)
    positions: dict[str, float] = Field(default_factory=dict)


class AdoptWeightsRequest(BaseModel):
    run_id: str
    method: str = Field(default="risk_parity")
    book_name: str = Field(default="Default book", max_length=120)
