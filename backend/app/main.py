"""
FastAPI application entry point.

The health endpoint reports PROBED capabilities rather than declared ones, so
a client can tell at a glance whether this deployment is doing real work or
running degraded -- and can label a run's results accordingly instead of
presenting synthetic output as though it were live.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, documents, ml, monitoring, portfolio, runs
from app.core.config import effective_capabilities, get_settings
from app.core.db import init_db
from app.core.observability import configure_logging, get_logger

log = get_logger("api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    from app.services.run_service import reap_stale_runs

    reaped = reap_stale_runs()
    caps = effective_capabilities()
    log.info("startup", reaped_stale_runs=reaped["reaped"], capabilities=caps)
    if caps.get("jwt_secret_is_default"):
        log.warning(
            "insecure_jwt_secret",
            detail="JWT_SECRET is the development default. Set it before deploying.",
        )
    yield
    log.info("shutdown")


app = FastAPI(
    title="Orbit",
    description=(
        "Orbit is an end-to-end applied machine-learning platform, demonstrated on "
        "financial data. Multi-agent pipeline orchestration with a validation feedback "
        "loop, point-in-time feature engineering, purged walk-forward model comparison "
        "(linear / gradient boosting / LSTM / Transformer), hybrid retrieval-augmented "
        "generation with citation and numeric verification, drift monitoring, "
        "experiment tracking, and a human approval gate."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins if o.strip()],
    # Matched in addition to the exact list, for hosts that mint a new URL per
    # deployment (Vercel preview builds). See core/config.py for why this must
    # be anchored to your own project rather than all of *.vercel.app.
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and log latency for every call."""
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        log.error(
            "request_failed",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            error=f"{type(e).__name__}: {e}",
            seconds=round(time.time() - start, 3),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )
    elapsed = time.time() - start
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round(elapsed * 1000, 1))
    if elapsed > 1.0:
        log.info(
            "slow_request",
            request_id=request_id,
            path=request.url.path,
            seconds=round(elapsed, 3),
        )
    return response


app.include_router(auth.router)
app.include_router(runs.router)
app.include_router(documents.router)
app.include_router(portfolio.router)
app.include_router(monitoring.router)
app.include_router(ml.router)


@app.get("/api/health", tags=["system"])
def health():
    """Probed capabilities -- what this deployment can genuinely do right now."""
    caps = effective_capabilities()
    degraded = [
        k
        for k in ("live_market_data", "neural_embeddings", "finbert_sentiment", "deep_learning")
        if not caps.get(k)
    ]
    return {
        "status": "ok",
        "version": "2.0.0",
        "capabilities": caps,
        "degraded_capabilities": degraded,
        "fidelity": "full" if not degraded else "degraded",
        "warnings": (
            ["JWT_SECRET is the development default -- set it before deploying."]
            if caps.get("jwt_secret_is_default")
            else []
        ),
        "note": "Every capability above is probed at call time, not declared. A false "
        "value means that path will fall back to a documented, honestly-labelled "
        "substitute rather than silently pretending to run.",
    }


@app.get("/api/config/presets", tags=["system"])
def presets():
    """Universe presets and factor families the planner can select from."""
    from app.agents.planner import FACTOR_FAMILIES, UNIVERSE_PRESETS

    return {
        "universe_presets": {k: {"tickers": v, "size": len(v)} for k, v in UNIVERSE_PRESETS.items()},
        "factor_families": FACTOR_FAMILIES,
    }
