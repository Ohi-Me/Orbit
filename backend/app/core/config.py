"""
Central configuration
=====================
Every capability that depends on an external service (market data vendor,
HuggingFace model hub, an LLM provider, Postgres) is behind a flag that is
*detected*, never assumed. The platform's standing discipline is that a
missing dependency degrades to a documented, deterministic fallback and
says so in the output -- it never silently pretends the real thing ran.

Read `effective_capabilities()` to see what a given deployment can actually
do; it is surfaced verbatim at GET /api/health so the UI can label degraded
runs instead of showing them as full-fidelity.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # ---- storage -------------------------------------------------------
    # SQLite by default so the platform runs with zero infrastructure.
    # Point DATABASE_URL at Postgres in deployment; the ORM layer is
    # identical either way (see core/db.py for the two dialect differences).
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./quant_platform.db")
    artifact_dir: str = os.environ.get("ARTIFACT_DIR", "./artifacts")

    # ---- auth ----------------------------------------------------------
    # Dev default is a fixed string so local runs work out of the box; any
    # real deployment MUST set JWT_SECRET (health check reports if it didn't).
    jwt_secret: str = os.environ.get("JWT_SECRET", "dev-only-insecure-secret-change-me")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "720"))
    auth_required: bool = _flag("AUTH_REQUIRED", False)

    # ---- external data -------------------------------------------------
    # SEC requires a self-identifying UA on every request; they rate-limit
    # or block generic ones. Override with your own contact per their policy.
    sec_user_agent: str = os.environ.get(
        "SEC_USER_AGENT", "agentic-quant-research-platform research@example.com"
    )
    allow_live_market_data: bool = _flag("ALLOW_LIVE_MARKET_DATA", True)
    allow_live_filings: bool = _flag("ALLOW_LIVE_FILINGS", True)
    allow_live_macro: bool = _flag("ALLOW_LIVE_MACRO", True)

    # ---- models --------------------------------------------------------
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    llm_model: str = os.environ.get("LLM_MODEL", "claude-sonnet-5")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    finbert_model: str = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")
    allow_model_download: bool = _flag("ALLOW_MODEL_DOWNLOAD", True)

    # ---- experiment tracking -------------------------------------------
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow_experiment: str = os.environ.get("MLFLOW_EXPERIMENT", "quant-research")
    enable_mlflow: bool = _flag("ENABLE_MLFLOW", True)

    # ---- api -----------------------------------------------------------
    # Localhost dev ports are allowed by default so the frontend works out of
    # the box on whichever port Next picks when 3000 is already taken. Set
    # CORS_ORIGINS explicitly in any real deployment -- this default is for
    # local development only and permits nothing outside localhost.
    cors_origins: list[str] = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3100,http://127.0.0.1:3000,http://127.0.0.1:3100",
    ).split(",")

    # An exact-match list is unworkable against hosts that mint a new URL per
    # deployment -- Vercel gives every push its own
    # <project>-<hash>-<scope>.vercel.app, so a hardcoded list breaks on the
    # next commit. CORS_ORIGIN_REGEX is matched against the Origin header in
    # addition to the list above.
    #
    # SCOPE IT TO YOUR PROJECT. Because credentials are allowed, a pattern like
    # ^https://.*\.vercel\.app$ would let ANY site hosted on vercel.app make
    # authenticated requests against this API on behalf of a logged-in user.
    # Anchor it to your own project AND scope, e.g. for project "orbit-web"
    # under the "ohi-me" scope:
    #
    #   ^https://orbit-web(-[a-z0-9-]+)?-ohi-me\.vercel\.app$
    #
    # Note the hyphen inside the character class. Vercel's branch URLs embed
    # the branch name, so main deploys to orbit-web-git-main-ohi-me.vercel.app
    # -- a pattern of [a-z0-9]+ without the hyphen silently fails to match it,
    # which looks exactly like the CORS setting having no effect at all.
    cors_origin_regex: str | None = os.environ.get("CORS_ORIGIN_REGEX") or None
    max_concurrent_runs: int = int(os.environ.get("MAX_CONCURRENT_RUNS", "2"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def effective_capabilities() -> dict:
    """What this deployment can genuinely do right now.

    Every value here is probed, not declared. The frontend renders these as
    badges so a user can tell a real-data run from a synthetic one at a
    glance rather than reading the fine print in the report.
    """
    s = get_settings()
    return {
        "database": "postgresql" if s.database_url.startswith("postgres") else "sqlite",
        "live_market_data": s.allow_live_market_data and _module_available("yfinance"),
        "live_filings": s.allow_live_filings,
        "live_macro": s.allow_live_macro,
        "llm": bool(s.anthropic_api_key) and _module_available("anthropic"),
        "neural_embeddings": s.allow_model_download and _module_available("sentence_transformers"),
        "reranker": s.allow_model_download and _module_available("sentence_transformers"),
        "finbert_sentiment": s.allow_model_download and _module_available("transformers"),
        "deep_learning": _module_available("torch"),
        "experiment_tracking": s.enable_mlflow and _module_available("mlflow"),
        "graph_orchestration": _module_available("langgraph"),
        "jwt_secret_is_default": s.jwt_secret == "dev-only-insecure-secret-change-me",
    }
