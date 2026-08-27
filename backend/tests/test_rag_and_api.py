"""
RAG grounding tests and API contract tests.

The RAG tests focus on the guarantees that keep a financial answer honest:
metadata filters must be hard, citations must be verifiable, and every number
in an answer must be traceable to source text.

The API tests focus on the contracts a client depends on, and on the two
security properties that matter: run ownership and the approval gate.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Point the app at a throwaway database BEFORE anything imports the engine.
_TMP_DB = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["ALLOW_MODEL_DOWNLOAD"] = "0"  # keep tests offline and fast

from fastapi.testclient import TestClient  # noqa: E402

from app.agents.fundamental_rag import (  # noqa: E402
    _normalize_number,
    ingest_document,
    retrieve,
    run_fundamental_agent,
    validate_citations,
    verify_numeric_grounding,
)
from app.core.db import SessionLocal, init_db  # noqa: E402
from app.core.security import hash_password, verify_password  # noqa: E402
from app.core.vectorstore import BM25, chunk_text, detect_section, tokenize  # noqa: E402
from app.main import app  # noqa: E402

FILING = """
Item 1A. Risk Factors

Our business faces supply chain disruption and competition from larger vendors.
A cybersecurity incident could materially affect operations.

Item 7. Management's Discussion and Analysis

Total net sales were $94,036 million for the quarter ended June 28, 2025, an
increase of 10% compared to the same quarter last year. Research and development
expense was $8,865 million. The effective tax rate was 16.2 percent.

Gross margin was 46.3% for the quarter. Operating income reached $28,202 million.
"""


@pytest.fixture(scope="module")
def db():
    init_db()
    session = SessionLocal()
    ingest_document(
        session,
        text=FILING,
        title="TEST 10-Q filed 2025-08-01",
        ticker="TEST",
        doc_type="10-Q",
        filing_date="2025-08-01",
        source="upload",
    )
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Numeric grounding -- the strongest hallucination guard
# ---------------------------------------------------------------------------
class TestNumericGrounding:
    def test_accepts_a_figure_present_in_the_source(self):
        chunks = [{"text": "Total net sales were $94,036 million for the quarter."}]
        out = verify_numeric_grounding("Net sales were $94,036 million.", chunks)
        assert out["fully_grounded"] is True
        assert out["n_ungrounded"] == 0

    def test_catches_a_fabricated_figure(self):
        """The failure mode that matters: a plausible number that is not in the source."""
        chunks = [{"text": "Total net sales were $94,036 million for the quarter."}]
        out = verify_numeric_grounding("Net sales were $97,500 million, up 12.4%.", chunks)
        assert out["fully_grounded"] is False
        assert out["n_ungrounded"] == 2

    def test_formatting_variants_are_the_same_number(self):
        """$1.2 billion and 1200000000 must not be reported as a discrepancy."""
        chunks = [{"text": "Revenue was $1.2 billion this year."}]
        out = verify_numeric_grounding("Revenue was 1200000000.", chunks)
        assert out["fully_grounded"] is True

    def test_small_integers_are_ignored(self):
        """Years and quarter numbers are not financial claims."""
        chunks = [{"text": "Revenue was $500 million."}]
        out = verify_numeric_grounding("In Q3 the top 3 segments grew; revenue was $500 million.", chunks)
        assert out["fully_grounded"] is True

    def test_normalizer_handles_units(self):
        assert _normalize_number("$1.2 billion") == _normalize_number("1200000000")
        assert _normalize_number("16.2%") == "16.2%"
        assert _normalize_number("not a number") is None


class TestCitationValidation:
    def test_detects_a_fabricated_citation(self):
        retrieved = [{"chunk_id": "abc::1"}, {"chunk_id": "abc::2"}]
        out = validate_citations("Revenue rose [abc::1] and margin fell [zzz::9].", retrieved)
        assert out["all_citations_valid"] is False
        assert out["fabricated_citations"] == ["zzz::9"]

    def test_accepts_valid_citations(self):
        retrieved = [{"chunk_id": "abc::1"}]
        out = validate_citations("Revenue rose [abc::1].", retrieved)
        assert out["all_citations_valid"] is True

    def test_flags_an_uncited_answer(self):
        out = validate_citations("Revenue rose sharply.", [{"chunk_id": "abc::1"}])
        assert out["uncited_answer"] is True


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class TestRetrieval:
    def test_finds_the_relevant_passage(self, db):
        out = retrieve(db, "What were total net sales?", tickers=["TEST"], use_reranker=False)
        assert out["status"] == "ok"
        assert any("94,036" in h["text"] for h in out["hits"])

    def test_metadata_filter_is_hard_not_soft(self, db):
        """A wrong-ticker query must return nothing, never a near miss.

        Silently widening the filter would answer about a different company.
        """
        assert retrieve(db, "total net sales", tickers=["NOPE"])["status"] == "no_documents"

    def test_doc_type_filter_applies(self, db):
        assert retrieve(db, "net sales", tickers=["TEST"], doc_types=["10-K"])["status"] == "no_documents"
        assert retrieve(db, "net sales", tickers=["TEST"], doc_types=["10-Q"])["status"] == "ok"

    def test_agent_refuses_to_answer_without_evidence(self, db):
        out = run_fundamental_agent(db, "What were sales?", tickers=["NOPE"])
        assert out["status"] == "no_evidence"
        assert out["answer"] is None

    def test_extractive_mode_returns_evidence_not_prose(self, db):
        """With no LLM key the agent must show sources, not synthesize."""
        out = run_fundamental_agent(db, "What were total net sales?", tickers=["TEST"])
        assert out["generation"]["mode"] == "extractive_evidence"
        assert out["generation"]["answer"] is None
        assert out["sources"]

    def test_ingestion_is_idempotent(self, db):
        result = ingest_document(db, text=FILING, title="dup", ticker="TEST", doc_type="10-Q")
        assert result["status"] == "already_ingested"


class TestChunking:
    def test_tokenizer_preserves_financial_tokens(self):
        """Stripping $ and . destroys exactly the tokens finance queries match on."""
        toks = tokenize("Revenue of $1.2 billion, up 10.5%")
        assert "$1.2" in toks
        assert "10.5%" in toks

    def test_bm25_ranks_the_matching_document_first(self):
        corpus = [tokenize(t) for t in [
            "revenue grew strongly this quarter",
            "the board approved a dividend",
            "cybersecurity risk factors and controls",
        ]]
        scores = BM25(corpus).score(tokenize("revenue quarter"))
        assert scores.argmax() == 0

    def test_section_heading_detected_at_start(self):
        assert detect_section("Item 1A. Risk Factors", None) == "Item 1A - Risk Factors"

    def test_table_of_contents_does_not_set_the_section(self):
        """Regression: a TOC listing every item name once mislabelled a whole
        filing, because the first chunk matched and the label then stuck."""
        toc = "See the following. " * 20 + "Item 1A. Risk Factors appears on page 14."
        assert detect_section(toc, None) is None

    def test_chunks_carry_ids_and_overlap(self):
        chunks = chunk_text("A paragraph.\n\n" * 60, max_chars=300, overlap=50, doc_id="doc")
        assert len(chunks) > 1
        assert all(c["chunk_id"].startswith("doc::") for c in chunks)


# ---------------------------------------------------------------------------
# Security primitives
# ---------------------------------------------------------------------------
class TestPasswordHashing:
    def test_round_trip(self):
        h = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", h)
        assert not verify_password("wrong password", h)

    def test_hash_is_salted(self):
        assert hash_password("same") != hash_password("same")

    def test_rejects_over_72_bytes(self):
        """bcrypt truncates silently past 72 bytes, which would let two
        different long passwords authenticate each other."""
        with pytest.raises(ValueError):
            hash_password("x" * 73)


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(monkeypatch_module):
    """API client with pipeline EXECUTION stubbed out.

    These tests exercise the HTTP contract -- status codes, ownership, the
    approval gate -- not the research pipeline. Without this stub, every
    submission test would kick off a real multi-minute run that fetches live
    market data and trains four models, which makes the suite unusable in CI
    and tests nothing the pipeline's own tests do not already cover.
    """
    from app.services import run_service

    monkeypatch_module.setattr(run_service, "submit_run", lambda *a, **k: None)
    monkeypatch_module.setattr("app.api.runs.run_service.submit_run", lambda *a, **k: None)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest's built-in fixture is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def auth(client):
    r = client.post(
        "/api/auth/signup",
        json={"email": "tester@example.com", "password": "a-strong-password", "display_name": "T"},
    )
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestApi:
    def test_health_reports_probed_capabilities(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert isinstance(body["capabilities"], dict)
        assert "fidelity" in body

    def test_health_warns_about_default_jwt_secret(self, client):
        body = client.get("/api/health").json()
        if body["capabilities"]["jwt_secret_is_default"]:
            assert body["warnings"]

    def test_duplicate_signup_is_rejected(self, client, auth):
        r = client.post("/api/auth/signup", json={"email": "tester@example.com", "password": "another-one"})
        assert r.status_code == 409

    def test_wrong_password_is_401(self, client, auth):
        r = client.post("/api/auth/login", json={"email": "tester@example.com", "password": "nope-nope"})
        assert r.status_code == 401

    def test_short_password_is_rejected(self, client):
        r = client.post("/api/auth/signup", json={"email": "x@y.com", "password": "short"})
        assert r.status_code == 422

    def test_garbage_token_is_401(self, client):
        assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401

    def test_approval_requires_authentication(self, client):
        """The gate is the point: an unattributable approval is not a control."""
        r = client.post("/api/runs/some-id/approve", json={"decision": "approved"})
        assert r.status_code == 401

    def test_run_submission_returns_202_not_200(self, client, auth):
        """202 because the work is accepted, not finished. Blocking the request
        for a multi-minute run was the original design's worst bug."""
        r = client.post(
            "/api/runs",
            json={"question": "Do momentum factors work in large caps?", "use_llm_planner": False},
            headers=auth,
        )
        assert r.status_code == 202
        assert r.json()["status"] == "queued"

    def test_run_is_not_visible_to_another_user(self, client, auth):
        """Ownership is enforced server-side, and a foreign run 404s rather than
        403s so the API does not confirm what the caller cannot see."""
        created = client.post(
            "/api/runs", json={"question": "A private research question here", "use_llm_planner": False},
            headers=auth,
        ).json()
        other = client.post(
            "/api/auth/signup", json={"email": "other@example.com", "password": "a-strong-password"}
        ).json()
        other_auth = {"Authorization": f"Bearer {other['access_token']}"}
        assert client.get(f"/api/runs/{created['run_id']}", headers=other_auth).status_code == 404

    def test_unknown_run_is_404(self, client, auth):
        assert client.get("/api/runs/does-not-exist", headers=auth).status_code == 404

    def test_presets_are_exposed(self, client):
        body = client.get("/api/config/presets").json()
        assert "large_cap_diversified" in body["universe_presets"]
        assert "momentum" in body["factor_families"]

    def test_document_search_validates_input(self, client):
        assert client.post("/api/documents/search", json={"query": "x"}).status_code == 422

    def test_corpus_stats_shape(self, client):
        body = client.get("/api/documents/stats").json()
        assert "n_documents" in body and "embedding_coverage" in body

    def test_adopting_weights_requires_an_approved_run(self, client, auth):
        """Research output must not become a position without human approval."""
        created = client.post(
            "/api/runs", json={"question": "Another research question for the book", "use_llm_planner": False},
            headers=auth,
        ).json()
        r = client.post(
            "/api/portfolio/books/adopt",
            json={"run_id": created["run_id"], "method": "risk_parity", "book_name": "B"},
            headers=auth,
        )
        assert r.status_code == 409
        assert "approved" in r.json()["detail"]
