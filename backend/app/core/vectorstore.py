"""
Vector store -- embeddings, hybrid retrieval, and reranking.

ARCHITECTURE NOTE (why there is no Qdrant/Pinecone here)
--------------------------------------------------------
Embeddings live as raw float32 bytes on the `chunks` table and search is an
exact brute-force cosine scan in NumPy. For a corpus of hundreds to low
thousands of filing chunks this is not a compromise -- it is faster than an
ANN index (which must be built and kept in sync), it returns exact rather
than approximate neighbours, and it removes a fourth service from the
deployment. `search()` is the single swap point if the corpus outgrows it.

RETRIEVAL DESIGN FOR FINANCIAL DOCUMENTS
Three properties that generic RAG tutorials get wrong for filings:

  1. FILTER BEFORE YOU RANK. "What was Q3 revenue" means this company, this
     period. A semantically perfect sentence from the wrong issuer or the
     wrong fiscal year is a wrong answer, not a near miss, so metadata
     filters are applied as a hard pre-filter rather than as a soft signal.

  2. DENSE ALONE LOSES NUMBERS AND TICKERS. Embeddings are excellent at
     paraphrase and poor at exact tokens -- and financial queries are full of
     exact tokens ("10-K", "FY2024", "$1.2 billion", "EBITDA"). Retrieval is
     therefore hybrid: dense similarity fused with lexical BM25 scoring.

  3. RERANK BEFORE YOU CITE. First-stage retrieval optimizes recall. When a
     number is going to be quoted, precision at rank 1 is what matters, and a
     cross-encoder that reads query and passage together is markedly better
     at that than any bi-encoder similarity.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from app.core.config import get_settings

_EMBED_MODEL = None
_RERANK_MODEL = None


# ---------------------------------------------------------------------------
# Embedding model (lazy, cached, optional)
# ---------------------------------------------------------------------------
def get_embedder():
    """Load the sentence-transformer once. Returns None when unavailable."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    settings = get_settings()
    if not settings.allow_model_download:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(settings.embedding_model)
        return _EMBED_MODEL
    except Exception:
        return None


def get_reranker():
    global _RERANK_MODEL
    if _RERANK_MODEL is not None:
        return _RERANK_MODEL
    settings = get_settings()
    if not settings.allow_model_download:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _RERANK_MODEL = CrossEncoder(settings.reranker_model)
        return _RERANK_MODEL
    except Exception:
        return None


def embed_texts(texts: list[str]) -> tuple[np.ndarray | None, str | None]:
    """Embed a batch. Returns (matrix, model_name) or (None, None) if no model."""
    model = get_embedder()
    if model is None or not texts:
        return None, None
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32), get_settings().embedding_model


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# ---------------------------------------------------------------------------
# Lexical scoring (BM25)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9$%.\-]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that KEEP currency, percent, decimals and hyphens.

    A tokenizer that strips '$' and '.' turns "$1.2 billion" into "1 2 billion"
    and destroys exactly the tokens a financial query is trying to match.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Standard Okapi BM25 over a small in-memory corpus."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(self.n, 1)
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            tf_i, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for term in query_tokens:
                if term not in tf_i:
                    continue
                f = tf_i[term]
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                )
            scores[i] = s
        return scores


# ---------------------------------------------------------------------------
# Fusion and search
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Combine ranked lists by reciprocal rank.

    RRF rather than a weighted score sum because dense cosine and BM25 live on
    incomparable scales; normalizing them against each other requires tuning
    that would not transfer to a different corpus. RRF only uses rank order.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def search(
    query: str,
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray | None = None,
    top_k: int = 8,
    use_reranker: bool = True,
    rerank_candidates: int = 25,
) -> list[dict]:
    """Hybrid retrieval: dense + BM25, fused, then optionally reranked.

    Returns a list of {index, score, dense_score, lexical_score, rerank_score,
    stage} preserving how each result was found, so a retrieval failure can be
    diagnosed rather than merely observed.
    """
    if not chunk_texts:
        return []

    n = len(chunk_texts)
    rankings, dense_scores, lexical_scores = [], None, None

    # --- dense ---
    if chunk_embeddings is not None and len(chunk_embeddings) == n:
        q_vec, _ = embed_texts([query])
        if q_vec is not None:
            dense_scores = chunk_embeddings @ q_vec[0]
            rankings.append(list(np.argsort(-dense_scores)[: rerank_candidates * 2]))

    # --- lexical ---
    corpus_tokens = [tokenize(t) for t in chunk_texts]
    bm25 = BM25(corpus_tokens)
    lexical_scores = bm25.score(tokenize(query))
    if lexical_scores.max() > 0:
        rankings.append(list(np.argsort(-lexical_scores)[: rerank_candidates * 2]))

    if not rankings:
        return []

    fused = reciprocal_rank_fusion(rankings)
    candidates = sorted(fused.items(), key=lambda kv: -kv[1])[:rerank_candidates]
    candidate_idx = [i for i, _ in candidates]

    stage = "hybrid_fusion"
    rerank_scores = {}
    if use_reranker and len(candidate_idx) > 1:
        reranker = get_reranker()
        if reranker is not None:
            pairs = [(query, chunk_texts[i]) for i in candidate_idx]
            try:
                scores = reranker.predict(pairs)
                rerank_scores = {i: float(s) for i, s in zip(candidate_idx, scores)}
                candidate_idx = sorted(candidate_idx, key=lambda i: -rerank_scores[i])
                stage = "hybrid_fusion+cross_encoder_rerank"
            except Exception:
                pass

    results = []
    for i in candidate_idx[:top_k]:
        results.append(
            {
                "index": int(i),
                "score": round(float(rerank_scores.get(i, fused.get(i, 0.0))), 5),
                "dense_score": round(float(dense_scores[i]), 5) if dense_scores is not None else None,
                "lexical_score": round(float(lexical_scores[i]), 5) if lexical_scores is not None else None,
                "rerank_score": round(rerank_scores[i], 5) if i in rerank_scores else None,
                "fusion_score": round(float(fused.get(i, 0.0)), 5),
                "stage": stage,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Section headings that matter in a 10-K. Retrieval quality improves markedly
# when a chunk knows which item it came from, because analysts think in items.
SECTION_PATTERNS = [
    (re.compile(r"item\s+1a\.?\s*[-:.]?\s*risk\s+factors", re.I), "Item 1A - Risk Factors"),
    (re.compile(r"item\s+7a\.?\s*[-:.]?\s*quantitative\s+and\s+qualitative", re.I), "Item 7A - Market Risk"),
    (re.compile(r"item\s+7\.?\s*[-:.]?\s*management.s\s+discussion", re.I), "Item 7 - MD&A"),
    (re.compile(r"item\s+8\.?\s*[-:.]?\s*financial\s+statements", re.I), "Item 8 - Financial Statements"),
    (re.compile(r"item\s+3\.?\s*[-:.]?\s*legal\s+proceedings", re.I), "Item 3 - Legal Proceedings"),
    (re.compile(r"item\s+1\.?\s*[-:.]?\s*business", re.I), "Item 1 - Business"),
]


def detect_section(text: str, current: str | None) -> str | None:
    """Update the current section only on something that looks like a HEADING.

    A naive `pattern.search(text)` mislabels an entire filing: the table of
    contents lists every item name in one block, so the first chunk matches
    "Item 1A Risk Factors" and that label then sticks to the whole document.
    A real heading is short and starts the paragraph, so both are required.
    """
    stripped = text.strip()
    if len(stripped) > 200:
        head = stripped[:120]
    else:
        head = stripped

    for pattern, label in SECTION_PATTERNS:
        m = pattern.search(head)
        # Must appear at (or very near) the start of the block to be a heading
        # rather than a cross-reference inside running prose.
        if m and m.start() <= 8:
            return label
    return current


def chunk_text(
    text: str, max_chars: int = 1200, overlap: int = 200, doc_id: str = "doc"
) -> list[dict]:
    """Split into overlapping, section-aware chunks on paragraph boundaries.

    Larger chunks than a naive tutorial default (1200 vs 500 chars) because a
    financial claim and the number supporting it are frequently separated by a
    sentence or two; splitting between them produces a citation that does not
    contain the figure it is cited for.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[dict] = []
    buffer = ""
    section: str | None = None
    index = 0

    for para in paragraphs:
        section = detect_section(para, section)
        if buffer and len(buffer) + len(para) > max_chars:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::{index}",
                    "chunk_index": index,
                    "text": buffer.strip(),
                    "section": section,
                    "char_count": len(buffer.strip()),
                }
            )
            index += 1
            tail = buffer[-overlap:] if overlap else ""
            buffer = (tail + " " + para).strip()
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    if buffer.strip():
        chunks.append(
            {
                "chunk_id": f"{doc_id}::{index}",
                "chunk_index": index,
                "text": buffer.strip(),
                "section": section,
                "char_count": len(buffer.strip()),
            }
        )
    return chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:32]


@lru_cache(maxsize=1)
def embedding_dimension() -> int | None:
    model = get_embedder()
    return int(model.get_sentence_embedding_dimension()) if model is not None else None
