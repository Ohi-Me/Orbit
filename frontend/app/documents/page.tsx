"use client";

import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  askDocuments,
  corpusStats,
  deleteDocument,
  ingestEdgar,
  listDocuments,
  searchDocuments,
} from "../lib/api";
import type { DocumentRow, RetrievalHit } from "../lib/types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Note,
  Spinner,
  Stat,
  Table,
  Td,
  int,
  num,
  shortDate,
} from "../components/ui";

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [ticker, setTicker] = useState("AAPL");
  const [ingesting, setIngesting] = useState(false);

  const [query, setQuery] = useState("What were total net sales for the quarter?");
  const [filterTickers, setFilterTickers] = useState("");
  const [hits, setHits] = useState<RetrievalHit[] | null>(null);
  const [answer, setAnswer] = useState<any>(null);
  const [searching, setSearching] = useState(false);
  const [retrievalMode, setRetrievalMode] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [d, s] = await Promise.all([listDocuments(), corpusStats()]);
      setDocs(d.documents);
      setStats(s);
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const doIngest = async () => {
    setIngesting(true);
    setError(null);
    try {
      await ingestEdgar(ticker.trim().toUpperCase(), 2);
      await refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setIngesting(false);
    }
  };

  const runSearch = async (withAnswer: boolean) => {
    setSearching(true);
    setError(null);
    setAnswer(null);
    setHits(null);
    const tickers = filterTickers
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
    try {
      if (withAnswer) {
        const a = await askDocuments({ query, tickers: tickers.length ? tickers : undefined, top_k: 6 });
        setAnswer(a);
        setHits(
          (a.sources ?? []).map((s: any) => ({
            chunk_id: s.chunk_id,
            text: s.excerpt,
            section: s.section,
            ticker: s.ticker,
            doc_type: s.doc_type,
            filing_date: s.filing_date,
            source_url: s.source_url,
            scores: { score: s.relevance, dense_score: null, lexical_score: null, rerank_score: null, fusion_score: 0 },
            retrieval_stage: a.retrieval_mode,
          }))
        );
        setRetrievalMode(a.retrieval_mode ?? "");
      } else {
        const r = await searchDocuments({ query, tickers: tickers.length ? tickers : undefined, top_k: 8 });
        setHits(r.hits);
        setRetrievalMode(r.retrieval_mode);
        if (r.status === "no_documents") setError("No documents match those filters.");
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <Card
          title="Search the filing corpus"
          subtitle="Metadata filter first, then hybrid dense + BM25 retrieval, then cross-encoder reranking"
        >
          <div className="space-y-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full border border-grid bg-white px-3 py-2 text-sm outline-none focus:border-navy"
              placeholder="Ask about revenue, risk factors, margins…"
            />
            <div className="flex flex-wrap gap-2">
              <input
                value={filterTickers}
                onChange={(e) => setFilterTickers(e.target.value)}
                className="flex-1 border border-grid bg-white px-3 py-1.5 text-xs outline-none focus:border-navy"
                placeholder="Filter by tickers, comma separated (e.g. AAPL, MSFT)"
              />
              <Button variant="ghost" onClick={() => runSearch(false)} disabled={searching}>
                Retrieve
              </Button>
              <Button onClick={() => runSearch(true)} disabled={searching}>
                Ask
              </Button>
            </div>
          </div>

          {searching && (
            <div className="mt-3">
              <Spinner label="Retrieving…" />
            </div>
          )}
          {error && (
            <div className="mt-3">
              <ErrorNote>{error}</ErrorNote>
            </div>
          )}

          {answer && (
            <div className="mt-4 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  tone={
                    answer.verification?.verdict === "VERIFIED"
                      ? "ok"
                      : answer.verification?.verdict === "evidence_only"
                        ? "info"
                        : "fail"
                  }
                >
                  {answer.verification?.verdict?.replace(/_/g, " ")}
                </Badge>
                <span className="text-[11px] text-muted">{answer.generation?.mode}</span>
              </div>

              {answer.generation?.answer ? (
                <div className="border border-grid bg-white px-3 py-2 text-sm leading-relaxed">
                  {answer.generation.answer}
                </div>
              ) : (
                <Note>{answer.generation?.note}</Note>
              )}

              {answer.verification?.numeric_grounding && (
                <div
                  className={`px-3 py-2 text-xs ${
                    answer.verification.numeric_grounding.fully_grounded
                      ? "border border-gain/40 bg-gain/5 text-gain"
                      : "border border-loss/40 bg-loss/5 text-loss"
                  }`}
                >
                  <strong>Numeric grounding:</strong>{" "}
                  {answer.verification.numeric_grounding.n_grounded}/
                  {answer.verification.numeric_grounding.n_numbers_in_answer} figures traced
                  verbatim to source text.
                  {answer.verification.numeric_grounding.ungrounded_values?.length > 0 && (
                    <div className="mt-1">
                      Ungrounded:{" "}
                      <span className="font-mono">
                        {answer.verification.numeric_grounding.ungrounded_values.join(", ")}
                      </span>{" "}
                      — these appear in the answer but not in the retrieved evidence.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {hits && hits.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-[11px] uppercase tracking-wider text-muted">
                {hits.length} passages · {retrievalMode}
              </p>
              {hits.map((h) => (
                <div key={h.chunk_id} className="border border-grid bg-white/70 px-3 py-2">
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
                    <span className="font-mono">{h.chunk_id}</span>
                    {h.ticker && <Badge tone="info">{h.ticker}</Badge>}
                    <span>{h.doc_type}</span>
                    <span>{h.filing_date}</span>
                    {h.section && <span className="italic">{h.section}</span>}
                    <span className="ml-auto font-mono">score {num(h.scores.score, 3)}</span>
                  </div>
                  <p className="mt-1.5 text-xs leading-relaxed">{h.text.slice(0, 480)}…</p>
                  {(h.scores.dense_score !== null || h.scores.lexical_score !== null) && (
                    <p className="mt-1 font-mono text-[10px] text-muted">
                      dense {num(h.scores.dense_score, 3)} · bm25 {num(h.scores.lexical_score, 2)} ·
                      rerank {num(h.scores.rerank_score, 3)}
                    </p>
                  )}
                  {h.source_url && (
                    <a
                      href={h.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-1 inline-block text-[10px] text-navy underline"
                    >
                      source filing
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}

          {hits && hits.length === 0 && <Empty>No passages matched.</Empty>}
        </Card>

        <div className="space-y-6">
          <Card title="Ingest filings" subtitle="Pulls real 10-K and 10-Q documents from SEC EDGAR">
            <div className="flex gap-2">
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                className="flex-1 border border-grid bg-white px-3 py-1.5 text-sm uppercase outline-none focus:border-navy"
                placeholder="Ticker"
              />
              <Button onClick={doIngest} disabled={ingesting || !ticker.trim()}>
                {ingesting ? "Ingesting…" : "Ingest"}
              </Button>
            </div>
            <div className="mt-3">
              <Note>
                Filings are chunked section-aware, embedded, and stored with their ticker, form
                type and filing date. Those metadata are a hard pre-filter at query time — a
                semantically similar passage from the wrong company or period is a wrong answer,
                not a near miss.
              </Note>
            </div>
          </Card>

          {stats && (
            <Card title="Corpus">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Documents" value={int(stats.n_documents)} />
                <Stat label="Chunks" value={int(stats.n_chunks)} />
                <Stat label="Embedded" value={int(stats.n_chunks_embedded)} />
                <Stat
                  label="Coverage"
                  value={`${(stats.embedding_coverage * 100).toFixed(0)}%`}
                  tone={stats.embedding_coverage > 0.95 ? "gain" : "warn"}
                />
              </div>
              {stats.embedding_models?.length > 0 && (
                <p className="mt-2 font-mono text-[10px] text-muted">
                  {stats.embedding_models.join(", ")}
                </p>
              )}
              {stats.embedding_models?.length > 1 && (
                <div className="mt-2">
                  <ErrorNote>
                    More than one embedding model is present. Vectors from different models are not
                    comparable — re-embed the corpus.
                  </ErrorNote>
                </div>
              )}
              {stats.by_ticker && Object.keys(stats.by_ticker).length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1">
                  {Object.entries(stats.by_ticker).map(([t, n]: [string, any]) => (
                    <span key={t} className="border border-grid px-1.5 py-0.5 text-[10px]">
                      {t ?? "—"} <span className="font-mono text-muted">{n}</span>
                    </span>
                  ))}
                </div>
              )}
            </Card>
          )}
        </div>
      </div>

      <Card title="Document library">
        {loading ? (
          <Spinner label="Loading…" />
        ) : docs.length === 0 ? (
          <Empty>No documents ingested yet. Pull a company&rsquo;s filings above.</Empty>
        ) : (
          <Table head={["Ticker", "Form", "Title", "Filed", "Chars", "Chunks", "Source", ""]}>
            {docs.map((d) => (
              <tr key={d.id} className="border-b border-grid/60">
                <Td mono>{d.ticker ?? "—"}</Td>
                <Td>
                  <Badge tone="info">{d.doc_type}</Badge>
                </Td>
                <Td>
                  <span className="text-xs">{d.title}</span>
                  {d.company_name && (
                    <div className="text-[10px] text-muted">{d.company_name}</div>
                  )}
                </Td>
                <Td mono>{d.filing_date ?? "—"}</Td>
                <Td mono>{int(d.char_count)}</Td>
                <Td mono>{d.n_chunks}</Td>
                <Td mono className="text-[10px]">
                  {d.source}
                </Td>
                <Td>
                  <button
                    onClick={async () => {
                      await deleteDocument(d.id);
                      refresh();
                    }}
                    className="text-[10px] uppercase tracking-wider text-muted hover:text-loss"
                  >
                    delete
                  </button>
                </Td>
              </tr>
            ))}
          </Table>
        )}
      </Card>
    </div>
  );
}
