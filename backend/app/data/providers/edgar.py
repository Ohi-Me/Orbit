"""
SEC EDGAR provider -- point-in-time fundamentals and filing text.

THE POINT-IN-TIME PROBLEM (the reason this file is careful)
-----------------------------------------------------------
Naive fundamental research is wrong in a way that is invisible in a backtest
and fatal in production: you look up a company's 2021 revenue *today* and
attach it to a 2021 date. But that number may have been restated in 2023,
and the original figure -- the only one an investor could actually have
acted on -- was different. Worse, the 2021 annual figure was not public
until the 10-K was filed in early 2022, so using it on any 2021 date is
straightforward look-ahead.

EDGAR's XBRL "company facts" API is one of the few free sources that solves
this properly, because every fact carries both:
    end   -- the period the number describes
    filed -- the date it actually became public
`build_pit_fundamentals` keys strictly on `filed`, so a factor computed for
date t only ever sees numbers filed on or before t, and keeps the FIRST
reported value for a period rather than the latest restatement.

Rate limits: SEC asks for <= 10 requests/second and a self-identifying
User-Agent. Both are honored here; a generic UA gets blocked.
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime, timezone
from functools import lru_cache

import httpx
import pandas as pd

from app.core.config import get_settings

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

_MIN_REQUEST_INTERVAL = 0.12  # ~8 req/s, comfortably under SEC's 10/s ceiling
_last_request_at = 0.0


class EdgarError(RuntimeError):
    pass


def _headers() -> dict:
    return {
        "User-Agent": get_settings().sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _get(url: str, timeout: float = 25.0) -> httpx.Response:
    """Rate-limited GET against SEC hosts."""
    global _last_request_at
    elapsed = time.time() - _last_request_at
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    resp = httpx.get(url, headers=_headers(), timeout=timeout, follow_redirects=True)
    _last_request_at = time.time()
    if resp.status_code != 200:
        raise EdgarError(f"SEC returned HTTP {resp.status_code} for {url}")
    return resp


# ---------------------------------------------------------------------------
# Ticker -> CIK
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, dict]:
    data = _get(SEC_TICKERS_URL).json()
    out = {}
    for row in data.values():
        out[str(row["ticker"]).upper()] = {
            "cik": str(row["cik_str"]).zfill(10),
            "cik_int": int(row["cik_str"]),
            "name": row["title"],
        }
    return out


def resolve_cik(ticker: str) -> dict | None:
    try:
        return _ticker_map().get(ticker.upper())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Point-in-time fundamentals
# ---------------------------------------------------------------------------
# us-gaap tags we care about, mapped to platform-level metric names. Several
# tags can map to one metric because filers are inconsistent; the first tag
# that yields data wins.
FUNDAMENTAL_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}


def fetch_company_facts(ticker: str) -> dict:
    meta = resolve_cik(ticker)
    if not meta:
        raise EdgarError(f"No CIK found for ticker {ticker}")
    return _get(SEC_FACTS_URL.format(cik=meta["cik"])).json()


def _extract_facts(facts_json: dict, tags: list[str]) -> list[dict]:
    """Pull every reported observation for the first tag that has data.

    Each record keeps `filed` (public availability) separately from `end`
    (the period described) -- the distinction the whole module exists for.
    """
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node = us_gaap.get(tag)
        if not node:
            continue
        records = []
        for unit_key, entries in node.get("units", {}).items():
            for e in entries:
                if not e.get("filed") or not e.get("end") or e.get("val") is None:
                    continue
                records.append(
                    {
                        "tag": tag,
                        "unit": unit_key,
                        "end": e["end"],
                        "start": e.get("start"),
                        "filed": e["filed"],
                        "value": float(e["val"]),
                        "form": e.get("form", ""),
                        "fiscal_year": e.get("fy"),
                        "fiscal_period": e.get("fp"),
                        "accession": e.get("accn"),
                    }
                )
        if records:
            return records
    return []


def build_pit_fundamentals(ticker: str) -> tuple[pd.DataFrame, dict]:
    """Return a point-in-time-safe long frame of fundamental observations.

    Columns: metric, value, period_end, filed_date, form, fiscal_year, fiscal_period.

    Two deliberate rules:
      * Only annual/quarterly report forms (10-K, 10-Q and their variants) are
        kept -- 8-K and S-1 restatements otherwise inject duplicate periods.
      * For a duplicated (metric, period_end), the row with the EARLIEST
        `filed` wins. That is the number that was actually knowable first;
        later restatements of the same period are dropped rather than
        overwriting history.
    """
    facts = fetch_company_facts(ticker)
    rows: list[dict] = []
    for metric, tags in FUNDAMENTAL_TAGS.items():
        for rec in _extract_facts(facts, tags):
            if not rec["form"].startswith(("10-K", "10-Q")):
                continue
            rows.append({"metric": metric, **rec})

    if not rows:
        return pd.DataFrame(
            columns=["metric", "value", "period_end", "filed_date", "form", "fiscal_year", "fiscal_period"]
        ), {"ticker": ticker, "n_facts": 0, "note": "No usable us-gaap facts returned."}

    df = pd.DataFrame(rows)
    df = df.rename(columns={"end": "period_end", "filed": "filed_date"})
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filed_date"] = pd.to_datetime(df["filed_date"])

    df = (
        df.sort_values("filed_date")
        .drop_duplicates(subset=["metric", "period_end"], keep="first")
        .sort_values(["metric", "period_end"])
        .reset_index(drop=True)
    )

    provenance = {
        "ticker": ticker,
        "company_name": (resolve_cik(ticker) or {}).get("name"),
        "n_facts": int(len(df)),
        "metrics": sorted(df["metric"].unique().tolist()),
        "earliest_period": str(df["period_end"].min().date()),
        "latest_period": str(df["period_end"].max().date()),
        "latest_filed": str(df["filed_date"].max().date()),
        "point_in_time": True,
        "restatement_policy": "first-reported value kept per (metric, period)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return df, provenance


def as_of(pit_frame: pd.DataFrame, metric: str, date) -> float | None:
    """Latest value of `metric` that was publicly filed on or before `date`.

    This is the only sanctioned way to read a fundamental inside a factor.
    """
    if pit_frame.empty:
        return None
    ts = pd.Timestamp(date)
    sub = pit_frame[(pit_frame["metric"] == metric) & (pit_frame["filed_date"] <= ts)]
    if sub.empty:
        return None
    return float(sub.sort_values("period_end").iloc[-1]["value"])


def pit_series(pit_frame: pd.DataFrame, metric: str, dates) -> pd.Series:
    """Vectorized `as_of` across a date index -- a forward-filled step function
    that only ever steps on a filing date."""
    idx = pd.DatetimeIndex(dates)
    if pit_frame.empty:
        return pd.Series([None] * len(idx), index=idx, dtype="float64")
    sub = pit_frame[pit_frame["metric"] == metric].sort_values("filed_date")
    if sub.empty:
        return pd.Series([None] * len(idx), index=idx, dtype="float64")
    # For each filing date, the then-current value (latest period filed so far).
    latest = sub.sort_values(["filed_date", "period_end"]).drop_duplicates("filed_date", keep="last")
    step = pd.Series(latest["value"].values, index=pd.DatetimeIndex(latest["filed_date"].values))
    step = step[~step.index.duplicated(keep="last")].sort_index()
    return step.reindex(step.index.union(idx)).ffill().reindex(idx)


# ---------------------------------------------------------------------------
# Filing text (for the RAG corpus)
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")
_NL_RE = re.compile(r"\n{3,}")


def _html_to_text(raw: str) -> str:
    txt = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    txt = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", txt)
    txt = _TAG_RE.sub(" ", txt)
    txt = html.unescape(txt)
    txt = _WS_RE.sub(" ", txt)
    txt = _NL_RE.sub("\n\n", txt)
    return "\n".join(line.strip() for line in txt.splitlines()).strip()


def list_filings(ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 4) -> list[dict]:
    meta = resolve_cik(ticker)
    if not meta:
        raise EdgarError(f"No CIK found for ticker {ticker}")
    data = _get(SEC_SUBMISSIONS_URL.format(cik=meta["cik"])).json()
    recent = data.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(recent.get("form", [])):
        if not form.startswith(forms):
            continue
        accession = recent["accessionNumber"][i].replace("-", "")
        out.append(
            {
                "ticker": ticker.upper(),
                "company_name": meta["name"],
                "form": form,
                "filing_date": recent["filingDate"][i],
                "report_date": recent.get("reportDate", [None] * (i + 1))[i],
                "accession": recent["accessionNumber"][i],
                "url": SEC_ARCHIVE_URL.format(
                    cik_int=meta["cik_int"], accession=accession, document=recent["primaryDocument"][i]
                ),
            }
        )
        if len(out) >= limit:
            break
    return out


def fetch_filing_text(filing: dict, max_chars: int = 400_000) -> str:
    """Download a filing's primary document and strip it to plain text.

    Truncated at max_chars: a full 10-K can exceed 2 MB of HTML, most of
    which is exhibits and tables that add retrieval noise rather than signal.
    """
    raw = _get(filing["url"], timeout=60.0).text
    text = _html_to_text(raw)
    return text[:max_chars]
