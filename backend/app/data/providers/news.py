"""
News provider -- dated headlines.

THE DATE IS THE POINT. The original build attached today's headlines to every
historical row of the factor panel, which turned sentiment into a constant
per ticker AND embedded present-day knowledge into every past date. Every
item returned here carries a `published_at` timestamp, and the sentiment
agent builds a dated series from it so a factor value on date t reflects only
news published on or before t.

COVERAGE IS HONEST. Free news endpoints return recent items only -- typically
days to weeks, not years. That means a genuine historical sentiment factor
cannot be built from this source, and the platform says so rather than
faking depth: `coverage_days` is reported, the factor's panel coverage is
measured, and the factor-selection gate drops it when coverage is too thin.
Pretending otherwise is exactly the failure this module exists to prevent.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


class NewsFetchError(RuntimeError):
    pass


def fetch_news(tickers: list[str], limit_per_ticker: int = 30) -> tuple[dict, dict]:
    """Return ({ticker: [items]}, provenance).

    Each item: {title, publisher, published_at (ISO), link, summary}.
    Falls back to a small bundled sample when no provider is reachable, and
    labels that fallback explicitly in provenance.
    """
    out: dict[str, list[dict]] = {}
    failures: dict[str, str] = {}
    used_live = False

    try:
        import yfinance as yf
    except ImportError:
        yf = None

    if yf is not None:
        for t in tickers:
            try:
                raw = yf.Ticker(t).news or []
                items = []
                for n in raw[:limit_per_ticker]:
                    content = n.get("content", n)
                    title = content.get("title") or n.get("title")
                    if not title:
                        continue
                    ts = (
                        n.get("providerPublishTime")
                        or content.get("pubDate")
                        or content.get("displayTime")
                    )
                    published = _parse_timestamp(ts)
                    if published is None:
                        continue
                    items.append(
                        {
                            "title": title,
                            "publisher": (content.get("provider") or {}).get("displayName")
                            if isinstance(content.get("provider"), dict)
                            else n.get("publisher", "unknown"),
                            "published_at": published.isoformat(),
                            "link": content.get("canonicalUrl", {}).get("url")
                            if isinstance(content.get("canonicalUrl"), dict)
                            else n.get("link"),
                            "summary": (content.get("summary") or "")[:600],
                        }
                    )
                if items:
                    out[t] = sorted(items, key=lambda x: x["published_at"])
                    used_live = True
                else:
                    failures[t] = "provider returned no dated items"
            except Exception as e:
                failures[t] = f"{type(e).__name__}: {str(e)[:100]}"

    if not out:
        out = _sample_news(tickers)

    all_dates = [pd.Timestamp(i["published_at"]) for items in out.values() for i in items]
    coverage_days = (
        int((max(all_dates) - min(all_dates)).days) if len(all_dates) > 1 else 0
    )

    provenance = {
        "provider": "yfinance_news" if used_live else "bundled_sample",
        "is_synthetic": not used_live,
        "n_tickers_with_news": len(out),
        "n_items": sum(len(v) for v in out.values()),
        "coverage_days": coverage_days,
        "earliest": min(all_dates).isoformat() if all_dates else None,
        "latest": max(all_dates).isoformat() if all_dates else None,
        "failures": failures,
        "limitation": "Free news endpoints return recent items only. A sentiment "
        "factor built from this cannot span a multi-year backtest; the factor "
        "coverage gate will drop it rather than broadcast recent sentiment across "
        "historical dates.",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return out, provenance


def _parse_timestamp(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    try:
        parsed = pd.Timestamp(ts)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        return parsed.to_pydatetime()
    except Exception:
        return None


def _sample_news(tickers: list[str]) -> dict:
    """Small dated sample used only when no provider is reachable.

    Dated relative to now so the downstream dating logic is exercised
    identically to the live path -- but flagged as synthetic in provenance.
    """
    now = pd.Timestamp.now(tz="UTC")
    templates = [
        ("{t} beats on revenue, raises full-year guidance on strong demand", -2),
        ("Analysts flag margin pressure at {t} from rising input costs", -6),
        ("{t} announces buyback; management signals confidence in backlog", -11),
        ("{t} misses consensus on softer volumes; shares slip", -19),
        ("{t} names new chief financial officer amid restructuring", -27),
    ]
    return {
        t: [
            {
                "title": tpl.format(t=t),
                "publisher": "bundled_sample",
                "published_at": (now + pd.Timedelta(days=off)).isoformat(),
                "link": None,
                "summary": "",
            }
            for tpl, off in templates
        ]
        for t in tickers
    }
