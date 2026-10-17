"""
Signal score: combines price history (computed in code) with news sentiment (scored by Gemma)
into one transparent 0-100 number.

    50 = neutral. Above 60 = "leans favorable". Below 40 = "leans unfavorable".
    score = 50 + 25 * history_score + 25 * sentiment_score     (each sub-score is -1 .. +1)
    If no headlines are available: score = 50 + 50 * history_score, and the UI says so.

This is a descriptive summary of recent data, NOT a prediction or advice. Say so in the demo.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List

import numpy as np
import yfinance as yf
from pydantic import BaseModel, Field

import backtest as bt
import gemma_core as gc

CACHE_DIR = "cache"


def _clip(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


# ---------------------------------------------------------------------
# 1. History score (pure code, no AI)
# ---------------------------------------------------------------------
def history_score(ticker: str) -> dict:
    df = bt.load_prices(ticker, 1)            # ~2 years of daily prices (cached)
    close = df["Close"]
    if len(close) < 210:
        raise ValueError(f"Not enough history for {ticker}")
    price = close.iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    ret_6m = price / close.iloc[-127] - 1
    last_year = close.iloc[-252:]
    drawdown = price / last_year.max() - 1
    vol = close.pct_change().iloc[-252:].std() * np.sqrt(252)

    parts = [
        {"name": "Trend", "detail": f"price is {(price / ma200 - 1) * 100:+.1f}% vs its 200-day average",
         "score": _clip((price / ma200 - 1) / 0.10)},
        {"name": "6-month momentum", "detail": f"{ret_6m * 100:+.1f}% over 6 months",
         "score": _clip(ret_6m / 0.25)},
        {"name": "Distance from high", "detail": f"{drawdown * 100:.1f}% below its 52-week high",
         "score": _clip(1 + drawdown / 0.10)},
        {"name": "Volatility", "detail": f"{vol * 100:.0f}% annualized swings",
         "score": _clip((0.30 - vol) / 0.20)},
    ]
    for p in parts:
        p["score"] = round(p["score"], 2)
    return {"score": round(float(np.mean([p["score"] for p in parts])), 2), "parts": parts,
            "price": round(float(price), 2), "close_1y": last_year}


# ---------------------------------------------------------------------
# 2. Headlines (yfinance news; cached so the demo survives bad wifi)
# ---------------------------------------------------------------------
def _parse_item(item: dict) -> dict | None:
    c = item.get("content", item)  # newer yfinance nests fields under "content"
    title = c.get("title")
    if not title:
        return None
    pub = c.get("provider", {}).get("displayName") if isinstance(c.get("provider"), dict) else None
    pub = pub or c.get("publisher") or ""
    date = c.get("pubDate") or ""
    if not date and c.get("providerPublishTime"):
        date = datetime.fromtimestamp(c["providerPublishTime"], tz=timezone.utc).isoformat()
    related = c.get("relatedTickers") or item.get("relatedTickers") or []
    return {"title": title.strip(), "publisher": pub, "date": str(date)[:10],
            "_related": [str(t).upper() for t in related]}


def _from_yf_ticker(ticker: str) -> list[dict]:
    return [h for h in (_parse_item(i) for i in (yf.Ticker(ticker).news or [])) if h]


def _from_yf_search(ticker: str) -> list[dict]:
    res = yf.Search(ticker, news_count=10)
    return [h for h in (_parse_item(i) for i in (getattr(res, "news", None) or [])) if h]


def _parse_rss(xml_text: str) -> list[dict]:
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    out = []
    for item in ET.fromstring(xml_text).iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        src = (item.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):          # Google News appends " - Publisher"
            title = title[: -len(" - " + src)]
        try:
            date = parsedate_to_datetime(item.findtext("pubDate")).date().isoformat()
        except Exception:
            date = ""
        out.append({"title": title, "publisher": src, "date": date})
    return out


def _from_google_news(ticker: str) -> list[dict]:
    import urllib.parse
    import urllib.request
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(f"{ticker} stock")
           + "&hl=en-US&gl=US&ceid=US:en")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return _parse_rss(r.read().decode("utf-8", "ignore"))


NEWS_SOURCES = [("Yahoo Finance", _from_yf_ticker), ("Yahoo search", _from_yf_search),
                ("Google News", _from_google_news)]
LAST_NEWS_SOURCE = None
MIN_RELEVANT = 3


def _company_words(ticker: str) -> list[str]:
    """Words that identify the company in a headline, e.g. NVDA -> ['nvda', 'nvidia']."""
    words = [ticker.lower()]
    try:
        with open(os.path.join(CACHE_DIR, f"{ticker}.json")) as f:
            name = json.load(f).get("name", "")
        first = name.replace(",", " ").split()[0].lower() if name else ""
        if len(first) >= 3 and first not in ("the", "inc", "corp"):
            words.append(first.rstrip("."))
    except Exception:
        pass
    return words


def _relevant(items: list[dict], ticker: str) -> list[dict]:
    words = _company_words(ticker)
    keep = []
    for h in items:
        title = h["title"].lower()
        if any(w in title.split() or w in title for w in words):
            keep.append(h)
    return keep


def get_headlines(ticker: str, n: int = 10) -> list[dict]:
    """Try each source in order; first one with headlines wins. Falls back to the saved copy."""
    global LAST_NEWS_SOURCE
    ticker = ticker.upper()
    path = os.path.join(CACHE_DIR, f"{ticker}_news.json")
    for name, fetch in NEWS_SOURCES:
        try:
            items = fetch(ticker)
        except Exception:
            continue
        if name != "Google News":            # Google News is already a targeted search
            items = _relevant(items, ticker)  # drop headlines about other companies
        items = sorted(items, key=lambda h: h.get("date") or "", reverse=True)[:n]   # newest first
        for h in items:
            h.pop("_related", None)
        if len(items) >= MIN_RELEVANT:
            LAST_NEWS_SOURCE = name
            for h in items:
                h.setdefault("source", name)
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(items, f)
            except Exception:
                pass
            return items
    if os.path.exists(path):
        LAST_NEWS_SOURCE = "saved copy"
        with open(path) as f:
            return json.load(f)
    LAST_NEWS_SOURCE = None
    return []


# ---------------------------------------------------------------------
# 3. Sentiment (Gemma scores each headline -1 .. +1)
# ---------------------------------------------------------------------
class HeadlineScore(BaseModel):
    i: int
    score: float = Field(ge=-1, le=1)


class SentimentReport(BaseModel):
    scores: List[HeadlineScore]
    summary: str


SENTIMENT_PROMPT = """You score news headlines for how they would affect a stock's shareholders.
For EACH numbered headline return its number `i` and a `score` from -1 to 1:
-1 = clearly bad news for the stock, 0 = neutral or unrelated, +1 = clearly good news.
Judge only the headline text. Most headlines are mildly positive or negative (between -0.5 and 0.5);
use the extremes only for major news. summary: one sentence on the overall tone.
Return only JSON."""


def sentiment_score(ticker: str, headlines: list[dict]) -> dict:
    if not headlines:
        return {"score": None, "n": 0, "summary": "No recent headlines available.", "headlines": []}
    numbered = "\n".join(f"{k}. {h['title']}" for k, h in enumerate(headlines))
    rep = gc.call_gemma(SENTIMENT_PROMPT, f"STOCK: {ticker}\nHEADLINES:\n{numbered}", SentimentReport)
    by_i = {s.i: s.score for s in rep.scores}
    scored = [{**h, "score": round(by_i[k], 2)} for k, h in enumerate(headlines) if k in by_i]
    if not scored:
        return {"score": None, "n": 0, "summary": "Gemma couldn't score the headlines.", "headlines": []}
    return {"score": round(float(np.mean([h["score"] for h in scored])), 2), "n": len(scored),
            "summary": rep.summary, "headlines": scored}


# ---------------------------------------------------------------------
# 4. Combined signal
# ---------------------------------------------------------------------
def signal(ticker: str) -> dict:
    ticker = ticker.upper()
    hist = history_score(ticker)
    sent = sentiment_score(ticker, get_headlines(ticker))
    if sent["score"] is None:
        score = 50 + 50 * hist["score"]
    else:
        score = 50 + 25 * hist["score"] + 25 * sent["score"]
    score = int(round(score))
    lean = "Leans favorable" if score >= 60 else "Leans unfavorable" if score <= 40 else "Neutral / mixed"
    return {"ticker": ticker, "score": score, "lean": lean, "history": hist, "sentiment": sent}