"""
Chat assistant: Gemma reads the user's message, decides which tool to run, and the app shows
real computed results. Gemma never invents numbers: every figure comes from code or data.

Intents:
    backtest -> test a trading rule            (gemma_core.parse_strategy + backtest.py)
    evaluate -> investing-style fit for a stock (gemma_core.evaluate_profiles)
    signal   -> 0-100 history + sentiment score (insights.signal)
    price    -> current price and 1-year chart  (insights.history_score)
    general  -> answer a finance concept question in plain words
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel

import gemma_core as gc
import insights


class Route(BaseModel):
    intent: Literal["backtest", "evaluate", "signal", "price", "general"]
    ticker: Optional[str] = None
    reply: str


ROUTER_PROMPT = """You are the assistant inside TestMyIdea, a stock research app for beginner investors.
Pick ONE intent for the user's latest message:
- backtest: they describe a trading rule to test ("buy X when it drops 3%", "what if I bought the dip on Y").
- signal: they ask whether a stock looks good, should they buy/sell, what the sentiment or outlook is.
- evaluate: they ask what kind of investor a stock suits, or about its valuation/dividend/growth style.
- price: they ask the price, how it has done, or to see a chart.
- general: a concept question (what is RSI, what is a P/E ratio, what is a dividend) or anything else.

ticker: uppercase symbol (Apple -> AAPL, Coca-Cola -> KO, Tesla -> TSLA, S&P 500 -> SPY).
If the message has no company but the conversation does ("what about its sentiment?"), reuse that ticker.
Use null if there is no stock at all.

reply:
- For general: answer in 2-4 plain sentences. Never state specific prices, returns, or numbers
  for a real stock; say the app can look them up instead.
- For every other intent: one short sentence saying what you'll do (e.g. "Let me test that on Apple.").
Never tell the user to buy or sell. Return only JSON."""


def route(message: str, history: list[dict]) -> Route:
    convo = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in history[-6:])
    text = (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "") + f"LATEST MESSAGE: {message}"
    r = gc.call_gemma(ROUTER_PROMPT, text, Route)
    if r.ticker:
        r.ticker = r.ticker.upper().strip().lstrip("$")
    if r.intent in ("evaluate", "signal", "price") and not r.ticker:
        r.intent = "general"
        r.reply = "Which stock do you mean? Give me a company name or ticker."
    return r


_cache: dict = {}


def _cached(key: tuple, fn, ttl: int = 1800):
    """Tiny time-based cache so repeat questions about the same stock are instant."""
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def handle(message: str, history: list[dict]) -> dict:
    """Returns a chat message dict: {role, text, kind, payload}. Never raises."""
    try:
        r = route(message, history)
    except Exception as e:
        return {"role": "assistant", "text": f"Gemma didn't respond ({e}). Try again in a moment.",
                "kind": "text", "payload": None}

    try:
        if r.intent == "backtest":
            s = gc.parse_strategy(message if not r.ticker or r.ticker in message.upper()
                                  else f"{message} (stock: {r.ticker})")
            if not s.supported:
                return {"role": "assistant", "kind": "text", "payload": None,
                        "text": f"I can't test that yet: {s.unsupported_reason or s.readback}"}
            return {"role": "assistant", "text": f"**Gemma understood:** {s.readback}",
                    "kind": "backtest", "payload": s}
        if r.intent == "evaluate":
            snap, report = _cached(("eval", r.ticker),
                                   lambda: (lambda sr: (sr[0], sr[1].model_dump()))(gc.evaluate_profiles(r.ticker)))
            return {"role": "assistant", "text": r.reply, "kind": "evaluate",
                    "payload": (r.ticker, snap, report)}
        if r.intent == "signal":
            return {"role": "assistant", "text": r.reply, "kind": "signal",
                    "payload": _cached(("signal", r.ticker), lambda: insights.signal(r.ticker))}
        if r.intent == "price":
            return {"role": "assistant", "text": r.reply, "kind": "price",
                    "payload": {"ticker": r.ticker, **insights.history_score(r.ticker)}}
    except Exception as e:
        return {"role": "assistant", "kind": "text", "payload": None,
                "text": f"I couldn't get data for {r.ticker or 'that'} ({e}). Check the ticker and try again."}

    return {"role": "assistant", "text": r.reply, "kind": "text", "payload": None}