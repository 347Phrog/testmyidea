"""
Gemma core for the Plain-English Backtester + Investor Profiles.

Setup (Google AI Studio, default):
    pip install -r requirements.txt
    Get a key at https://aistudio.google.com/apikey and put it in a .env file next to this script:
        GEMINI_API_KEY=your-key
        GEMMA_MODEL=gemma-4-31b-it
        GEMMA_JSON_MODE=0
    (.env is in .gitignore, so it never gets pushed. `export` in the terminal also still works.)

Backup (local Ollama):
    pip install ollama ; ollama pull <gemma tag>
    export GEMMA_BACKEND=ollama ; export OLLAMA_MODEL=<that tag>

Contents:
    1. Strategy parser   -> parse_strategy("buy apple when it drops 3%...")
    2. Stock snapshot    -> get_snapshot("AAPL")   (real numbers, cached for offline demo)
    3. Investor profiles -> evaluate_profiles("AAPL")
    4. Default strategy per profile (for the "Test this style" button)
    5. Results explainer -> explain_results(strategy, metrics)   (nice-to-have)
"""
import json
import os
import re
import time
from typing import List, Literal, Optional

import numpy as np
import yfinance as yf
from pydantic import BaseModel, Field, ValidationError

# Load settings from a .env file next to this script, if present (never committed: it's in .gitignore)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

BACKEND = os.getenv("GEMMA_BACKEND", "studio")                    # "studio" or "ollama"
STUDIO_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")     # or "gemma-4-31b-it" (bigger, slower)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
CACHE_DIR = "cache"

DISCLAIMER = (
    "This shows how different investing styles would view this stock, "
    "based on past data. It is not financial advice. The decision is yours."
)


# =====================================================================
# Shared: call Gemma, pull JSON out, validate against the schema, retry
# =====================================================================
_client = None
_studio_json_mode = os.getenv("GEMMA_JSON_MODE", "1") != "0"  # flips off automatically if the API rejects JSON-schema mode for Gemma


def _studio_generate(system_prompt: str, user_text: str, schema: type[BaseModel]) -> str:
    global _client, _studio_json_mode
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)

    for attempt in range(4):
        try:
            if _studio_json_mode:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema(),
                )
            else:
                config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0)
            resp = _client.models.generate_content(model=STUDIO_MODEL, contents=user_text, config=config)
            return resp.text or ""
        except Exception as e:
            msg = str(e)
            if _studio_json_mode and any(k in msg for k in ("400", "INVALID_ARGUMENT", "500", "INTERNAL")):
                _studio_json_mode = False                       # Gemma may not accept JSON mode: use prompt-only
                continue
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "INTERNAL", "UNAVAILABLE")):
                time.sleep(2 * (attempt + 1))                   # rate limit / server hiccup: back off and retry
                continue
            raise
    raise RuntimeError("Google AI Studio kept rate-limiting; wait a minute and try again.")


def _ollama_generate(system_prompt: str, user_text: str, schema: type[BaseModel]) -> str:
    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
        format=schema.model_json_schema(),
        options={"temperature": 0},
    )
    return resp["message"]["content"]


def _extract_json(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if start != -1 and end > start else raw


def call_gemma(system_prompt: str, user_text: str, schema: type[BaseModel], retries: int = 2):
    generate = _ollama_generate if BACKEND == "ollama" else _studio_generate
    prompt, last_err = user_text, None
    for _ in range(retries + 1):
        raw = generate(system_prompt, prompt, schema)
        try:
            return schema.model_validate_json(_extract_json(raw))
        except ValidationError as e:
            last_err = e
            prompt = (f"{user_text}\n\nYour previous answer was invalid:\n{raw}\n\n"
                      f"Errors: {e}\nReturn corrected JSON only.")
    raise RuntimeError(f"Gemma failed after retries: {last_err}")


# =====================================================================
# 1. STRATEGY PARSER
# =====================================================================
class Entry(BaseModel):
    type: Literal["pct_change", "rsi", "ma_cross"]
    direction: Literal["above", "below"]
    value: Optional[float] = None  # pct_change: signed % (-3 = drop of 3%); rsi: level; ma_cross: null
    window: int = 1                # pct_change: lookback days; rsi: period; ma_cross: MA length


class Exit(BaseModel):
    hold_days: Optional[int] = None
    take_profit_pct: Optional[float] = None  # positive, e.g. 10 = +10%
    stop_loss_pct: Optional[float] = None    # positive, e.g. 5 = exit at -5%


class Strategy(BaseModel):
    supported: bool
    unsupported_reason: Optional[str] = None
    ticker: Optional[str] = None
    entry: Optional[Entry] = None
    exit: Optional[Exit] = None
    years: int = 5
    readback: str  # one sentence: "Gemma understood: ..."


STRATEGY_PROMPT = """You convert a user's trading idea into a JSON rule for a backtester.

ENTRY RULES (pick exactly one):
- pct_change: price changes by `value` percent over `window` trading days.
  direction "below" = change is at or below value (a drop, value negative).
  direction "above" = change is at or above value (a rise, value positive).
- rsi: RSI with period `window` (default 14) is "below" or "above" `value`.
- ma_cross: closing price crosses "above" or "below" its `window`-day moving average. value = null.

EXIT RULES (at least one): hold_days, take_profit_pct, stop_loss_pct (positive numbers).
Time conversions in trading days: 1 week = 5, 1 month = 21, 3 months = 63, 1 year = 252.
If the user gives no exit, use hold_days = 10.

OTHER RULES:
- ticker: uppercase symbol. Convert company names (Apple -> AAPL, Nvidia -> NVDA). If none given, use SPY.
- years: how much history to test, default 5.
- If the idea needs anything not listed (earnings, news, tweets, fundamentals, options,
  shorting, more than one entry condition), set supported=false, give a short
  unsupported_reason, and set entry and exit to null.
- readback: ONE plain sentence restating exactly what you encoded.
- Never invent numbers the user didn't give, except the defaults above.

EXAMPLES
User: buy apple when it drops 3% in a day, sell after a week
{"supported": true, "unsupported_reason": null, "ticker": "AAPL", "entry": {"type": "pct_change", "direction": "below", "value": -3, "window": 1}, "exit": {"hold_days": 5, "take_profit_pct": null, "stop_loss_pct": null}, "years": 5, "readback": "Buy AAPL after a 1-day drop of 3% or more, sell 5 trading days later."}

User: buy nvidia when RSI goes under 30, take profit at 10% or stop out at 5%
{"supported": true, "unsupported_reason": null, "ticker": "NVDA", "entry": {"type": "rsi", "direction": "below", "value": 30, "window": 14}, "exit": {"hold_days": null, "take_profit_pct": 10, "stop_loss_pct": 5}, "years": 5, "readback": "Buy NVDA when 14-day RSI drops below 30, exit at +10% or -5%."}

User: get into SPY when it crosses above the 200 day average and hold 3 months
{"supported": true, "unsupported_reason": null, "ticker": "SPY", "entry": {"type": "ma_cross", "direction": "above", "value": null, "window": 200}, "exit": {"hold_days": 63, "take_profit_pct": null, "stop_loss_pct": null}, "years": 5, "readback": "Buy SPY when price crosses above its 200-day average, sell 63 trading days later."}

User: buy the dip on microsoft
{"supported": true, "unsupported_reason": null, "ticker": "MSFT", "entry": {"type": "pct_change", "direction": "below", "value": -3, "window": 1}, "exit": {"hold_days": 10, "take_profit_pct": null, "stop_loss_pct": null}, "years": 5, "readback": "Buy MSFT after a 1-day drop of 3% or more (my reading of 'the dip'), sell 10 trading days later."}

User: buy tesla before earnings if the CEO tweets something positive
{"supported": false, "unsupported_reason": "Earnings dates and social media posts aren't supported yet.", "ticker": "TSLA", "entry": null, "exit": null, "years": 5, "readback": "Can't test this yet: it depends on earnings dates and tweets."}

Return only JSON."""


def parse_strategy(user_text: str) -> Strategy:
    try:
        s = call_gemma(STRATEGY_PROMPT, user_text, Strategy)
    except Exception as e:
        return Strategy(supported=False, unsupported_reason=str(e),
                        readback="Sorry, I couldn't understand that strategy. Try rephrasing.")
    # sanity checks Gemma can't be trusted with
    if s.supported:
        if s.entry is None or s.exit is None or not any(
            [s.exit.hold_days, s.exit.take_profit_pct, s.exit.stop_loss_pct]
        ):
            s.supported = False
            s.unsupported_reason = "Missing entry or exit rule."
        elif s.entry.type == "rsi" and s.entry.value is None:
            s.supported = False
            s.unsupported_reason = "RSI rule needs a level (like 30)."
        elif s.entry.type == "pct_change" and s.entry.value is None:
            s.supported = False
            s.unsupported_reason = "Price-change rule needs a percent."
    return s


# =====================================================================
# 2. STOCK SNAPSHOT (real numbers only; cached for wifi failures)
# =====================================================================
def _r(x, n=2):
    try:
        return None if x is None or np.isnan(x) else round(float(x), n)
    except TypeError:
        return None


def get_snapshot(ticker: str) -> dict:
    ticker = ticker.upper()
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist.empty:
            raise ValueError("no price data")
        try:
            info = t.info or {}
        except Exception:
            info = {}

        close = hist["Close"]
        price = close.iloc[-1]
        rets = close.pct_change().dropna()
        rev_g = info.get("revenueGrowth")
        # dividend yield computed ourselves (yfinance's field has changed units before)
        div_12m = hist["Dividends"].sum() if "Dividends" in hist else 0.0

        snap = {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "n/a"),
            "price": _r(price),
            "return_1y_pct": _r((price / close.iloc[0] - 1) * 100, 1),
            "volatility_annual_pct": _r(rets.std() * np.sqrt(252) * 100, 1),
            "pct_from_52w_high": _r((price / close.max() - 1) * 100, 1),
            "trailing_pe": _r(info.get("trailingPE"), 1),
            "forward_pe": _r(info.get("forwardPE"), 1),
            "revenue_growth_pct": _r(rev_g * 100, 1) if rev_g is not None else None,
            "dividend_yield_pct": _r(div_12m / price * 100),
            "beta": _r(info.get("beta")),
        }
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snap, f)
        return snap
    except Exception:
        if os.path.exists(path):
            with open(path) as f:
                snap = json.load(f)
            snap["_cached"] = True
            return snap
        raise


# =====================================================================
# 3. INVESTOR PROFILES
# =====================================================================
ProfileName = Literal["Value", "Growth", "Income", "Momentum"]


class ProfileView(BaseModel):
    profile: ProfileName
    fit: Literal["Strong fit", "Mixed", "Poor fit"]
    reasons: List[str] = Field(min_length=2, max_length=2)
    key_risk: str


class ProfileReport(BaseModel):
    views: List[ProfileView] = Field(min_length=4, max_length=4)
    one_line_summary: str


# Thresholds are rough rules of thumb, stated so Gemma's reasoning is consistent and explainable.
PROFILE_PROMPT = """You describe how well a stock FITS four investing styles.
You do not recommend anything. Never use the words "buy", "sell", or "recommend".

STYLES
- Value: prefers trailing P/E under about 18 and price well below its 52-week high.
- Growth: prefers revenue growth above 15%; accepts high P/E and high volatility.
- Income: prefers dividend yield above 2.5%, annual volatility under 25%, beta under 1.
- Momentum: prefers 1-year return above 15% and price within 10% of its 52-week high.

RULES
- Use ONLY the numbers in DATA. Do not use outside knowledge about the company.
- Every reason must cite a number from DATA (e.g. "P/E of 31 is above the value range").
- If a number a style needs is null, say it's unavailable and lean "Mixed".
- Exactly 2 reasons per style, each under 20 words. One key_risk under 20 words.
- Return all four styles in this order: Value, Growth, Income, Momentum.
- one_line_summary: which styles this stock suits best and least, in one sentence.

Return only JSON."""


def evaluate_profiles(ticker: str):
    snap = get_snapshot(ticker)
    data = {k: v for k, v in snap.items() if not k.startswith("_")}
    report = call_gemma(PROFILE_PROMPT, "DATA:\n" + json.dumps(data, indent=2), ProfileReport)
    return snap, report


# =====================================================================
# 4. DEFAULT STRATEGY PER PROFILE ("Test this style" button)
#    Illustrative rules, not claims about how real investors trade.
# =====================================================================
def profile_strategy(profile: str, ticker: str) -> Strategy:
    t = ticker.upper()
    presets = {
        "Value": (Entry(type="rsi", direction="below", value=30, window=14), Exit(hold_days=21),
                  f"Buy {t} when it looks oversold (RSI under 30), hold 1 month."),
        "Growth": (Entry(type="ma_cross", direction="above", window=200), Exit(hold_days=126),
                   f"Buy {t} when it crosses above its 200-day average, hold 6 months."),
        "Income": (Entry(type="pct_change", direction="below", value=-5, window=5), Exit(hold_days=63),
                   f"Buy {t} after a 5% weekly dip, hold 3 months."),
        "Momentum": (Entry(type="ma_cross", direction="above", window=50),
                     Exit(take_profit_pct=10, stop_loss_pct=5),
                     f"Buy {t} when it crosses above its 50-day average, exit at +10% or -5%."),
    }
    entry, exit_, readback = presets[profile]
    return Strategy(supported=True, ticker=t, entry=entry, exit=exit_, readback=readback)


# =====================================================================
# 5. RESULTS EXPLAINER (nice-to-have): plain-English summary of a backtest
# =====================================================================
class Explanation(BaseModel):
    summary: str
    caveat: str


EXPLAIN_PROMPT = """You explain a trading-strategy backtest to a beginner investor.
Use ONLY the numbers in RESULTS. Compare the strategy to buy-and-hold.
summary: 2-3 short sentences. caveat: 1 sentence on why past results may not repeat
(mention the trade count if it is under 20). Never say "buy", "sell", or "recommend".
Return only JSON."""


def explain_results(strategy: Strategy, metrics: dict) -> Explanation:
    payload = {"strategy": strategy.readback, **metrics}
    return call_gemma(EXPLAIN_PROMPT, "RESULTS:\n" + json.dumps(payload, indent=2), Explanation)


# =====================================================================
# Quick test:  python gemma_core.py
# =====================================================================
if __name__ == "__main__":
    for q in [
        "buy apple when it drops 3% in a day and sell after a week",
        "get into nvidia when rsi is under 30, take profit at 10 percent",
        "buy coke when the CEO goes on TV",
    ]:
        s = parse_strategy(q)
        print(q, "\n  ->", s.readback, "| supported:", s.supported,
              "" if s.supported else f"| reason: {s.unsupported_reason}", "\n")

    # Warm the cache for demo tickers so the demo survives bad wifi
    for tk in ["AAPL", "KO", "NVDA", "SPY"]:
        try:
            get_snapshot(tk)
        except Exception as e:
            print("cache failed for", tk, e)

    snap, rep = evaluate_profiles("KO")
    print(json.dumps(snap, indent=2))
    for v in rep.views:
        print(f"{v.profile}: {v.fit} | {v.reasons} | risk: {v.key_risk}")
    print(rep.one_line_summary)
    print("\n" + DISCLAIMER)