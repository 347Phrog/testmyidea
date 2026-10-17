![TestMyIdea](logo/logo_wordmark.svg)

# TestMyIdea

**Type a trading idea in plain English. See how it would have performed before you risk real money.**

Built at Hack@URI 2026 with **Gemma 4**. Try it live: **[testmyidea.streamlit.app](https://testmyidea.streamlit.app)**

> ## ⚠️ Not financial advice
> TestMyIdea is an **educational tool**. Nothing in this app is a recommendation to buy, sell, or hold any security.
> - Backtests show how a rule **would have** performed on **past** data. Past performance does not predict future results.
> - The signal score **describes** recent price trends and news tone. It is **not** a prediction.
> - Investing-style ratings describe how a stock **fits** a style, never whether you should own it.
> - Results ignore fees, taxes, and slippage, and data can be delayed or incomplete.
>
> **Every investment decision is yours.** Consider talking to a licensed financial professional before investing.

---

## The problem

New investors trade on gut rules like *"buy the dip"* or *"sell after a pump"*, but almost nobody tests them, because testing a strategy takes code. TestMyIdea lets anyone test an idea in one sentence, and understand a stock through the eyes of different kinds of investors.

## Features

| Tab | What it does |
|---|---|
| 🏠 **Home** | What the app does, plus a guide to how Value, Growth, Income, and Momentum investors think |
| 💬 **Ask** | Chat about any stock. Gemma reads your question and runs the right tool: backtest, signal score, style fit, price chart, or a plain-English explanation |
| 🧪 **Test an idea** | Type a trading rule in plain English. Gemma turns it into an exact strategy and shows you what it understood, then it is backtested on 5 years of real prices vs. buy-and-hold |
| 🔍 **Evaluate a stock** | See how four investing styles rate a stock (citing real numbers), test each style's rule, and get a 0–100 signal score |
| 👤 **My profile** | Income, expenses, savings goal, and a risk quiz. Shows your savings pace, emergency-fund check, matching styles, and a max-per-stock guardrail used across the app |
| 🎨 **Color schemes** | Six themes with a live color display |

## How Gemma 4 is used

| Job | What Gemma does |
|---|---|
| Strategy parser | Plain English → a validated JSON trading rule (entry, exit, ticker), read back to the user |
| Chat router | Decides which tool answers each message, and remembers the conversation |
| News sentiment | Scores each recent headline from −1 (bad for shareholders) to +1 (good) |
| Style profiles | Rates fit for four investing styles using **only** the numbers it is given |
| Results explainer | Summarizes a backtest in plain words with an honest caveat |

**Gemma never invents numbers.** Every price, return, and metric is computed in code from real data. Gemma's outputs are validated against a schema and retried if invalid. Gemma 4 runs through Google AI Studio, with optional fallback to an on-device Gemma 4 model (Ollama).

## How the numbers are calculated

**Backtests (honest by design)**
- Signals use only data up to a day's **close**; trades enter at the **next day's open**, so there is no look-ahead.
- One position at a time, long only, $10,000 start, split/dividend-adjusted prices.
- No fees or slippage. A warning appears when there are too few trades to trust the result.

**Signal score (0–100)**
- **Price history (−1 to +1):** average of trend vs. the 200-day average, 6-month momentum, distance from the 52-week high, and volatility.
- **News sentiment (−1 to +1):** average of Gemma's headline scores (sources: Yahoo Finance, then Google News).
- **Score = 50 + 25 × history + 25 × sentiment.** 60+ "leans favorable", 40 or below "leans unfavorable".
- The weights are transparent heuristics, not fitted or validated predictors.

## Privacy

- **Your financial profile is never sent to Gemma or any server.** All profile math runs locally in plain Python.
- Running locally, it is saved only in `user_profile.json` on your computer (excluded from Git).
- On the public site, it lives only in your browser session and disappears when you close the tab.

## Run it locally

```bash
git clone https://github.com/347Phrog/testmyidea
cd testmyidea
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (never commit it):

```dotenv
GEMINI_API_KEY=your-key
GEMMA_MODEL=gemma-4-26b-a4b-it
GEMMA_JSON_MODE=0
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then run `streamlit run app.py`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `gemma_core.py` | Gemma calls, prompts, schemas, stock data |
| `assistant.py` | Chat routing |
| `insights.py` | Signal score and headlines |
| `backtest.py` | Backtest engine and metrics |
| `user_profile.py` | Local investor profile |
| `branding.py` | Logo and color schemes |
| `cache/` | Saved price data so demos work offline |

## What's next

- Validate the signal score itself: backtest whether high-score stocks actually outperformed.
- Add trading fees and slippage to backtests.
- Support multi-condition strategies (e.g. RSI **and** moving average).

---

**⚠️ Reminder: TestMyIdea is for education only and is not financial advice. Past results do not guarantee future returns. All investment decisions are your own.**