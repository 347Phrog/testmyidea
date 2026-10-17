# rapidhack-26
My first every Hackathon hosted at the University of Rhode Island


# 📈 TestMyIdea

**Type a trading idea in plain English. See how it would have performed before you risk real money.**

Built at Hack@URI 2026 with **Gemma 4**.

## The problem

New investors trade on gut rules like "buy the dip" or "sell after a pump," but almost nobody tests them, because testing a strategy takes code. TestMyIdea lets anyone test an idea in one sentence.

## What it does

**🧪 Test an idea**
1. You type a rule: *"buy Apple when it drops 3% in a day, sell after a week."*
2. **Gemma 4** translates it into a precise, structured strategy and reads it back to you ("Gemma understood: …"), so you can see exactly what will be tested.
3. A backtest engine runs it on 5 years of real price data and compares it to simply holding the stock: return, worst drawdown, trade count, and win rate.
4. If an idea can't be tested (news, earnings, tweets), Gemma says so instead of faking a result.

**🔍 Evaluate a stock**
- Pulls real numbers (P/E, revenue growth, dividend yield, volatility, 1-year return).
- **Gemma 4** explains how well the stock fits four investing styles (Value, Growth, Income, Momentum), citing only those numbers.
- One click tests each style's rule in the backtester.
- It describes *fit*, never "buy" or "sell." The decision is always the user's.

## How Gemma is used

| Job | What Gemma does |
|---|---|
| Strategy parser | Plain English → validated JSON strategy (entry rule, exit rule, ticker) |
| Investor profiles | Real metrics → fit rating, 2 cited reasons, and a key risk per style |
| Results explainer | Backtest numbers → plain-English summary with an honest caveat |

Every Gemma output is validated against a schema, and invalid output is retried with the error. Gemma 4 is an open model, so the same pipeline can run fully on-device; for the hackathon it runs through Google AI Studio.

## Backtest rules (why the results are honest)

- Signals use only data up to a day's **close**; trades enter at the **next day's open**. No look-ahead.
- One position at a time, long only, $10,000 start.
- No fees or slippage, and split/dividend-adjusted prices.
- Warns when there are too few trades to trust the result.

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Optional: set `GEMMA_MODEL` to `gemma-4-26b-a4b-it` or `gemma-4-31b-it`.

## Files

- `app.py`: Streamlit interface
- `gemma_core.py`: Gemma prompts, schemas, and stock data
- `backtest.py`: backtest engine and metrics

## Disclaimer

Educational tool using historical data. Not financial advice. Past performance doesn't predict future results.
