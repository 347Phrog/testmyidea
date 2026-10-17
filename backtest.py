"""
Backtest engine: takes a Strategy from gemma_core and returns results.

Rules (state these in the pitch, they're what make it honest):
- Signals use only data up to day t's CLOSE; the trade ENTERS at day t+1's OPEN (no look-ahead).
- Exits are checked on each day's close: hold_days reached, +take_profit, or -stop_loss.
- One position at a time, long only, all-in. No fees or slippage (say so).
- Prices are split/dividend adjusted (yfinance history default).
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = "cache"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_prices(ticker: str, years: int = 5) -> pd.DataFrame:
    """Daily OHLC with 1 extra year so long moving averages are warmed up. Cached to CSV."""
    ticker = ticker.upper()
    path = os.path.join(CACHE_DIR, f"{ticker}_prices.csv")
    try:
        df = yf.Ticker(ticker).history(period=f"{years + 1}y")
        if df.empty:
            raise ValueError(f"No price data for {ticker}")
        df = df[["Open", "High", "Low", "Close"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(path)
        return df
    except Exception:
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise


# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def entry_signal(df: pd.DataFrame, entry) -> pd.Series:
    """True on days whose CLOSE meets the entry rule."""
    close = df["Close"]
    w = max(int(entry.window or 1), 1)
    if entry.type == "pct_change":
        chg = (close / close.shift(w) - 1) * 100
        sig = chg <= entry.value if entry.direction == "below" else chg >= entry.value
    elif entry.type == "rsi":
        r = rsi(close, w if w > 1 else 14)
        sig = r < entry.value if entry.direction == "below" else r > entry.value
    elif entry.type == "ma_cross":
        ma = close.rolling(w).mean()
        above, prev_above = close > ma, close.shift(1) > ma.shift(1)
        if entry.direction == "above":
            sig = above & ~prev_above & ma.shift(1).notna()
        else:
            sig = ~above & prev_above & ma.shift(1).notna()
    else:
        raise ValueError(f"Unknown entry type {entry.type}")
    return sig.fillna(False).astype(bool)


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
def run_backtest(strategy, df: pd.DataFrame | None = None, start_cash: float = 10_000.0) -> dict:
    if not strategy.supported:
        raise ValueError(strategy.unsupported_reason or "Strategy not supported")

    years = strategy.years or 5
    if df is None:
        df = load_prices(strategy.ticker or "SPY", years)

    sig_full = entry_signal(df, strategy.entry)
    # trim warm-up year, keep the requested window
    cutoff = df.index[-1] - pd.DateOffset(years=years)
    df = df[df.index >= cutoff]
    sig = sig_full.loc[df.index].to_numpy()
    if len(df) < 30:
        raise ValueError("Not enough price history to test this.")

    o, c, dates = df["Open"].to_numpy(), df["Close"].to_numpy(), df.index
    ex = strategy.exit
    hold = ex.hold_days
    tp = ex.take_profit_pct / 100 if ex.take_profit_pct else None
    sl = ex.stop_loss_pct / 100 if ex.stop_loss_pct else None

    cash, shares, entry_px, entry_i = start_cash, 0.0, None, None
    equity, trades = np.empty(len(df)), []
    in_mkt = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        # enter at today's open if YESTERDAY's close gave a signal
        if shares == 0 and i > 0 and sig[i - 1]:
            shares, entry_px, entry_i = cash / o[i], o[i], i
            cash = 0.0

        if shares > 0:
            in_mkt[i] = True
            ret = c[i] / entry_px - 1
            held = i - entry_i + 1
            reason = None
            if sl is not None and ret <= -sl:
                reason = "stop loss"
            elif tp is not None and ret >= tp:
                reason = "take profit"
            elif hold is not None and held >= hold:
                reason = f"held {hold} days"
            if reason:
                cash, shares = shares * c[i], 0.0
                trades.append({"entry_date": dates[entry_i].date(), "exit_date": dates[i].date(),
                               "entry_price": round(entry_px, 2), "exit_price": round(c[i], 2),
                               "return_pct": round(ret * 100, 2), "days": held, "exit_reason": reason})

        equity[i] = cash + shares * c[i]

    if shares > 0:  # still open at the end: report it, marked open
        trades.append({"entry_date": dates[entry_i].date(), "exit_date": None,
                       "entry_price": round(entry_px, 2), "exit_price": round(c[-1], 2),
                       "return_pct": round((c[-1] / entry_px - 1) * 100, 2),
                       "days": len(df) - entry_i, "exit_reason": "still open"})

    eq = pd.Series(equity, index=dates, name="Strategy")
    bh = pd.Series(start_cash * c / o[0], index=dates, name="Buy & hold")
    return {"equity": eq, "buy_hold": bh, "trades": pd.DataFrame(trades),
            "metrics": compute_metrics(eq, bh, trades, in_mkt)}


def _max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min() * 100)


def _sharpe(s: pd.Series) -> float | None:
    r = s.pct_change().dropna()
    return None if r.std() == 0 or np.isnan(r.std()) else float(r.mean() / r.std() * np.sqrt(252))


def compute_metrics(eq: pd.Series, bh: pd.Series, trades: list, in_mkt) -> dict:
    closed = [t for t in trades if t["exit_reason"] != "still open"]
    rets = [t["return_pct"] for t in closed]
    sharpe = _sharpe(eq)
    return {
        "strategy_return_pct": round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
        "buy_hold_return_pct": round(float(bh.iloc[-1] / bh.iloc[0] - 1) * 100, 1),
        "strategy_max_drawdown_pct": round(_max_drawdown(eq), 1),
        "buy_hold_max_drawdown_pct": round(_max_drawdown(bh), 1),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "buy_hold_sharpe": round(_sharpe(bh) or 0, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(100 * float(np.mean([r > 0 for r in rets])), 1) if rets else None,
        "avg_trade_pct": round(float(np.mean(rets)), 2) if rets else None,
        "time_in_market_pct": round(100 * float(np.mean(in_mkt)), 1),
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
    }
