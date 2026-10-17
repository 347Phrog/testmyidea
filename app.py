"""
Run:  streamlit run app.py
Needs GEMINI_API_KEY set (or GEMMA_BACKEND=ollama).
"""
import json

import pandas as pd
import streamlit as st

import backtest as bt
import gemma_core as gc

st.set_page_config(page_title="TestMyIdea", page_icon="📈", layout="wide")

FIT_COLOR = {"Strong fit": "green", "Mixed": "orange", "Poor fit": "red"}
EXAMPLES = [
    "buy apple when it drops 3% in a day, sell after a week",
    "buy nvidia when RSI goes under 30, take profit at 10% or stop out at 5%",
    "get into SPY when it crosses above the 200 day average and hold 3 months",
]


@st.cache_data(show_spinner=False, ttl=3600)
def cached_prices(ticker: str, years: int) -> pd.DataFrame:
    return bt.load_prices(ticker, years)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_profiles(ticker: str):
    snap, report = gc.evaluate_profiles(ticker)
    return snap, report.model_dump()


def show_results(strategy: gc.Strategy, key: str):
    """Run the backtest for a Strategy and render metrics, chart, trades."""
    try:
        with st.spinner(f"Backtesting {strategy.ticker}..."):
            res = bt.run_backtest(strategy, df=cached_prices(strategy.ticker, strategy.years))
    except Exception as e:
        st.error(f"Couldn't run that backtest: {e}")
        return
    m = res["metrics"]
    st.caption(f"{strategy.ticker} · {m['start']} to {m['end']} · $10,000 start · no fees")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy return", f"{m['strategy_return_pct']}%",
              f"{m['strategy_return_pct'] - m['buy_hold_return_pct']:+.1f} pts vs buy & hold")
    c2.metric("Buy & hold return", f"{m['buy_hold_return_pct']}%")
    c3.metric("Worst drop (strategy)", f"{m['strategy_max_drawdown_pct']}%",
              f"buy & hold: {m['buy_hold_max_drawdown_pct']}%", delta_color="off")
    c4.metric("Trades", m["num_trades"],
              f"win rate {m['win_rate_pct']}%" if m["win_rate_pct"] is not None else "no closed trades",
              delta_color="off")

    st.line_chart(pd.concat([res["equity"], res["buy_hold"]], axis=1), y_label="Portfolio value ($)")

    if m["num_trades"] < 20:
        st.warning(f"Only {m['num_trades']} trades: too few to trust this result much.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        with st.expander(f"All {m['num_trades']} trades"):
            st.dataframe(res["trades"], hide_index=True)
    with col_b:
        with st.expander("More stats"):
            st.write({"Sharpe (strategy)": m["sharpe"], "Sharpe (buy & hold)": m["buy_hold_sharpe"],
                      "Avg trade %": m["avg_trade_pct"], "Time in market %": m["time_in_market_pct"]})

    if st.button("✨ Explain these results", key=f"explain_{key}"):
        try:
            with st.spinner("Gemma is reading the results..."):
                exp = gc.explain_results(strategy, m)
            st.info(exp.summary)
            st.caption(exp.caveat)
        except Exception as e:
            st.error(f"Gemma couldn't explain this one: {e}")


# ---------------------------------------------------------------------
st.title("📈 TestMyIdea")
st.write("Type a trading idea in plain English. See how it would have done **before** you risk real money.")
st.caption(f"Powered by Gemma ({gc.STUDIO_MODEL if gc.BACKEND != 'ollama' else gc.OLLAMA_MODEL})")

tab1, tab2 = st.tabs(["🧪 Test an idea", "🔍 Evaluate a stock"])

# ---------------- TAB 1: plain English -> backtest ----------------
with tab1:
    st.write("Try one:")
    ex_cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i].button(ex, key=f"ex{i}"):
            st.session_state["idea"] = ex

    idea = st.text_area("Your trading idea", key="idea", height=80,
                        placeholder="e.g. buy tesla when it drops 5% in a week and sell after a month")

    if st.button("Test it", type="primary", disabled=not idea.strip()):
        with st.spinner("Gemma is reading your idea..."):
            st.session_state["strategy"] = gc.parse_strategy(idea)

    s = st.session_state.get("strategy")
    if s is not None:
        if s.supported:
            st.success(f"**Gemma understood:** {s.readback}")
        else:
            st.warning(f"**Can't test that yet:** {s.unsupported_reason or s.readback}\n\n"
                       "I can test: price drops/rises over N days, RSI levels, moving-average "
                       "crosses, with exits after N days or at a profit/loss target.")
        with st.expander("What Gemma produced (raw JSON)"):
            st.code(json.dumps(s.model_dump(), indent=2), language="json")
        if s.supported:
            show_results(s, key="tab1")

# ---------------- TAB 2: investor profiles ----------------
with tab2:
    st.write("See how four investing styles would view a stock, using real numbers.")
    ticker = st.text_input("Ticker", value="KO", max_chars=8).strip().upper()

    if st.button("Evaluate", type="primary", disabled=not ticker):
        try:
            with st.spinner(f"Pulling data and asking Gemma about {ticker}..."):
                st.session_state["profiles"] = (ticker, *cached_profiles(ticker))
            st.session_state.pop("style_test", None)
        except Exception as e:
            st.error(f"Couldn't evaluate {ticker}: {e}")

    if "profiles" in st.session_state:
        tk, snap, report = st.session_state["profiles"]
        st.subheader(f"{snap.get('name', tk)} ({tk})")
        if snap.get("_cached"):
            st.caption("Using saved data (live data unavailable).")

        with st.expander("The numbers Gemma was given"):
            st.json({k: v for k, v in snap.items() if not k.startswith("_")})

        cols = st.columns(4)
        for col, v in zip(cols, report["views"]):
            with col:
                st.markdown(f"### {v['profile']}")
                st.markdown(f":{FIT_COLOR[v['fit']]}[**{v['fit']}**]")
                for r in v["reasons"]:
                    st.markdown(f"- {r}")
                st.caption(f"⚠️ Risk: {v['key_risk']}")
                if st.button(f"Test {v['profile']} style", key=f"style_{v['profile']}"):
                    st.session_state["style_test"] = v["profile"]

        st.write(f"**Summary:** {report['one_line_summary']}")
        st.info(gc.DISCLAIMER)

        style = st.session_state.get("style_test")
        if style:
            st.divider()
            strat = gc.profile_strategy(style, tk)
            st.success(f"**{style} style rule:** {strat.readback}")
            show_results(strat, key="tab2")
