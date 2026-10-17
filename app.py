"""
Run:  streamlit run app.py
Needs GEMINI_API_KEY set (or GEMMA_BACKEND=ollama).
"""
import json

import pandas as pd
import streamlit as st

import assistant
import backtest as bt
import branding
import budget
import gemma_core as gc
import insights
import user_profile as up

try:
    st.set_page_config(page_title="TestMyIdea", page_icon=branding.icon_bytes(), layout="wide")
except Exception:
    st.set_page_config(page_title="TestMyIdea", page_icon="📈", layout="wide")
branding.apply_branding()

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
def cached_parse(idea: str) -> dict:
    """Same idea typed twice = instant (great for rehearsed demo ideas). Failures aren't cached."""
    s = gc.parse_strategy(idea)
    if not s.supported and s.readback.startswith("Sorry"):
        raise RuntimeError(s.unsupported_reason or "Gemma call failed")
    return s.model_dump()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_profiles(ticker: str):
    snap, report = gc.evaluate_profiles(ticker)
    return snap, report.model_dump()


@st.cache_data(show_spinner=False, ttl=1800)
def cached_signal(ticker: str) -> dict:
    return insights.signal(ticker)


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

    mine = current_profile()
    if mine:
        cap = up.summarize(mine)["max_per_stock"]
        if cap > 0:
            end = cap * (1 + m["strategy_return_pct"] / 100)
            bh_end = cap * (1 + m["buy_hold_return_pct"] / 100)
            st.caption(f"👤 With your max-per-stock amount (${cap:,.0f}), this strategy would have ended at "
                       f"**${end:,.0f}** vs **${bh_end:,.0f}** buying and holding.")

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



def render_profiles(tk: str, snap: dict, report: dict, key: str, buttons: bool = True):
    st.subheader(f"{snap.get('name', tk)} ({tk})")
    if snap.get("_cached"):
        st.caption("Using saved data (live data unavailable).")
    with st.expander("The numbers Gemma was given"):
        st.json({k: v for k, v in snap.items() if not k.startswith("_")})
    cols = st.columns(4)
    for col, v in zip(cols, report["views"]):
        with col:
            mine = current_profile()
            star = " ⭐" if mine and v["profile"] in up.summarize(mine)["styles"] else ""
            st.markdown(f"#### {v['profile']}{star}")
            st.markdown(f":{FIT_COLOR[v['fit']]}[**{v['fit']}**]")
            for r in v["reasons"]:
                st.markdown(f"- {r}")
            st.caption(f"⚠️ Risk: {v['key_risk']}")
            if buttons and st.button(f"Test {v['profile']} style", key=f"style_{key}_{v['profile']}"):
                st.session_state["style_test"] = v["profile"]
    st.write(f"**Summary:** {report['one_line_summary']}")
    if current_profile():
        st.caption("⭐ = matches the risk level in your profile")
    st.info(gc.DISCLAIMER)


def render_signal(sig: dict):
    color = "green" if sig["score"] >= 60 else "red" if sig["score"] <= 40 else "orange"
    c1, c2 = st.columns([1, 2])
    c1.metric(f"{sig['ticker']} signal score", f"{sig['score']} / 100")
    c1.markdown(f":{color}[**{sig['lean']}**]")
    c2.progress(sig["score"] / 100)
    hist, sent = sig["history"], sig["sentiment"]
    c2.caption("50 = neutral · score = 50 + 25 × price history + 25 × news sentiment "
               "(each from −1 to +1)")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**📈 Price history: {hist['score']:+.2f}**")
        for p in hist["parts"]:
            st.markdown(f"- {p['name']} ({p['score']:+.2f}): {p['detail']}")
    with b:
        if sent["score"] is None:
            st.markdown("**📰 News sentiment: unavailable**")
            st.caption(sent["summary"] + " Score uses price history only.")
        else:
            st.markdown(f"**📰 News sentiment: {sent['score']:+.2f}** ({sent['n']} headlines, scored by Gemma)")
            st.caption(sent["summary"])
            latest = sorted(sent["headlines"], key=lambda h: h.get("date") or "", reverse=True)
            st.markdown("**Latest headlines**")
            for h in latest[:3]:
                tone = "🟢" if h["score"] > 0.15 else "🔴" if h["score"] < -0.15 else "⚪"
                meta = " · ".join(x for x in (h.get("publisher"), h.get("date")) if x)
                st.markdown(f"{tone} **{h['score']:+.2f}** {h['title']}  \n<small>{meta}</small>",
                            unsafe_allow_html=True)
            if len(latest) > 3:
                with st.expander(f"All {len(latest)} headlines"):
                    st.dataframe(pd.DataFrame(latest)[["score", "title", "publisher", "date"]],
                                 hide_index=True)
    st.info("This summarizes recent price trends and news tone. It is not a prediction or "
            "financial advice. Past trends and headlines don't guarantee future returns. "
            "The decision is yours.")


def render_price(p: dict):
    close = p["close_1y"]
    chg = (close.iloc[-1] / close.iloc[0] - 1) * 100
    st.metric(f"{p['ticker']} price", f"${p['price']:,.2f}", f"{chg:+.1f}% over 1 year")
    st.line_chart(close, y_label="Price ($)")


def render_payload(m: dict, key: str):
    kind, payload = m.get("kind"), m.get("payload")
    try:
        if kind == "backtest":
            show_results(payload, key=key)
        elif kind == "evaluate":
            render_profiles(*payload, key=key, buttons=False)
        elif kind == "signal":
            render_signal(payload)
        elif kind == "price":
            render_price(payload)
    except Exception as e:
        st.error(f"Couldn't display this result: {e}")


PROFILE_GUIDE = {
    "Value": {
        "icon": "🏷️",
        "tagline": "Buy good businesses when they're on sale.",
        "mindset": "Value investors look for stocks priced below what the business is really worth. "
                   "They're patient, comfortable buying what others are selling, and willing to wait years "
                   "for the market to notice. The approach goes back to Benjamin Graham and was made famous "
                   "by Warren Buffett.",
        "looks_for": ["Low price-to-earnings (P/E), roughly under 18",
                      "Price well below its 52-week high (bought on a dip, not a peak)",
                      "Steady, understandable businesses"],
        "avoids": ["Expensive \"story\" stocks with high P/E", "Buying after a big run-up"],
        "risk": "A cheap stock can stay cheap, or get cheaper, if the business is actually declining "
                "(a \"value trap\").",
        "rule": "Buy when it looks oversold (RSI under 30), hold 1 month.",
    },
    "Growth": {
        "icon": "🚀",
        "tagline": "Pay up for companies growing fast.",
        "mindset": "Growth investors care most about how fast a company's sales and earnings are rising. "
                   "They accept high prices and big price swings, betting that rapid growth will justify "
                   "today's valuation.",
        "looks_for": ["Revenue growth above about 15% a year",
                      "Expanding markets and new products",
                      "Willingness to accept a high P/E and high volatility"],
        "avoids": ["Slow-growing, mature companies", "Businesses with shrinking sales"],
        "risk": "If growth slows even a little, high-priced stocks can fall sharply.",
        "rule": "Buy when price crosses above its 200-day average, hold 6 months.",
    },
    "Income": {
        "icon": "💵",
        "tagline": "Get paid to wait.",
        "mindset": "Income investors want steady cash from dividends and a smooth ride. They prefer "
                   "stable, established companies over exciting ones, and they value sleeping well at "
                   "night over big wins.",
        "looks_for": ["Dividend yield above about 2.5%",
                      "Low volatility (annual swings under about 25%)",
                      "Beta under 1 (moves less than the overall market)"],
        "avoids": ["Companies that pay no dividend", "Wild, unpredictable price swings"],
        "risk": "Dividends can be cut, and slow-growing stocks can lag the market for years.",
        "rule": "Buy after a 5% weekly dip, hold 3 months.",
    },
    "Momentum": {
        "icon": "📈",
        "tagline": "Ride what's already working.",
        "mindset": "Momentum investors follow the trend: stocks that have risen tend to keep rising for a "
                   "while. They don't need a story about why. They watch price action and get out "
                   "quickly when the trend breaks.",
        "looks_for": ["1-year return above about 15%",
                      "Price within 10% of its 52-week high",
                      "A clear uptrend"],
        "avoids": ["Stocks in a downtrend", "Holding losers and hoping they recover"],
        "risk": "Trends reverse suddenly. Momentum strategies can give back gains fast in a sell-off.",
        "rule": "Buy when price crosses above its 50-day average, exit at +10% or −5%.",
    },
}


def render_home():
    st.subheader("Welcome to TestMyIdea")
    st.write("Most new investors trade on gut rules like *\"buy the dip\"* without ever testing them. "
             "TestMyIdea lets you test any idea in one sentence, and shows how different kinds of "
             "investors would look at a stock, using real data.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**💬 Ask**  \nChat about any stock: price, signal score, or which style it suits.")
    c2.markdown("**🧪 Test an idea**  \nType a trading rule in plain English and see how it would "
                "have performed vs. just holding.")
    c3.markdown("**🔍 Evaluate a stock**  \nSee how four investing styles rate a stock, plus a "
                "0–100 signal score.")

    st.divider()
    st.subheader("How each investing style thinks")
    st.caption("These are the same rules Gemma uses when it rates a stock in the app.")
    tabs = st.tabs([f"{g['icon']} {name}" for name, g in PROFILE_GUIDE.items()])
    for tab, (name, g) in zip(tabs, PROFILE_GUIDE.items()):
        with tab:
            st.markdown(f"### {g['icon']} {name}: *{g['tagline']}*")
            st.write(g["mindset"])
            a, b = st.columns(2)
            with a:
                st.markdown("**✅ Looks for**")
                for x in g["looks_for"]:
                    st.markdown(f"- {x}")
            with b:
                st.markdown("**🚫 Avoids**")
                for x in g["avoids"]:
                    st.markdown(f"- {x}")
            st.markdown(f"**⚠️ Main risk:** {g['risk']}")
            st.markdown(f"**🧪 Rule we test for this style:** {g['rule']}")
            st.caption("Try it: go to 🔍 Evaluate a stock, enter a ticker, then click "
                       f"\"Test {name} style\".")

    with st.expander("🎨 Color schemes"):
        branding.color_display()

    with st.expander("📊 How the signal score works"):
        st.markdown(
            "- **Price history (−1 to +1):** average of trend vs. the 200-day average, 6-month momentum, "
            "distance from the 52-week high, and volatility.\n"
            "- **News sentiment (−1 to +1):** Gemma scores recent headlines, and we average them.\n"
            "- **Score = 50 + 25 × history + 25 × sentiment.** 60+ leans favorable, 40 or below leans "
            "unfavorable.\n\n"
            "It describes recent conditions. It does not predict the future.")
    st.info("TestMyIdea is an educational tool using historical data. It is not financial advice. "
            "Every decision is yours.")



def current_profile():
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = up.load()
    return st.session_state["user_profile"]


def render_my_profile():
    st.subheader("👤 My profile")
    if up.PERSIST:
        st.caption("🔒 Saved only on this computer (user_profile.json). It is never sent to Gemma or any "
                   "server, and every calculation here is plain math.")
    else:
        st.caption("🔒 Kept only in your browser session and cleared when you close the tab. It is "
                   "never saved on the server or sent to Gemma, and every calculation here is plain math.")
    p = current_profile() or dict(up.DEFAULTS)

    with st.form("profile_form"):
        st.markdown("**Money basics (monthly, after taxes)**")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("First name (optional)", value=p["name"])
        income = c2.number_input("Monthly income ($)", min_value=0.0, step=100.0, value=float(p["monthly_income"]))
        expenses = c3.number_input("Monthly expenses ($)", min_value=0.0, step=100.0, value=float(p["monthly_expenses"]))

        st.markdown("**Savings goal**")
        g1, g2, g3, g4 = st.columns(4)
        savings = g1.number_input("Current savings ($)", min_value=0.0, step=100.0, value=float(p["current_savings"]))
        goal_name = g2.text_input("Goal", value=p["goal_name"])
        goal_amount = g3.number_input("Goal amount ($)", min_value=0.0, step=500.0, value=float(p["goal_amount"]))
        default_date = pd.Timestamp(p["goal_date"]).date() if p.get("goal_date") else None
        goal_date = g4.date_input("Target date (optional)", value=default_date)

        st.markdown("**Risk quiz**")
        answers = []
        for i, (q, opts) in enumerate(up.RISK_QUESTIONS):
            idx = int(p["risk_answers"][i]) if i < len(p["risk_answers"]) else 1
            choice = st.radio(q, opts, index=idx, key=f"riskq{i}", horizontal=True)
            answers.append(opts.index(choice))

        st.markdown("**Guardrails**")
        h1, h2 = st.columns(2)
        max_pct = h1.slider("Max % of investable money in any one stock", 1, 50, int(p["max_pct_per_stock"]))
        favs = h2.text_input("Watchlist tickers (comma separated)", value=p["favorite_tickers"])

        saved = st.form_submit_button("💾 Save profile", type="primary")

    if saved:
        new = {"name": name.strip(), "monthly_income": income, "monthly_expenses": expenses,
               "current_savings": savings, "goal_name": goal_name.strip() or "Savings goal",
               "goal_amount": goal_amount, "goal_date": goal_date.isoformat() if goal_date else None,
               "risk_answers": answers, "max_pct_per_stock": max_pct,
               "favorite_tickers": favs.upper()}
        up.save(new)
        st.session_state["user_profile"] = new
        st.success("Saved on this computer." if up.PERSIST else "Saved for this session.")
        p = new

    if not current_profile():
        st.info("Fill this in and save to get a personal snapshot, your matching investing style, "
                "and position-size guardrails across the app.")
        return

    s = up.summarize(p)
    st.divider()
    st.subheader(f"Your snapshot{', ' + p['name'] if p['name'] else ''}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Left over each month", f"${s['surplus']:,.0f}")
    m2.metric("Savings rate", f"{s['savings_rate']:.0f}%")
    m3.metric("Risk level", s["risk_level"], f"{s['risk_total']}/6 on the quiz", delta_color="off")
    m4.metric("Styles that fit you", " & ".join(s["styles"]))

    # Goal progress
    goal = float(p["goal_amount"])
    if goal > 0:
        pct = min(float(p["current_savings"]) / goal, 1.0)
        st.markdown(f"**🎯 {p['goal_name']}: ${float(p['current_savings']):,.0f} of ${goal:,.0f}**")
        st.progress(pct)
        if s["remaining"] == 0:
            st.success("Goal reached!")
        elif s["months_at_current_pace"] is None:
            st.warning("You're not saving anything each month yet, so this goal can't be reached at the "
                       "current pace. Lowering expenses or raising income is the first step.")
        else:
            line = f"At your current pace (${s['surplus']:,.0f}/month) you'll get there in about " \
                   f"**{s['months_at_current_pace']} months**."
            if s["needed_per_month"] is not None:
                line += f" To hit your target date ({s['months_left']} months away) you need " \
                        f"**${s['needed_per_month']:,.0f}/month**: " + \
                        ("✅ you're on track." if s["on_track"] else "⚠️ you're behind that pace.")
            st.markdown(line)

    # Emergency fund + investable money
    if float(p["monthly_expenses"]) > 0:
        if s["emergency_ok"]:
            st.success(f"✅ Emergency fund covered: you have at least 3 months of expenses "
                       f"(${s['emergency_target']:,.0f}) saved.")
        else:
            st.warning(f"⚠️ A common rule of thumb is to keep 3 months of expenses "
                       f"(${s['emergency_target']:,.0f}) as an emergency fund before investing.")
    st.markdown(f"**Money beyond your emergency fund:** ${s['investable']:,.0f} · "
                f"**Your max per stock ({p['max_pct_per_stock']}%):** ${s['max_per_stock']:,.0f}")
    st.caption("These are general rules of thumb for education, not financial advice. The "
               "\"styles that fit you\" are starred on the Evaluate tab, and backtests show what your "
               "max-per-stock amount would have become.")

    if st.button("🗑️ Delete my profile", key="del_profile"):
        up.delete()
        st.session_state["user_profile"] = None
        st.rerun()


CHAT_SUGGESTIONS = [
    "Does Nvidia look good right now?",
    "What if I bought Apple every time it dropped 3%?",
    "What kind of investor is Coca-Cola good for?",
    "What is RSI?",
]


# ---------------------------------------------------------------------
try:
    branding.header()
except Exception:
    st.title("📈 TestMyIdea")
st.write("Type a trading idea in plain English. See how it would have done **before** you risk real money.")
st.caption(f"Powered by Gemma ({gc.STUDIO_MODEL if gc.BACKEND != 'ollama' else gc.OLLAMA_MODEL})")

tab_home, tab0, tab1, tab2, tab_me, tab_budget = st.tabs(
    ["🏠 Home", "💬 Ask", "🧪 Test an idea", "🔍 Evaluate a stock", "👤 My profile", "💰 Budget"])

with tab_budget:
    try:
        budget.render(current_profile())
    except Exception as e:  # never let the budget page break the rest of the app
        st.error(f"Budget page error: {e}")

with tab_me:
    try:
        render_my_profile()
    except Exception as e:  # never let the profile page break the rest of the app
        st.error(f"Profile page error: {e}")

with tab_home:
    render_home()

# ---------------- TAB 0: chat ----------------
with tab0:
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    chat = st.session_state["chat"]

    if not chat:
        st.write("Ask about any stock, or describe a trading idea to test. Try:")
        sug_cols = st.columns(len(CHAT_SUGGESTIONS))
        for i, sug in enumerate(CHAT_SUGGESTIONS):
            if sug_cols[i].button(sug, key=f"sug{i}"):
                st.session_state["pending"] = sug

    for i, m in enumerate(chat):
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m["role"] == "assistant":
                render_payload(m, key=f"chat{i}")

    if chat and st.button("Clear chat", key="clear_chat"):
        st.session_state["chat"] = []
        st.rerun()

    prompt = st.chat_input("Ask about a stock or test an idea…") or st.session_state.pop("pending", None)
    if prompt:
        history = list(chat)
        chat.append({"role": "user", "text": prompt, "kind": "text", "payload": None})
        with st.spinner("Gemma is thinking…"):
            chat.append(assistant.handle(prompt, history))
        st.rerun()

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
            try:
                st.session_state["strategy"] = gc.Strategy(**cached_parse(idea.strip()))
            except Exception as e:
                st.session_state["strategy"] = gc.Strategy(
                    supported=False, unsupported_reason=f"Gemma didn't respond in time ({e}). Try again.",
                    readback="Gemma call failed.")

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
        render_profiles(tk, snap, report, key="tab2")

        if st.button(f"📊 Signal score for {tk} (price history + news sentiment)", key="sig_btn"):
            try:
                with st.spinner(f"Scoring {tk}'s recent news with Gemma…"):
                    st.session_state["signal"] = cached_signal(tk)
            except Exception as e:
                st.error(f"Couldn't score {tk}: {e}")
        sig = st.session_state.get("signal")
        if sig and sig["ticker"] == tk:
            render_signal(sig)

        style = st.session_state.get("style_test")
        if style:
            st.divider()
            strat = gc.profile_strategy(style, tk)
            st.success(f"**{style} style rule:** {strat.readback}")
            show_results(strat, key="tab2")