"""
Budget planner: 50/30/20 rule, your actual split, and how much you could invest over time.
Pure math, no AI, nothing sent anywhere. All projections are illustrative, not predictions.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def split(income: float, needs_pct: int, wants_pct: int) -> dict:
    save_pct = max(100 - needs_pct - wants_pct, 0)
    return {"needs": income * needs_pct / 100, "wants": income * wants_pct / 100,
            "save": income * save_pct / 100, "save_pct": save_pct}


def future_value(monthly: float, annual_rate: float, years: int) -> list[float]:
    """Balance at the end of each year when investing `monthly` every month."""
    r = annual_rate / 100 / 12
    out, bal = [], 0.0
    for _ in range(years):
        for _ in range(12):
            bal = bal * (1 + r) + monthly
        out.append(bal)
    return out


def months_to_fill(target: float, have: float, monthly: float) -> int | None:
    gap = max(target - have, 0.0)
    if gap == 0:
        return 0
    if monthly <= 0:
        return None
    return int(-(-gap // monthly))  # ceiling


def render(profile: dict | None):
    st.subheader("💰 Budget planner (50/30/20)")
    st.caption("A common budgeting rule of thumb: about 50% of take-home pay for needs, 30% for wants, "
               "and 20% for saving and investing. Adjust it to fit your life. Pure math, nothing is sent "
               "anywhere, and this is not financial advice.")

    p = profile or {}
    c1, c2 = st.columns(2)
    income = c1.number_input("Monthly take-home pay ($)", min_value=0.0, step=100.0,
                             value=float(p.get("monthly_income") or 3000.0), key="bud_income")
    savings_now = c2.number_input("Savings you already have ($)", min_value=0.0, step=100.0,
                                  value=float(p.get("current_savings") or 0.0), key="bud_have")

    st.markdown("**Your target split**")
    s1, s2 = st.columns(2)
    needs_pct = s1.slider("Needs % (rent, food, bills, transport)", 0, 100, 50, key="bud_needs")
    wants_pct = s2.slider("Wants % (eating out, fun, subscriptions)", 0, 100 - needs_pct,
                          min(30, 100 - needs_pct), key="bud_wants")
    plan = split(income, needs_pct, wants_pct)
    st.caption(f"Saving & investing gets the rest: **{plan['save_pct']}%**")

    st.markdown("**What you actually spend each month (optional)**")
    a1, a2 = st.columns(2)
    act_needs = a1.number_input("Actual needs ($)", min_value=0.0, step=50.0, value=0.0, key="bud_act_needs")
    act_wants = a2.number_input("Actual wants ($)", min_value=0.0, step=50.0, value=0.0, key="bud_act_wants")
    has_actual = act_needs + act_wants > 0
    act_save = max(income - act_needs - act_wants, 0.0)

    rows = {"Needs": [plan["needs"]], "Wants": [plan["wants"]], "Save & invest": [plan["save"]]}
    if has_actual:
        rows = {"Needs": [plan["needs"], act_needs], "Wants": [plan["wants"], act_wants],
                "Save & invest": [plan["save"], act_save]}
    df = pd.DataFrame(rows, index=["Target", "Actual"][:len(next(iter(rows.values())))]).T
    m1, m2, m3 = st.columns(3)
    m1.metric("Needs", f"${plan['needs']:,.0f}", f"{needs_pct}%", delta_color="off")
    m2.metric("Wants", f"${plan['wants']:,.0f}", f"{wants_pct}%", delta_color="off")
    m3.metric("Save & invest", f"${plan['save']:,.0f}", f"{plan['save_pct']}%", delta_color="off")
    st.bar_chart(df, stack=False, y_label="$ per month")

    if has_actual:
        if act_needs + act_wants > income:
            st.error(f"You're spending ${act_needs + act_wants - income:,.0f}/month more than you earn. "
                     "Fixing that comes before any investing.")
        elif act_save < plan["save"]:
            st.warning(f"You're saving ${act_save:,.0f}/month, ${plan['save'] - act_save:,.0f} short of "
                       f"your {plan['save_pct']}% target. Trimming wants is usually the fastest fix.")
        else:
            st.success(f"You're saving ${act_save:,.0f}/month, at or above your target. 🎉")

    monthly_save = act_save if has_actual else plan["save"]

    # --- How it flows into investing -------------------------------------------------
    st.divider()
    st.markdown("### 📈 How this affects your investing")
    expenses = (act_needs + act_wants) if has_actual else (plan["needs"] + plan["wants"])
    e_target = expenses * 3  # rule of thumb: 3 months of expenses
    months = months_to_fill(e_target, savings_now, monthly_save)
    if months == 0:
        st.success(f"✅ Your emergency fund (3 months of expenses, ${e_target:,.0f}) is covered, so all "
                   f"${monthly_save:,.0f}/month could go toward investing.")
        invest_monthly = monthly_save
    elif months is None:
        st.warning("With nothing left over each month, there's no money to build an emergency fund or invest yet.")
        invest_monthly = 0.0
    else:
        st.info(f"🛟 First, build an emergency fund of ${e_target:,.0f} (3 months of expenses). At "
                f"${monthly_save:,.0f}/month that takes about **{months} months**. After that, the full "
                f"${monthly_save:,.0f}/month could go toward investing.")
        invest_monthly = monthly_save

    g1, g2 = st.columns(2)
    years = g1.slider("Years of investing", 1, 40, 10, key="bud_years")
    rate = g2.slider("Assumed yearly return % (illustrative)", 0.0, 12.0, 7.0, 0.5, key="bud_rate")

    if invest_monthly > 0:
        idx = list(range(1, years + 1))
        proj = pd.DataFrame({
            "Cash (0%)": future_value(invest_monthly, 0, years),
            "Savings account (~4%)": future_value(invest_monthly, 4, years),
            f"Invested ({rate:g}%, assumed)": future_value(invest_monthly, rate, years),
        }, index=pd.Index(idx, name="Year"))
        final = proj.iloc[-1]
        contributed = invest_monthly * 12 * years
        k1, k2, k3 = st.columns(3)
        k1.metric("You put in", f"${contributed:,.0f}")
        k2.metric(f"Invested at {rate:g}% after {years} yrs", f"${final.iloc[2]:,.0f}",
                  f"+${final.iloc[2] - contributed:,.0f} growth", delta_color="off")
        k3.metric("Same money in cash", f"${final.iloc[0]:,.0f}")
        st.line_chart(proj, y_label="Balance ($)")
        st.caption("Assumes the same amount every month and a steady return. Real markets go up and "
                   "down, sometimes a lot, and returns are never guaranteed. This illustrates the power "
                   "of compounding; it is not a prediction or financial advice.")

    st.info("Educational tool only, not financial advice. Every decision is yours.")