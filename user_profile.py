"""
Personal investor profile, stored ONLY in a local file (user_profile.json) on this machine.
Nothing here is ever sent to Gemma or any API: all calculations are plain Python.
user_profile.json is in .gitignore so it never gets pushed to GitHub.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_profile.json")

RISK_QUESTIONS = [
    ("If your investments dropped 20% in one month, you would…",
     ["Sell to stop the losses", "Wait it out", "Buy more while it's cheap"]),
    ("When will you need most of this money?",
     ["Within 2 years", "In 2–7 years", "7+ years from now"]),
    ("How much investing experience do you have?",
     ["I'm brand new", "Some: I've bought a few stocks or funds", "A lot: I follow markets closely"]),
]

DEFAULTS = {
    "name": "", "monthly_income": 0.0, "monthly_expenses": 0.0, "current_savings": 0.0,
    "goal_name": "Emergency fund", "goal_amount": 0.0, "goal_date": None,
    "risk_answers": [1, 1, 1], "max_pct_per_stock": 10, "favorite_tickers": "",
}


def load() -> dict | None:
    try:
        with open(PATH) as f:
            return {**DEFAULTS, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(profile: dict) -> None:
    with open(PATH, "w") as f:
        json.dump(profile, f, indent=2, default=str)


def delete() -> None:
    if os.path.exists(PATH):
        os.remove(PATH)


def risk_level(answers: list[int]) -> tuple[str, int]:
    """Each answer is 0/1/2 -> total 0..6."""
    total = sum(answers)
    if total <= 2:
        return "Conservative", total
    if total <= 4:
        return "Moderate", total
    return "Aggressive", total


STYLE_MATCH = {
    "Conservative": ["Income"],
    "Moderate": ["Value", "Income"],
    "Aggressive": ["Growth", "Momentum"],
}


def summarize(p: dict, today: date | None = None) -> dict:
    """All the numbers the profile page shows. Pure math, no AI."""
    today = today or date.today()
    income, expenses = float(p["monthly_income"]), float(p["monthly_expenses"])
    savings, goal = float(p["current_savings"]), float(p["goal_amount"])
    surplus = income - expenses
    savings_rate = surplus / income * 100 if income > 0 else 0.0

    emergency_target = expenses * 3                      # common rule of thumb: 3-6 months of expenses
    emergency_ok = expenses > 0 and savings >= emergency_target

    remaining = max(goal - savings, 0.0)
    months_at_current_pace = (math.ceil(remaining / surplus) if surplus > 0 else None) if remaining > 0 else 0

    needed_per_month = None
    months_left = None
    if p.get("goal_date") and remaining > 0:
        gd = date.fromisoformat(str(p["goal_date"]))
        months_left = max((gd.year - today.year) * 12 + (gd.month - today.month), 1)
        needed_per_month = remaining / months_left

    level, risk_total = risk_level(p["risk_answers"])
    # Money beyond the emergency fund is what's reasonable to consider investing
    investable = max(savings - emergency_target, 0.0)
    max_per_stock = investable * p["max_pct_per_stock"] / 100

    return {
        "surplus": surplus, "savings_rate": savings_rate,
        "emergency_target": emergency_target, "emergency_ok": emergency_ok,
        "remaining": remaining, "months_at_current_pace": months_at_current_pace,
        "months_left": months_left, "needed_per_month": needed_per_month,
        "on_track": needed_per_month is None or (surplus > 0 and surplus >= needed_per_month),
        "risk_level": level, "risk_total": risk_total, "styles": STYLE_MATCH[level],
        "investable": investable, "max_per_stock": max_per_stock,
    }