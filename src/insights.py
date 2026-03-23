# src/insights.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def _safe_pct(new, old):
    if old == 0:
        return None
    pct = (new - old) / old * 100

    if pct > 200:
        pct = 200
    elif pct < -200:
        pct = -200

    return round(pct, 1)


def week_over_week(df, user_id, category=None, ref_date=None):
    if ref_date is None:
        ref_date = pd.Timestamp(df['date'].max())

    this_week_start = ref_date - timedelta(days=7)
    last_week_start = ref_date - timedelta(days=14)

    base = df[(df['user_id'] == user_id) & (df['type'] == 'expense')]
    if category:
        base = base[base['category'] == category]

    this_week = base[base['date'] >= this_week_start]['amount'].sum()
    last_week = base[
        (base['date'] >= last_week_start) &
        (base['date'] < this_week_start)
    ]['amount'].sum()

    pct = _safe_pct(this_week, last_week)
    if pct is None:
        return None

    direction = 'more' if pct > 0 else 'less'

    if abs(pct) > 100:
        level = "significantly"
    elif abs(pct) > 30:
        level = "noticeably"
    else:
        level = "slightly"

    cat_str = f" on {category}" if category else ""

    message = (
        f"You spent {level} {direction}{cat_str} compared to last week "
        f"(₹{this_week:,.0f} vs ₹{last_week:,.0f}, change: {abs(pct):.1f}%)"
    )

    return {
        "type": "week_over_week",
        "this_week": float(this_week),
        "last_week": float(last_week),
        "pct_change": pct,
        "direction": direction,
        "message": message,
        "severity": "warning" if pct > 20 else ("good" if pct < -10 else "info"),
    }


def top_category_insight(df, business_id, ref_date=None):
    if ref_date is None:
        ref_date = pd.Timestamp(df['date'].max())

    month_data = df[
        (df['business_id'] == business_id) &
        (df['type'] == 'expense') &
        (df['date'].dt.month == ref_date.month) &
        (df['date'].dt.year == ref_date.year)
    ]

    if month_data.empty:
        return None

    by_cat = month_data.groupby('category')['amount'].sum()
    top_cat = by_cat.idxmax()
    top_amt = by_cat.max()
    total_exp = by_cat.sum()

    pct = round(top_amt / total_exp * 100, 1)
    clean_cat = top_cat.replace("_", " ").title()

    return {
        "type": "top_category",
        "category": clean_cat,
        "amount": float(top_amt),
        "percentage": pct,
        "message": f"{clean_cat} is your highest expense at ₹{top_amt:,.0f} ({pct}%)",
        "severity": "warning" if pct > 40 else "info",
    }


def savings_rate_insight(df, business_id, target_rate=20.0, ref_date=None):
    if ref_date is None:
        ref_date = pd.Timestamp(df['date'].max())

    month_data = df[
        (df['business_id'] == business_id) &
        (df['date'].dt.month == ref_date.month) &
        (df['date'].dt.year == ref_date.year)
    ]

    if month_data.empty:
        return None

    income = month_data[month_data['type'] == 'income']['amount'].sum()
    expense = month_data[month_data['type'] == 'expense']['amount'].sum()

    if income == 0:
        return None

    sav_rate = round((income - expense) / income * 100, 1)

    if sav_rate >= target_rate:
        status = "On track"
        severity = "good"
        message = f"Savings rate is {sav_rate}% — good job!"
    else:
        status = "Below target"
        severity = "warning"
        message = f"Savings rate is {sav_rate}% — reduce expenses."

    return {
        "type": "savings_rate",
        "rate": sav_rate,
        "target": target_rate,
        "income": float(income),
        "expense": float(expense),
        "message": message,
        "severity": severity,
    }


def personal_daily_avg_insight(df, user_id, amount, category):
    user_exp = df[
        (df['user_id'] == user_id) &
        (df['type'] == 'expense') &
        (df['category'] == category)
    ]['amount']

    if len(user_exp) < 5:
        return None

    avg = user_exp.mean()
    ratio = amount / avg if avg > 0 else None

    if ratio is None:
        return None

    return {
        "type": "personal_average",
        "user_avg": float(avg),
        "current_amt": float(amount),
        "ratio": round(ratio, 1),
        "message": f"Your avg {category} spend is ₹{avg:,.0f}, this is {ratio:.1f}x",
        "severity": "warning" if ratio > 2 else "info",
    }


def generate_all_insights(df, business_id, user_id, category, amount, ref_date=None):
    insights = []

    wow = week_over_week(df, user_id, category, ref_date)
    if wow:
        insights.append(wow)

    top_cat = top_category_insight(df, business_id, ref_date)
    if top_cat:
        insights.append(top_cat)

    sav = savings_rate_insight(df, business_id, ref_date=ref_date)
    if sav:
        insights.append(sav)

    personal = personal_daily_avg_insight(df, user_id, amount, category)
    if personal:
        insights.append(personal)

    return insights
def business_summary(df, business_id):
    data = df[df['business_id'] == business_id]

    if data.empty:
        return {"error": f"No data found for {business_id}"}

    income = data[data['type'] == 'income']['amount'].sum()
    expense = data[data['type'] == 'expense']['amount'].sum()
    net = income - expense

    savings_rate = round((net / income) * 100, 1) if income > 0 else 0

    if savings_rate > 20:
        health = "good"
    elif savings_rate >= 0:
        health = "average"
    else:
        health = "poor"

    cat_data = data[data['type'] == 'expense']
    cat_group = cat_data.groupby('category')['amount'].sum().sort_values(ascending=False)

    total_exp = cat_group.sum()

    top_categories = []
    for cat, amt in cat_group.head(3).items():
        top_categories.append({
            "category": cat,
            "amount": float(amt),
            "percentage": round((amt / total_exp) * 100, 1)
        })

    summary = (
        f"Your business is running at a {'profit' if net > 0 else 'loss'} "
        f"of ₹{abs(net):,.0f}. Savings rate is {savings_rate}%."
    )

    recommendation = (
        "Reduce expenses or increase revenue."
        if savings_rate < 0 else
        "Your financial health is stable."
    )

    return {
        "business_id": business_id,
        "total_income": float(income),
        "total_expense": float(expense),
        "net_balance": float(net),
        "savings_rate_pct": savings_rate,
        "financial_health": health,
        "top_categories": top_categories,
        "summary": summary,
        "recommendation": recommendation,
        "total_transactions": len(data)
    }