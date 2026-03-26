import os
from typing import Dict, Optional, Tuple

import pandas as pd

from src.pipeline import analyze_transaction

DATA_DIR = os.getenv('DATA_DIR', 'data')
TRANSACTIONS_CSV = os.path.join(DATA_DIR, 'transactions.csv')
DAILY_CASHFLOW_CSV = os.path.join(DATA_DIR, 'daily_cashflow.csv')

_all_transactions = None
_all_cashflow = None
_available_biz = None
_industry_benchmark = None


def _load_data() -> None:
    global _all_transactions, _all_cashflow, _available_biz, _industry_benchmark

    if _all_transactions is None:
        _all_transactions = pd.read_csv(TRANSACTIONS_CSV)
        if 'date' in _all_transactions.columns:
            _all_transactions['date'] = pd.to_datetime(_all_transactions['date'])

    if _all_cashflow is None and os.path.exists(DAILY_CASHFLOW_CSV):
        _all_cashflow = pd.read_csv(DAILY_CASHFLOW_CSV)
        if 'date' in _all_cashflow.columns:
            _all_cashflow['date'] = pd.to_datetime(_all_cashflow['date'])
    elif _all_cashflow is None:
        _all_cashflow = pd.DataFrame()

    if _available_biz is None and _all_transactions is not None:
        _available_biz = sorted(_all_transactions['business_id'].astype(str).str.upper().unique().tolist())

    # Avoid recursion by not calling build_industry_benchmark here.
    if _industry_benchmark is None:
        _industry_benchmark = {}


def detect_anomalies(expenses_df: pd.DataFrame) -> list:
    """Find transactions that are 3x above their category average."""
    if expenses_df is None or expenses_df.empty:
        return []

    anomalies = []
    cat_avg = expenses_df.groupby('category')['amount'].mean().to_dict()

    for _, row in expenses_df.iterrows():
        avg = cat_avg.get(row['category'], 0)
        if avg > 0 and row['amount'] > 3 * avg:
            anomalies.append({
                'date': row.get('date'),
                'description': row.get('description'),
                'amount': round(float(row.get('amount', 0)), 2),
                'category': row.get('category'),
                'normal_avg': round(float(avg), 2),
                'times_higher': round(float(row['amount'] / avg), 1),
            })

    anomalies.sort(key=lambda x: x['times_higher'], reverse=True)
    return anomalies[:5]


def build_industry_benchmark() -> dict:
    global _industry_benchmark

    if _industry_benchmark:
        return _industry_benchmark

    _load_data()

    benchmark = {}
    if _all_transactions is None or _all_transactions.empty:
        _industry_benchmark = {
            'per_business': {},
            'avg_profit_margin': 0,
            'avg_total_income': 0,
            'avg_total_expense': 0,
            'avg_category_ratios': {},
            'total_businesses': 0,
        }
        return _industry_benchmark

    for biz in _available_biz:
        df = _all_transactions[_all_transactions['business_id'].astype(str).str.upper() == biz]
        expenses = df[df['type'] == 'expense']
        income = df[df['type'] == 'income']

        total_expense = float(expenses['amount'].sum())
        total_income = float(income['amount'].sum())

        if total_income <= 0:
            continue

        cat_ratios = (expenses.groupby('category')['amount'].sum() / total_expense * 100).to_dict() if total_expense > 0 else {}
        profit_margin = round((total_income - total_expense) / total_income * 100, 1)

        benchmark[biz] = {
            'total_income': total_income,
            'total_expense': total_expense,
            'net_profit': total_income - total_expense,
            'profit_margin': profit_margin,
            'cat_ratios': {k: round(float(v), 1) for k, v in cat_ratios.items()},
        }

    all_margins = [v['profit_margin'] for v in benchmark.values()] if benchmark else [0]
    all_incomes = [v['total_income'] for v in benchmark.values()] if benchmark else [0]
    all_expenses = [v['total_expense'] for v in benchmark.values()] if benchmark else [0]

    all_cats = {}
    for v in benchmark.values():
        for cat, ratio in v.get('cat_ratios', {}).items():
            all_cats.setdefault(cat, []).append(ratio)

    avg_cat_ratios = {cat: round(sum(vals) / len(vals), 1) for cat, vals in all_cats.items()} if all_cats else {}

    _industry_benchmark = {
        'per_business': benchmark,
        'avg_profit_margin': round(sum(all_margins) / len(all_margins), 1) if all_margins else 0,
        'avg_total_income': round(sum(all_incomes) / len(all_incomes), 2) if all_incomes else 0,
        'avg_total_expense': round(sum(all_expenses) / len(all_expenses), 2) if all_expenses else 0,
        'avg_category_ratios': avg_cat_ratios,
        'total_businesses': len(benchmark),
    }
    return _industry_benchmark


def build_comparison_context(biz_id: str) -> str:
    _load_data()
    global _industry_benchmark
    if not _industry_benchmark:
        _industry_benchmark = build_industry_benchmark()

    biz_id = biz_id.strip().upper()

    if _industry_benchmark is None or biz_id not in _industry_benchmark['per_business']:
        return "No comparison data available."

    biz_data = _industry_benchmark['per_business'][biz_id]
    avg_margin = _industry_benchmark['avg_profit_margin']
    avg_income = _industry_benchmark['avg_total_income']
    avg_expense = _industry_benchmark['avg_total_expense']
    avg_cats = _industry_benchmark['avg_category_ratios']

    # Rank
    all_margins = sorted(
        [(b, d['profit_margin']) for b, d in _industry_benchmark['per_business'].items()],
        key=lambda x: x[1], reverse=True
    )
    rank = next((i + 1 for i, (b, _) in enumerate(all_margins) if b == biz_id), None)

    cat_comparison = []
    for cat, biz_ratio in biz_data.get('cat_ratios', {}).items():
        ind_ratio = avg_cats.get(cat, 0)
        diff = round(biz_ratio - ind_ratio, 1)
        status = 'OVERSPENDING' if diff > 5 else ('🟢 EFFICIENT' if diff < -5 else '🟡 NORMAL')
        cat_comparison.append(
            f"  - {cat}: {biz_ratio:.1f}% vs industry avg {ind_ratio:.1f}% → {'+' if diff>0 else ''}{diff}% {status}"
        )

    comparison_str = (
        f"=== PEER COMPARISON FOR {biz_id} ===\n"
        f"Profit Margin    : {biz_data['profit_margin']}% vs industry avg {avg_margin}%\n"
        f"Total Income     : ₹{biz_data['total_income']:,.0f} vs industry avg ₹{avg_income:,.0f}\n"
        f"Total Expense    : ₹{biz_data['total_expense']:,.0f} vs industry avg ₹{avg_expense:,.0f}\n"
        f"Industry Rank    : #{rank} out of {_industry_benchmark['total_businesses']} businesses\n"
        f"Expense Category Breakdown vs Industry:\n"
        f"{chr(10).join(cat_comparison)}\n"
    )

    return comparison_str


def build_financial_context(business_id: str, ml_result: Optional[dict] = None) -> Tuple[dict, str]:
    _load_data()
    biz_id = business_id.strip().upper()

    if _all_transactions is None or _all_transactions.empty:
        raise ValueError('No transaction data loaded')

    df = _all_transactions[_all_transactions['business_id'].astype(str).str.upper() == biz_id]
    if df.empty:
        raise ValueError(f"Business '{biz_id}' not found")

    expenses_df = df[df['type'] == 'expense']
    income_df = df[df['type'] == 'income']

    total_income = float(income_df['amount'].sum())
    total_expense = float(expenses_df['amount'].sum())
    net_profit = total_income - total_expense
    profit_margin = round((net_profit / total_income * 100), 1) if total_income > 0 else 0

    category_breakdown = (
        expenses_df.groupby('category')['amount'].sum().sort_values(ascending=False).to_dict()
    )

    anomalies = detect_anomalies(expenses_df)
    anomaly_summary = 'No anomalies detected.'
    if anomalies:
        anomaly_summary = '\n'.join([
            f"{a['date']} | {a['description']} | ₹{a['amount']:,.0f} ({a['times_higher']}x normal avg ₹{a['normal_avg']:,.0f})"
            for a in anomalies
        ])

    comparison = build_comparison_context(biz_id)

    ml_insights_text = ''
    if ml_result:
        insights = ml_result.get('insights')
        anomaly_flag = ml_result.get('anomaly', {}).get('is_anomaly')
        ml_insights_text = "\n=== ML INSIGHTS ===\n"
        ml_insights_text += f"Predicted Category: {ml_result.get('category')}\n"
        ml_insights_text += f"Anomaly Flag: {anomaly_flag}\n"
        if insights:
            try:
                ml_insights_text += "Insights:\n"
                for i in insights:
                    ml_insights_text += f" - {i.get('message', i)}\n"
            except Exception:
                ml_insights_text += str(insights) + "\n"

    context_str = (
        f"=== FINANCIAL CONTEXT FOR {biz_id} ===\n"
        f"Total Income : ₹{total_income:,.2f}\n"
        f"Total Expense: ₹{total_expense:,.2f}\n"
        f"Net Profit   : ₹{net_profit:,.2f}\n"
        f"Profit Margin: {profit_margin}%\n\n"
        f"Expense Category Breakdown:\n"
        + '\n'.join([f"  - {cat}: ₹{amt:,.2f}" for cat, amt in category_breakdown.items()])
        + "\n\n"
        f"Anomaly Summary:\n{anomaly_summary}\n\n"
        f"Peer Comparison:\n{comparison}\n"
        f"{ml_insights_text}"
    )

    summary_dict = {
        'business_id': biz_id,
        'total_income': round(total_income, 2),
        'total_expense': round(total_expense, 2),
        'net_profit': round(net_profit, 2),
        'profit_margin': profit_margin,
        'category_breakdown': {k: round(float(v), 2) for k, v in category_breakdown.items()},
        'anomalies': anomalies,
        'anomaly_summary': anomaly_summary,
        'peer_comparison': comparison,
        'ml_result': ml_result,
    }

    # Add insights summary from ml_result
    if ml_result and ml_result.get('insights'):
        summary_dict['ml_insights'] = ml_result.get('insights')

    return summary_dict, context_str
