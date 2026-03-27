# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta

from src.pipeline import analyze_transaction, store
from src.insights import business_summary as generate_business_summary
from src.chatbot_engine import build_financial_context, append_forecast_insights_to_response


# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="FinSense AI",
    description="AI-powered financial assistant for small businesses",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load Models on Startup ───────────────────────────────────
@app.on_event("startup")
async def startup_event():
    store.load(models_dir='models')
    print("FinSense AI ready.")


# ── Load Transaction History ────────────────────────────────
_history_df: Optional[pd.DataFrame] = None
_cashflow_df: Optional[pd.DataFrame] = None


def get_history() -> Optional[pd.DataFrame]:
    global _history_df
    if _history_df is None and os.path.exists('data/transactions.csv'):
        _history_df = pd.read_csv('data/transactions.csv')
        _history_df['date'] = pd.to_datetime(_history_df['date'])
    return _history_df


def get_cashflow() -> Optional[pd.DataFrame]:
    global _cashflow_df
    if _cashflow_df is not None:
        return _cashflow_df

    paths = [
        os.path.join('data', 'daily_cashflow_by_business.csv'),
        os.path.join('data', 'daily_cashflow.csv'),
    ]

    for path in paths:
        if os.path.exists(path):
            _cashflow_df = pd.read_csv(path)
            if 'date' in _cashflow_df.columns:
                _cashflow_df['date'] = pd.to_datetime(_cashflow_df['date'], errors='coerce')
            return _cashflow_df

    return None


def compute_forecast(business_id: str, days: int = 7, current_balance: Optional[float] = None) -> Optional[dict]:
    if days < 1 or days > 30:
        raise ValueError('days must be between 1 and 30')

    cf = get_cashflow()
    if cf is None or cf.empty:
        return None

    biz_id = business_id.strip().upper()
    if 'business_id' not in cf.columns:
        return None

    bdf = cf[cf['business_id'].astype(str).str.upper() == biz_id].copy()
    if bdf.empty:
        return None

    bdf = bdf.sort_values('date').reset_index(drop=True)
    if 'net_cashflow' not in bdf.columns:
        return None

    net_series = bdf['net_cashflow'].astype(float)
    recent_30 = net_series.tail(30)

    window = min(14, len(recent_30))
    if window <= 0:
        return None

    weights = np.exp(np.linspace(-1, 0, window))
    weights = weights / weights.sum()
    recent = recent_30.tail(window).values

    base_daily_net = float(np.sum(recent * weights))

    std_daily = float(net_series.std()) if len(net_series) > 3 else abs(base_daily_net) * 0.15
    std_daily = min(std_daily, abs(base_daily_net) * 0.3) if abs(base_daily_net) > 0 else 1.0

    if current_balance is not None:
        running_balance = float(current_balance)
    elif 'cumulative_balance' in bdf.columns and not bdf['cumulative_balance'].dropna().empty:
        running_balance = float(bdf['cumulative_balance'].iloc[-1])
    else:
        running_balance = 0.0

    np.random.seed(42)
    daily_forecast = []
    zero_date = None

    for i in range(days):
        noise = np.random.normal(0, std_daily * 0.1)
        daily_net = base_daily_net + noise
        running_balance += daily_net

        balance = round(float(running_balance), 2)
        if zero_date is None and balance <= 0:
            last_date = bdf['date'].max() if 'date' in bdf.columns else datetime.now()
            zero_date = (last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d')

        daily_forecast.append({
            'day': i + 1,
            'net_cashflow': round(float(daily_net), 2),
            'balance': balance,
        })

    min_balance = min(d['balance'] for d in daily_forecast)
    final_balance = daily_forecast[-1]['balance']
    trend = 'growing' if base_daily_net > 0 else 'declining'

    alert = None
    if min_balance < 0:
        low_day = next(d['day'] for d in daily_forecast if d['balance'] == min_balance)
        alert = f"Balance turns negative by Day {low_day} (₹{min_balance:,.0f}) — urgent action needed"
    elif min_balance < 10000:
        low_day = next(d['day'] for d in daily_forecast if d['balance'] == min_balance)
        alert = f"Balance may drop to ₹{min_balance:,.0f} by Day {low_day} — consider reducing expenses"

    runway_days = None
    if base_daily_net < 0 and base_daily_net != 0:
        runway_days = int(abs(running_balance / base_daily_net)) if base_daily_net != 0 else None

    mom_pct = 0.0
    if len(bdf) >= 60:
        last_30 = bdf.tail(30)['net_cashflow'].sum()
        prev_30 = bdf.tail(60).head(30)['net_cashflow'].sum()
        mom_pct = round((last_30 - prev_30) / abs(prev_30) * 100, 2) if prev_30 != 0 else 0.0

    summary = f"Projected balance after {days} days: ₹{int(final_balance):,} ({trend})"
    insights = [f"Cashflow change vs prior month: {mom_pct:.1f}%"] if len(bdf) >= 60 else []

    return {
        'business_id': biz_id,
        'days': days,
        'starting_balance': round(float(running_balance - sum(d['net_cashflow'] for d in daily_forecast)), 2),
        'daily': daily_forecast,
        'trend': trend,
        'min_balance': round(min_balance, 2),
        'final_balance': round(final_balance, 2),
        'alert': alert,
        'runway_days': runway_days,
        'zero_date': zero_date,
        'mom_pct': mom_pct,
        'insights': insights,
        'summary': summary,
    }


@app.get('/forecast/{business_id}')
def forecast_endpoint(business_id: str, days: int = 7, balance: Optional[float] = None):
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail='days must be between 1 and 30')

    try:
        result = compute_forecast(business_id, days=days, current_balance=balance)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not result:
        raise HTTPException(status_code=404, detail=f'Cashflow data not found for {business_id}')

    return {
        'success': True,
        'data': result,
    }


# ── Request Schemas ─────────────────────────────────────────
class TransactionRequest(BaseModel):
    description: str
    amount: float
    user_id: str
    business_id: str
    current_balance: Optional[float] = None
    forecast_days: int = 7


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "FinSense AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": store.loaded
    }


@app.post("/analyze-transaction")
def analyze(req: TransactionRequest):
    try:
        history = get_history()

        result = analyze_transaction(
            description=req.description,
            amount=req.amount,
            user_id=req.user_id,
            business_id=req.business_id,
            transaction_history=history,
            forecast_days=req.forecast_days,
            current_balance=req.current_balance,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    history = get_history()
    results = []

    for tx in req.transactions:
        try:
            r = analyze_transaction(
                description=tx.description,
                amount=tx.amount,
                user_id=tx.user_id,
                business_id=tx.business_id,
                transaction_history=history,
                forecast_days=tx.forecast_days,
                current_balance=tx.current_balance,
            )
            results.append({"status": "ok", "result": r})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {"results": results, "count": len(results)}


class ChatRequest(BaseModel):
    user_id: str
    business_id: str
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    category: Optional[str] = None
    monthly_revenue: Optional[float] = None
    message: str
    description: Optional[str] = None
    amount: Optional[float] = None


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    hf_api_key = os.getenv('HF_API_KEY')
    if not hf_api_key:
        raise HTTPException(status_code=500, detail='Missing HF_API_KEY environment variable')

    biz_id = req.business_id.strip().upper()
    if not biz_id:
        raise HTTPException(status_code=400, detail='business_id is required')

    history = get_history()
    if history is None:
        raise HTTPException(status_code=503, detail='Transaction data not found')

    # New business cold start (no 404). We'll onboard if no history exists for biz.
    all_biz = history['business_id'].astype(str).str.upper().unique().tolist() if not history.empty else []

    ml_result = None
    if req.description and req.amount is not None:
        ml_result = analyze_transaction(
            description=req.description,
            amount=req.amount,
            user_id=req.user_id,
            business_id=biz_id,
            transaction_history=history,
        )

    summary, context_str = build_financial_context(
        biz_id,
        user_id=req.user_id,
        business_type=req.business_type,
        category=req.category,
        monthly_revenue=req.monthly_revenue,
        ml_result=ml_result,
    )
    print("CONTEXT BUILT for", biz_id)

    # Cold start with profile-based welcome (no onboarding questions)
    if summary.get('mode') == 'cold_start':
        profile_message = summary.get('profile_message', "Here’s what I can infer so far based on your business profile...")
        profile_message += "\n\n⚠️ These are estimated insights based on similar businesses."
        profile_message += "\n\n👉 Try: 'Paid 5000 for raw material'"

        return {
            'success': True,
            'business_id': biz_id,
            'reply': profile_message,
            'summary': summary,
            'ml_result': ml_result,
        }

    # QUICK TEMP FALLBACK (for offline/demo):
    if os.getenv('DEBUG_CHAT_NO_LLM', '0') == '1':
        return {
            'success': True,
            'business_id': biz_id,
            'reply': 'Chat working without LLM',
            'summary': summary,
            'ml_result': ml_result,
        }

    hf_url = 'https://router.huggingface.co/v1/chat/completions'
    hf_model = 'meta-llama/Llama-3.1-8B-Instruct:cerebras'

    system_prompt = f"""You are FinBot, an expert AI financial assistant for SMBs.\n"""
    user_prompt = f"""User message: {req.message}\n\nFinancial context:\n{context_str}"""

    payload = {
        'model': hf_model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'max_tokens': 512,
        'temperature': 0.7,
    }

    try:
        response = requests.post(
            hf_url,
            headers={
                'Authorization': f'Bearer {hf_api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=60,
        )

        print('HF STATUS:', response.status_code)
        print('HF RESPONSE:', response.text)

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HF API error {response.status_code}: {response.text}")

        data = response.json()

        if 'choices' not in data or not data['choices']:
            raise HTTPException(status_code=502, detail={'message': 'Invalid response from HF API', 'body': data})

        reply = data['choices'][0].get('message', {}).get('content', '').strip()

        if summary.get('mode') == 'hybrid':
            reply += "\n\n📊 You're getting started. Add more transactions to unlock deeper insights and forecasting."

        reply = append_forecast_insights_to_response(reply, summary.get('forecast', {}))

        financial_summary = summary.get('financial_summary', {})
        if financial_summary:
            total_income = financial_summary.get('total_income', 0)
            total_expense = financial_summary.get('total_expense', 0)
            reply += f"\n\n💰 Total Income: ₹{int(total_income):,}"
            reply += f"\n💸 Total Expense: ₹{int(total_expense):,}"

            cat_breakdown = financial_summary.get('category_breakdown', {})
            if cat_breakdown:
                top_category = max(cat_breakdown, key=cat_breakdown.get)
                reply += f"\n\n📊 Highest spending category: {top_category}"

        # Ensure structure with mandatory narrative sections when needed
        if 'Here’s what I can infer so far' in context_str:
            reply = (
                "What happened: We don’t have enough historical data yet.\n"
                "What it means: We’ll start with a simplified view and improve as you add more transactions.\n"
                "Insight: " + summary.get('onboarding_insight', "Here’s what I can infer so far based on your inputs...") + "\n"
                "Suggestion: Start by adding your first transaction and a few business details."
            )

        return {
            'success': True,
            'business_id': biz_id,
            'reply': reply,
            'summary': summary,
            'ml_result': ml_result,
        }

    except Exception as e:
        print('ERROR:', str(e))
        return {
            'success': False,
            'error': str(e),
            'business_id': biz_id,
            'reply': 'Your top expense is food. Try reducing it by 15%.',
        }


@app.get("/business-summary/{business_id}")
def get_business_summary(business_id: str):
    try:
        df = get_history()

        if df is None:
            raise HTTPException(status_code=503, detail="Transaction data not found")

        result = generate_business_summary(df, business_id)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))