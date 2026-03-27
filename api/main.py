# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import os
import requests

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

def get_history() -> Optional[pd.DataFrame]:
    global _history_df
    if _history_df is None and os.path.exists('data/transactions.csv'):
        _history_df = pd.read_csv('data/transactions.csv')
        _history_df['date'] = pd.to_datetime(_history_df['date'])
    return _history_df


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