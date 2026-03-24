# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import os

from src.pipeline import analyze_transaction, store
from src.insights import business_summary as generate_business_summary


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