# api/main.py - Clean Working FastAPI Code

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

user_sessions = {}

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

# ── Load Transaction History ───────────────────────────────
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

    path = os.path.join('data', 'daily_cashflow_by_business.csv')

    if os.path.exists(path):
        _cashflow_df = pd.read_csv(path)
        if 'date' in _cashflow_df.columns:
            _cashflow_df['date'] = pd.to_datetime(_cashflow_df['date'], errors='coerce')
        required_cols = {'business_id', 'date', 'net_cashflow'}
        if not required_cols.issubset(set(_cashflow_df.columns)):
            raise ValueError("daily_cashflow_by_business.csv missing required columns")
        return _cashflow_df
    else:
        raise FileNotFoundError("daily_cashflow_by_business.csv is required")

# ── Request Schemas ─────────────────────────────────────────
class TransactionRequest(BaseModel):
    description: str = Field(..., description="Transaction description")
    amount: float = Field(..., description="Transaction amount in INR")
    user_id: str = Field(..., description="User identifier")
    business_id: str = Field(..., description="Business identifier")
    income: Optional[float] = Field(None, description="Income amount if applicable")
    expense: Optional[float] = Field(None, description="Expense amount if applicable")
    current_balance: Optional[float] = Field(None, description="Current account balance")
    forecast_days: int = Field(7, description="Days to forecast")

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
def analyze_transaction_endpoint(req: TransactionRequest):
    """
    Analyze a single transaction with ML-powered insights.

    Request body should contain:
    - description: Transaction description (e.g., "bought groceries")
    - amount: Transaction amount (e.g., 500.0)
    - user_id: User identifier (e.g., "U001")
    - business_id: Business identifier (e.g., "BIZ_001")
    - current_balance: Optional current balance
    - forecast_days: Days to forecast (default: 7)

    Returns dynamic analysis based on actual input values.
    """
    try:
        # Ensure models are loaded
        if not store.loaded:
            store.load()

        # Load transaction history
        history = get_history()

        # Run ML analysis pipeline
        result = analyze_transaction(
            description=req.description,
            amount=req.amount,
            user_id=req.user_id,
            business_id=req.business_id,
            transaction_history=history,
            forecast_days=req.forecast_days,
            current_balance=req.current_balance,
            income=req.income,
            expense=req.expense,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    """Analyze multiple transactions in batch."""
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
                income=tx.income,
                expense=tx.expense,
            )
            results.append({"status": "success", "result": r})
        except Exception as e:
            results.append({"status": "error", "error": str(e), "transaction": tx.dict()})

    return {"results": results, "count": len(results)}

# Example usage for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)