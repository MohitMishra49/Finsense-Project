# src/pipeline.py
# ────────────────────────────────────────────────────────────
# THE UNIFIED PIPELINE — One input, complete intelligent output
# This is the core of the entire ML system.
# ────────────────────────────────────────────────────────────

import json
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from typing import Optional

from src.preprocess import clean_text
from src.explainer  import explain_prediction, explain_anomaly
from src.insights   import generate_all_insights
from src.forecaster import forecast_cashflow


# ════════════════════════════════════════════════════════════
# MODEL LOADER — loads once, reused for every request
# ════════════════════════════════════════════════════════════
class ModelStore:
    """Loads and caches all models at startup."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, models_dir: str = 'models'):
        if self._loaded:
            return
        print("Loading models into memory...")

        self.cat_model    = joblib.load(f'{models_dir}/category_model.pkl')
        self.vectorizer   = joblib.load(f'{models_dir}/vectorizer.pkl')
        self.anomaly_model= joblib.load(f'{models_dir}/anomaly_model.pkl')
        self.anomaly_scaler=joblib.load(f'{models_dir}/anomaly_scaler.pkl')

        with open(f'{models_dir}/categorizer_meta.json') as f:
            self.cat_meta = json.load(f)

        self.cat_code_map = {}
        if os.path.exists(f'{models_dir}/cat_code_map.json'):
            with open(f'{models_dir}/cat_code_map.json') as f:
                self.cat_code_map = {int(k): v
                                     for k, v in json.load(f).items()}

        self._loaded = True
        print(f"✓ All models loaded | "
              f"Categories: {len(self.cat_meta['classes'])}")

    @property
    def loaded(self):
        return self._loaded


# Singleton instance
store = ModelStore()


# ════════════════════════════════════════════════════════════
# FORECASTING — per-business, derived from real cashflow data
# ════════════════════════════════════════════════════════════
def _forecast_cashflow(
    current_balance: float,
    days: int = 7,
    business_id: str = '',
) -> dict:
    """
    Generate a per-business cash flow forecast.
    Uses src/forecaster.py which reads daily_cashflow_by_business.csv.
    Each business gets its own unique forecast based on its own history.
    """
    try:
        from src.forecaster import forecast_cashflow
        return forecast_cashflow(
            business_id     = business_id or 'UNKNOWN',
            current_balance = current_balance,
            days            = days,
        )
    except Exception:
        # Hard fallback — should never reach here
        daily_forecasts = [
            {'day': i+1, 'net_cashflow': 0, 'balance': round(current_balance, 2)}
            for i in range(days)
        ]
        return {
            'days':          days,
            'daily':         daily_forecasts,
            'min_balance':   round(current_balance, 2),
            'final_balance': round(current_balance, 2),
            'alert':         None,
            'summary':       f"Projected balance after {days} days: ₹{current_balance:,.0f}",
        }


# ════════════════════════════════════════════════════════════
# THE UNIFIED PIPELINE
# ════════════════════════════════════════════════════════════
def analyze_transaction(
    description:         str,
    amount:              float,
    user_id:             str,
    business_id:         str,
    transaction_history: Optional[pd.DataFrame] = None,
    forecast_days:       int = 7,
    current_balance:     Optional[float] = None,
) -> dict:
    """
    THE MAIN PIPELINE.

    Input  : one transaction (text + amount + ids)
    Output : category + confidence + explanation +
             anomaly + insights + forecast

    Args:
        description         : raw transaction text
        amount              : transaction amount in INR
        user_id             : user identifier
        business_id         : business identifier
        transaction_history : DataFrame of past transactions (optional)
        forecast_days       : how many days to forecast
        current_balance     : current account balance (optional)

    Returns:
        dict with all ML outputs combined
    """
    if not store.loaded:
        store.load()

    result = {
        'input': {
            'description': description,
            'amount':      amount,
            'user_id':     user_id,
            'business_id': business_id,
        }
    }

    # ── STEP 1: Categorization + Explainability ─────────────
    cleaned  = clean_text(description)
    vec_inp  = store.vectorizer.transform([cleaned])
    category = store.cat_model.predict(vec_inp)[0]

    explanation = explain_prediction(
        description, category,
        store.cat_model, store.vectorizer, top_n=5,
    )
    result['category']    = category
    result['confidence']  = explanation['confidence']
    result['reason']      = explanation['top_keywords'][:3]
    result['explanation'] = explanation

    # ── STEP 2: Anomaly Detection ────────────────────────────
    # Get category code for isolation forest
    cat_codes = {v: k for k, v in store.cat_code_map.items()} \
                if store.cat_code_map else {}
    cat_code  = cat_codes.get(category, 0)

    log_amt   = np.log1p(amount)
    X_inp     = store.anomaly_scaler.transform([[log_amt, cat_code]])
    iso_pred  = store.anomaly_model.predict(X_inp)[0]   # 1=normal, -1=anomaly
    iso_score = float(store.anomaly_model.score_samples(X_inp)[0])

    # Z-score check using history (more interpretable)
    if transaction_history is not None:
        hist_amounts = transaction_history[
            (transaction_history['user_id']   == user_id) &
            (transaction_history['category']  == category) &
            (transaction_history['type']      == 'expense')
        ]['amount'].tolist()
    else:
        hist_amounts = []

    anomaly_info = explain_anomaly(amount, category, hist_amounts)

    # Combine ISO forest + Z-score
    is_anomaly = (iso_pred == -1) or anomaly_info.get('is_anomaly', False)
    result['anomaly'] = {
        'is_anomaly':      is_anomaly,
        'isolation_score': round(iso_score, 4),
        'z_score':         anomaly_info.get('z_score'),
        'explanation':     anomaly_info['explanation'],
    }

    # ── STEP 3: Smart Insights ──────────────────────────────
    if transaction_history is not None and len(transaction_history) > 10:
        th = transaction_history.copy()
        th['date'] = pd.to_datetime(th['date'])
        insights = generate_all_insights(
            th, business_id, user_id,
            category, amount
        )
    else:
        insights = [{
            'type':     'info',
            'message':  'Add more transactions to unlock personalized insights.',
            'severity': 'info',
        }]
    result['insights'] = insights

    # ── STEP 4: Cash Flow Forecast (per-business) ───────────
    result['forecast'] = forecast_cashflow(business_id, current_balance, forecast_days)




    # ── STEP 5: Summary (top-level for quick display) ───────
    top_insight = insights[0]['message'] if insights else None
    forecast_alert = result['forecast'].get('alert')

    result['summary'] = {
        'category':       category,
        'confidence_pct': explanation['confidence'],
        'top_keywords':   explanation['top_keywords'][:3],
        'is_anomaly':     is_anomaly,
        'anomaly_msg':    anomaly_info['explanation'] if is_anomaly else None,
        'key_insight':    top_insight,
        'forecast_alert': forecast_alert,
    }

    return result
