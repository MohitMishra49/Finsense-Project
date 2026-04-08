# src/ml_pipeline.py
# ══════════════════════════════════════════════════════════════
# Production-ready ML Pipeline for Cashflow Prediction
# Business-specific filtering, forecasting, and insights
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.preprocess import clean_text, rule_based_categorize
from src.forecaster import forecast_cashflow, detect_cashflow_anomalies, get_business_cashflow
from src.insights import generate_all_insights
from src.explainer import explain_prediction, explain_anomaly


class CashflowMLPipeline:
    """
    Production-ready ML pipeline for business-specific cashflow analysis.
    """

    def __init__(self, models_dir: str = 'models', data_dir: str = 'data'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self._load_models()
        self._load_data()

    def _load_models(self):
        """Load all trained models."""
        try:
            self.category_model = joblib.load(f'{self.models_dir}/category_model.pkl')
            self.vectorizer = joblib.load(f'{self.models_dir}/vectorizer.pkl')
            self.anomaly_model = joblib.load(f'{self.models_dir}/anomaly_model.pkl')
            self.anomaly_scaler = joblib.load(f'{self.models_dir}/anomaly_scaler.pkl')

            with open(f'{self.models_dir}/categorizer_meta.json') as f:
                self.category_meta = pd.read_json(f)

            with open(f'{self.models_dir}/cat_code_map.json') as f:
                self.cat_code_map = pd.read_json(f)

            print("✓ All models loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def _load_data(self):
        """Load transaction and cashflow datasets."""
        try:
            self.transactions_df = pd.read_csv(f'{self.data_dir}/transactions.csv')
            self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])

            self.cashflow_df = pd.read_csv(f'{self.data_dir}/daily_cashflow_by_business.csv')
            self.cashflow_df['date'] = pd.to_datetime(self.cashflow_df['date'])

            print("✓ Datasets loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def get_business_data(self, business_id: str) -> Dict[str, pd.DataFrame]:
        """
        Filter all datasets by business_id.
        Returns dict with 'transactions' and 'cashflow' DataFrames.
        """
        biz_id = business_id.strip().upper()

        # Filter transactions
        transactions = self.transactions_df[
            self.transactions_df['business_id'].astype(str).str.upper() == biz_id
        ].copy()

        # Filter cashflow
        cashflow = self.cashflow_df[
            self.cashflow_df['business_id'].astype(str).str.upper() == biz_id
        ].copy()

        return {
            'transactions': transactions,
            'cashflow': cashflow
        }

    def categorize_transaction(self, description: str) -> Dict:
        """
        Categorize transaction using rule-based + ML approach.
        """
        desc = clean_text(description.lower().strip()) if description else ""

        if not desc:
            category = "misc"
            confidence = 0.0
        else:
            # Try rule-based first
            category = rule_based_categorize(desc)
            confidence = 0.8 if category != "misc" else 0.0

            # Fallback to ML if rule-based fails
            if category == "misc":
                try:
                    X = self.vectorizer.transform([desc])
                    category = self.category_model.predict(X)[0]
                    proba = self.category_model.predict_proba(X)[0]
                    confidence = float(proba.max())
                except:
                    category = "misc"
                    confidence = 0.0

        explanation = explain_prediction(
            description, category,
            self.category_model, self.vectorizer
        )

        return {
            'category': category,
            'confidence': round(confidence * 100, 1),
            'explanation': explanation
        }

    def detect_anomaly(self, amount: float, category: str, business_id: str) -> Dict:
        """
        Detect anomalies using Isolation Forest + business-specific Z-score.
        """
        # Global anomaly detection
        cat_code = self.cat_code_map.get(category, 0)
        log_amt = np.log1p(amount)
        X_input = self.anomaly_scaler.transform([[log_amt, cat_code]])
        iso_pred = self.anomaly_model.predict(X_input)[0]
        iso_score = float(self.anomaly_model.score_samples(X_input)[0])

        # Business-specific Z-score
        biz_data = self.get_business_data(business_id)
        hist_amounts = biz_data['transactions'][
            (biz_data['transactions']['category'] == category) &
            (biz_data['transactions']['type'] == 'expense')
        ]['amount'].tolist()

        z_score_analysis = explain_anomaly(amount, category, hist_amounts)

        is_anomaly = (iso_pred == -1) or z_score_analysis.get('is_anomaly', False)

        return {
            'is_anomaly': bool(is_anomaly),
            'isolation_score': round(iso_score, 4),
            'z_score': z_score_analysis.get('z_score'),
            'explanation': z_score_analysis['explanation']
        }

    def generate_forecast(self, business_id: str, days: int = 7,
                         current_balance: Optional[float] = None) -> Dict:
        """
        Generate business-specific cashflow forecast.
        """
        return forecast_cashflow(
            business_id=business_id,
            current_balance=current_balance,
            days=days
        )

    def get_insights(self, business_id: str, user_id: str,
                    category: str, amount: float) -> List[Dict]:
        """
        Generate business-specific insights.
        """
        biz_data = self.get_business_data(business_id)
        transactions = biz_data['transactions']

        if transactions.empty:
            return []

        # Filter by business and ensure date column
        transactions = transactions.copy()
        transactions['date'] = pd.to_datetime(transactions['date'])

        return generate_all_insights(
            transactions, business_id, user_id, category, amount
        )

    def analyze_transaction(self, description: str, amount: float,
                          user_id: str, business_id: str,
                          current_balance: Optional[float] = None,
                          forecast_days: int = 7) -> Dict:
        """
        Complete ML analysis pipeline for a single transaction.
        """
        # 1. Categorization
        category_result = self.categorize_transaction(description)

        # 2. Anomaly Detection
        anomaly_result = self.detect_anomaly(amount, category_result['category'], business_id)

        # 3. Forecast
        forecast_result = self.generate_forecast(business_id, forecast_days, current_balance)

        # 4. Insights
        insights = self.get_insights(business_id, user_id, category_result['category'], amount)

        # 5. Financial Summary
        biz_data = self.get_business_data(business_id)
        transactions = biz_data['transactions']

        if not transactions.empty:
            income_total = transactions[transactions['type'] == 'income']['amount'].sum()
            expense_total = transactions[transactions['type'] == 'expense']['amount'].sum()
            net_profit = income_total - expense_total

            category_breakdown = transactions[
                transactions['type'] == 'expense'
            ].groupby('category')['amount'].sum().to_dict()
        else:
            income_total = expense_total = net_profit = 0
            category_breakdown = {}

        # 6. Cashflow Anomalies
        cashflow_anomalies = detect_cashflow_anomalies(business_id)

        return {
            'business_id': business_id.upper(),
            'input': {
                'description': description,
                'amount': amount,
                'user_id': user_id,
                'current_balance': current_balance
            },
            'category': category_result,
            'anomaly': anomaly_result,
            'forecast': forecast_result,
            'insights': insights,
            'financial_summary': {
                'total_income': float(income_total),
                'total_expense': float(expense_total),
                'net_profit': float(net_profit),
                'category_breakdown': category_breakdown
            },
            'cashflow_anomalies': cashflow_anomalies,
            'timestamp': datetime.now().isoformat()
        }


# Example Usage
if __name__ == '__main__':
    # Initialize pipeline
    pipeline = CashflowMLPipeline()

    # Example transaction analysis
    result = pipeline.analyze_transaction(
        description="bought groceries from supermarket",
        amount=2500.0,
        user_id="U001",
        business_id="BIZ_001",
        current_balance=100000.0,
        forecast_days=7
    )

    print("=== Transaction Analysis Result ===")
    print(f"Business: {result['business_id']}")
    print(f"Category: {result['category']['category']} ({result['category']['confidence']}%)")
    print(f"Anomaly: {result['anomaly']['is_anomaly']}")
    print(f"Forecast Start Balance: ₹{result['forecast']['start_balance']:,.0f}")
    print(f"Trend: {result['forecast']['trend']}")
    print(f"Insights: {len(result['insights'])} generated")
    print(f"Cashflow Anomalies: {len(result['cashflow_anomalies']['anomalies'])} detected")