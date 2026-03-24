# src/train_models.py
# ────────────────────────────────────────────────────────────
# Trains all 3 models and saves them to models/
# Run: python src/train_models.py
# ────────────────────────────────────────────────────────────

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model           import LogisticRegression
from sklearn.model_selection        import train_test_split
from sklearn.metrics                import (accuracy_score,
                                            classification_report,
                                            confusion_matrix)
from sklearn.ensemble               import IsolationForest
from sklearn.preprocessing          import StandardScaler

from src.preprocess import clean_text, load_and_prepare

os.makedirs('models', exist_ok=True)

# ════════════════════════════════════════════════════════════
# MODEL 1 — Expense Categorization (TF-IDF + LogReg)
# ════════════════════════════════════════════════════════════
def train_categorizer(data_path='data/categorization_train.csv'):
    print("\n" + "─"*50)
    print("MODEL 1: Expense Categorization")
    print("─"*50)

    cat_df = pd.read_csv(data_path)
    cat_df['clean'] = cat_df['description'].apply(clean_text)
    cat_df = cat_df[cat_df['clean'].str.len() > 1].reset_index(drop=True)

    X = cat_df['clean']
    y = cat_df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Classes: {sorted(y.unique())}")

    # TF-IDF: unigrams + bigrams, top 5000 features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        sublinear_tf=True,   # log-scale TF — better for short text
        analyzer='word',
    )
    X_tr_vec = vectorizer.fit_transform(X_train)
    X_te_vec = vectorizer.transform(X_test)

    # Logistic Regression with L2 regularization
    model = LogisticRegression(
        max_iter=1000,
        C=1.5,
        solver='lbfgs',
        random_state=42,
    )
    model.fit(X_tr_vec, y_train)

    y_pred = model.predict(X_te_vec)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save
    joblib.dump(model,      'models/category_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    # Save class labels for reference
    meta = {
        'classes':   list(model.classes_),
        'accuracy':  round(acc, 4),
        'n_features': vectorizer.max_features,
        'ngram_range': list(vectorizer.ngram_range),
    }
    with open('models/categorizer_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved: models/category_model.pkl")
    print(f"✓ Saved: models/vectorizer.pkl")
    return model, vectorizer, acc

# ════════════════════════════════════════════════════════════
# MODEL 2 — Anomaly Detection (Isolation Forest)
# ════════════════════════════════════════════════════════════
def train_anomaly_detector(data_path='data/anomaly_data.csv'):
    print("\n" + "─"*50)
    print("MODEL 2: Anomaly Detection")
    print("─"*50)

    anom_df = pd.read_csv(data_path)

    # Features: amount + category code
    anom_df['cat_code'] = anom_df['category'].astype('category').cat.codes
    anom_df['log_amt']  = np.log1p(anom_df['amount'])

    # Save category code mapping
    cat_map = dict(enumerate(anom_df['category'].astype('category').cat.categories))
    with open('models/cat_code_map.json', 'w') as f:
        json.dump(cat_map, f)

    X = anom_df[['log_amt', 'cat_code']].values
    # 🔥 ADD THIS BLOCK (Z-score stats)
    mean_amt = anom_df['amount'].mean()
    std_amt  = anom_df['amount'].std()

    stats = {
    "mean": float(mean_amt),
    "std": float(std_amt)
     }
    with open('models/anomaly_stats.json', 'w') as f:
        json.dump(stats, f)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train on normal transactions only (5% contamination)
    iso = IsolationForest(
        contamination=0.05,
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # Evaluate against labeled anomalies
    preds     = iso.predict(X_scaled)   # 1=normal, -1=anomaly
    predicted = (preds == -1).astype(int)
    actual    = anom_df['is_anomaly'].values

    from sklearn.metrics import classification_report as cr
    print("\nEvaluation vs labeled anomalies:")
    print(cr(actual, predicted, target_names=['normal','anomaly'],
             zero_division=0))

    joblib.dump(iso,    'models/anomaly_model.pkl')
    joblib.dump(scaler, 'models/anomaly_scaler.pkl')

    with open('models/anomaly_categories.json', 'w') as f:
        categories = anom_df['category'].unique().tolist()
        json.dump(categories, f)

    print(f"✓ Saved: models/anomaly_model.pkl")
    return iso, scaler

# ════════════════════════════════════════════════════════════
# MODEL 3 — Cash Flow Prediction (ARIMA-style)
# ════════════════════════════════════════════════════════════
def train_cashflow_predictor(data_path='data/daily_cashflow.csv'):
    print("\n" + "─"*50)
    print("MODEL 3: Cash Flow Prediction")
    print("─"*50)

    cf = pd.read_csv(data_path)
    cf['date'] = pd.to_datetime(cf['date'])
    cf = cf.sort_values('date').reset_index(drop=True)

    series = cf['net_cashflow'].values
    print(f"Time series length: {len(series)} days")
    print(f"Mean daily cash flow: ₹{series.mean():,.0f}")
    print(f"Std: ₹{series.std():,.0f}")

    # Try ARIMA, fall back to rolling mean if statsmodels not available
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools  import adfuller

        # ADF test for stationarity
        adf_result = adfuller(series, autolag='AIC')
        is_stationary = adf_result[1] < 0.05
        print(f"ADF p-value: {adf_result[1]:.4f} | "
              f"{'Stationary' if is_stationary else 'Non-stationary'}")

        # Choose d based on stationarity
        d = 0 if is_stationary else 1
        arima = ARIMA(series, order=(2, d, 2))
        arima_fit = arima.fit()

        forecast = arima_fit.forecast(steps=14)
        print(f"\n14-day forecast (ARIMA):")
        for i, v in enumerate(forecast, 1):
            print(f"  Day +{i:2d}: ₹{v:,.0f}")

        joblib.dump(arima_fit, 'models/arima_model.pkl')
        model_type = 'arima'
        print(f"\n✓ Saved: models/arima_model.pkl")

    except ImportError:
        print("statsmodels not found — using Rolling Mean predictor")
        # Simple but effective: weighted rolling average
        window   = 7
        weights  = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()

        def rolling_forecast(series, steps=14, window=7):
            history  = list(series[-window:])
            forecast = []
            for _ in range(steps):
                pred = np.sum(np.array(history[-window:]) * weights)
                forecast.append(round(pred, 2))
                history.append(pred)
            return forecast

        forecast = rolling_forecast(series, steps=14)

        # 🔥 CLAMP VALUES (IMPORTANT)
        forecast = [max(min(x, 200000), -100000) for x in forecast]
        print("\n14-day forecast (Rolling Mean):")
        for i, v in enumerate(forecast, 1):
            print(f"  Day +{i:2d}: ₹{v:,.0f}")

        joblib.dump({'type': 'rolling', 'history': series[-14:].tolist()},
                    'models/arima_model.pkl')
        model_type = 'rolling'

    # Save cashflow stats for inference
    stats = {
        'model_type':    model_type,
        'mean_daily':    float(series.mean()),
        'std_daily':     float(series.std()),
        'last_balance':  float(cf['cumulative_balance'].iloc[-1]),
        'last_date':     str(cf['date'].iloc[-1].date()),
    }
    with open('models/cashflow_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return forecast

# ════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*50)
    print("FinSense AI — Model Training")
    print("="*50)

    cat_model, vectorizer, acc = train_categorizer()
    iso_model, scaler          = train_anomaly_detector()
    forecast                   = train_cashflow_predictor()

    print("\n" + "="*50)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print(f"Categorization accuracy : {acc*100:.1f}%")
    print(f"Models saved to         : models/")
    print("="*50)
