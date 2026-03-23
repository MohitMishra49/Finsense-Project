# FinSense AI — Complete ML System
# README.md
# ════════════════════════════════════════════════════════════

## Project Structure

```
finsense/
│
├── data/                          # Put your CSV files here
│   ├── transactions.csv           # 11k+ rows main dataset
│   ├── categorization_train.csv   # 5k NLP training data
│   ├── daily_cashflow.csv         # Time series data
│   ├── anomaly_data.csv           # Labeled anomaly data
│   └── sentence_transformer_data.csv
│
├── notebooks/
│   └── 01_EDA.py                  # Full exploratory analysis
│                                  # Generates 6 charts
│
├── src/
│   ├── preprocess.py              # Text cleaning, Hinglish,
│   │                              # typo fixing, feature engineering
│   ├── train_models.py            # Trains all 3 models + saves them
│   ├── explainer.py               # WHY did model predict this?
│   │                              # TF-IDF keyword extraction
│   ├── insights.py                # Smart insights engine
│   │                              # Week-over-week, savings rate,
│   │                              # personal averages
│   └── pipeline.py                # THE UNIFIED PIPELINE
│                                  # One input → complete output
│
├── api/
│   └── main.py                    # FastAPI application
│                                  # /analyze-transaction endpoint
│
├── models/                        # Auto-created on training
│   ├── category_model.pkl
│   ├── vectorizer.pkl
│   ├── anomaly_model.pkl
│   ├── anomaly_scaler.pkl
│   ├── arima_model.pkl
│   ├── categorizer_meta.json
│   └── cashflow_stats.json
│
├── demo/
│   └── run_demo.py               # 5 demo examples + judge script
│
└── requirements.txt

```

## Setup & Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas numpy \
            matplotlib seaborn joblib statsmodels pydantic
```

### 2. Place datasets in data/
```bash
cp /path/to/transactions.csv data/
cp /path/to/categorization_train.csv data/
cp /path/to/daily_cashflow.csv data/
cp /path/to/anomaly_data.csv data/
```

### 3. Run EDA
```bash
python notebooks/01_EDA.py
# Generates 6 charts in notebooks/
```

### 4. Train all models
```bash
python src/train_models.py
# Trains categorizer (~93% accuracy)
# Trains anomaly detector
# Trains cash flow predictor
# Saves all to models/
```

### 5. Run demo
```bash
python demo/run_demo.py
# Shows 5 examples with full output
```

### 6. Start API
```bash
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## API Usage

### Single transaction
```bash
curl -X POST http://localhost:8000/analyze-transaction \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Swiggy dinner order kiya",
    "amount": 850,
    "user_id": "U008",
    "business_id": "BIZ_008",
    "current_balance": 42000
  }'
```

### Response structure
```json
{
  "category":    "food",
  "confidence":  91.4,
  "reason":      ["dinner", "swiggy", "order"],
  "explanation": {
    "predicted":          "food",
    "confidence":         91.4,
    "top_keywords":       ["dinner", "swiggy", "order"],
    "reasoning_sentence": "Predicted as food because..."
  },
  "anomaly": {
    "is_anomaly":   false,
    "z_score":      0.8,
    "explanation":  "₹850 is within normal range..."
  },
  "insights": [
    {
      "type":     "week_over_week",
      "message":  "Your food spending is 28% higher than last week",
      "severity": "warning"
    }
  ],
  "forecast": {
    "summary": "Projected balance after 7 days: ₹38,200",
    "alert":   null
  }
}
```

## What Each ML Component Does

| Component | File | Algorithm | Purpose |
|-----------|------|-----------|---------|
| Categorization | train_models.py | TF-IDF + LogReg | Text → category |
| Explainability | explainer.py | Feature weights | Why this category? |
| Anomaly detection | train_models.py | Isolation Forest + Z-score | Flag unusual spend |
| Insights engine | insights.py | Statistical comparisons | Smart messages |
| Cash flow forecast | train_models.py | ARIMA | Predict future balance |
| Unified pipeline | pipeline.py | Orchestration | All models → 1 output |

## Full Workflow

```
CSV Data
   │
   ├── EDA (01_EDA.py)
   │   └── Understand patterns, distributions, anomalies
   │
   ├── Training (train_models.py)
   │   ├── Model 1: TF-IDF vectorizer + LogReg → category_model.pkl
   │   ├── Model 2: IsolationForest → anomaly_model.pkl
   │   └── Model 3: ARIMA → arima_model.pkl
   │
   ├── Pipeline (pipeline.py)
   │   └── analyze_transaction() → unified output dict
   │
   ├── API (api/main.py)
   │   └── POST /analyze-transaction → JSON response
   │
   └── Frontend
       └── Calls /analyze-transaction → displays results
```
