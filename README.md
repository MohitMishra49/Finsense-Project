# FinSense AI — Intelligent Financial Analytics Assistant for Small Businesses

FinSense AI is an end-to-end AI-powered financial intelligence backend designed for small and medium businesses to monitor expenses, detect unusual spending, forecast future cash flow, and interact with business financial data through a conversational AI assistant.

This project combines Machine Learning, Forecast Analytics, Explainable AI, and LLM-powered financial chat into one deployable FastAPI backend.

---

## 🚀 Key Features

### ✅ Smart Transaction Categorization
Automatically predicts the category of a transaction using NLP-based text classification.

Example:
> "Paid 4500 for vegetables from wholesale market" → Inventory / Raw Materials

---

### ✅ Explainable AI Predictions
Every ML categorization is accompanied by:
- confidence score
- top contributing keywords
- prediction reasoning

This makes model outputs interpretable instead of black-box.

---

### ✅ Expense Anomaly Detection
Detects suspicious or unusually high expenses using:
- Isolation Forest anomaly model
- Z-score historical deviation logic

Useful for identifying financial leakage and abnormal spending patterns.

---

### ✅ Business Financial Insights Engine
Generates:
- week-over-week spending comparisons
- top expense category analysis
- savings rate monitoring
- personalized user spending insights

---

### ✅ Cash Flow Forecasting
Forecasts the next 7–30 days of:
- projected balance
- declining/growing trend
- low balance alerts
- negative balance risk

using business-specific historical daily cash flow.

---

### ✅ AI Financial Chatbot (LLM Powered)
Integrated with HuggingFace LLM to allow natural language business financial conversations.

Users can ask:
- "How is my business performing?"
- "Where am I overspending?"
- "Will my balance go negative this week?"

and receive contextual AI-generated financial guidance.

---

### ✅ Business Summary API
Provides instant business health summary:
- total income
- total expense
- net profit/loss
- savings rate
- top spending categories
- financial health recommendation

---

# 🧠 Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas / NumPy
- Joblib
- HuggingFace Inference API
- Uvicorn

---

# 🏗️ Project Architecture

```bash
FinSense/
│
├── api/
│   └── main.py                # FastAPI backend entrypoint
│
├── src/
│   ├── pipeline.py           # unified ML pipeline
│   ├── preprocess.py         # text cleaning + rule categorization
│   ├── explainer.py          # explainable AI logic
│   ├── insights.py           # business analytics insights
│   ├── forecaster.py         # cashflow forecasting engine
│   └── chatbot_engine.py     # LLM financial context builder
│
├── models/
│   ├── category_model.pkl
│   ├── vectorizer.pkl
│   ├── anomaly_model.pkl
│   ├── anomaly_scaler.pkl
│   ├── categorizer_meta.json
│   └── cat_code_map.json
│
├── data/
│   ├── transactions.csv
│   └── daily_cashflow_by_business.csv
│
├── requirements.txt
├── runtime.txt
└── README.md