# FinSense ML Pipeline

Production-ready machine learning system for business cashflow prediction and transaction analysis.

## Features

- ✅ **Business-specific filtering**: All computations filtered by `business_id`
- ✅ **Dynamic balance calculation**: Uses last known balance from dataset
- ✅ **Improved categorization**: Rule-based + TF-IDF ML model
- ✅ **Time-series forecasting**: Exponential smoothing per business
- ✅ **Anomaly detection**: Isolation Forest + Z-score analysis
- ✅ **Week-over-week insights**: Correctly computed per business
- ✅ **Modular code**: Clean, production-ready functions

## Key Improvements

1. **Balance Issue Fixed**: Now uses `cumulative_balance` from dataset instead of hardcoded values
2. **Category Diversity**: Rule-based classifier prevents same category for all transactions
3. **Business-Specific Forecasts**: Each business gets unique forecast based on its historical data
4. **Anomaly Detection**: Detects outliers in cashflow using mean/std per business
5. **Insights**: Week-over-week spend calculated correctly per business

## Usage

### Basic Transaction Analysis

```python
from src.ml_pipeline import CashflowMLPipeline

# Initialize pipeline
pipeline = CashflowMLPipeline()

# Analyze a transaction
result = pipeline.analyze_transaction(
    description="bought groceries from supermarket",
    amount=2500.0,
    user_id="U001",
    business_id="BIZ_001",
    current_balance=100000.0,
    forecast_days=7
)

print(f"Category: {result['category']['category']}")
print(f"Anomaly: {result['anomaly']['is_anomaly']}")
print(f"Forecast Balance: ₹{result['forecast']['final_balance']:,.0f}")
```

### Individual Components

```python
# Get business-specific data
biz_data = pipeline.get_business_data("BIZ_001")
print(f"Transactions: {len(biz_data['transactions'])}")
print(f"Balance: ₹{biz_data['cashflow']['cumulative_balance'].iloc[-1]:,.0f}")

# Categorize transaction
category = pipeline.categorize_transaction("paid electricity bill")
print(f"Category: {category['category']} ({category['confidence']}%)")

# Generate forecast
forecast = pipeline.generate_forecast("BIZ_001", days=7)
print(f"Trend: {forecast['trend']}")
print(f"Min Balance: ₹{forecast['min_balance']:,.0f}")

# Detect anomalies
anomalies = detect_cashflow_anomalies("BIZ_001")
print(f"Anomalies found: {len(anomalies['anomalies'])}")
```

## Model Loading

The pipeline automatically loads trained models from the `models/` directory:

- `category_model.pkl` - Transaction categorization
- `vectorizer.pkl` - TF-IDF vectorizer
- `anomaly_model.pkl` - Isolation Forest for anomalies
- `anomaly_scaler.pkl` - Feature scaler

## Data Requirements

- `data/transactions.csv` - Transaction history
- `data/daily_cashflow_by_business.csv` - Daily cashflow by business

## Business-Specific Outputs

All outputs now vary by `business_id`:

- **Balance**: Different starting balances from dataset
- **Forecast**: Based on business's historical cashflow patterns
- **Anomalies**: Detected using business-specific statistics
- **Insights**: Week-over-week calculated from business transactions
- **Categories**: ML model trained on diverse data

## Testing

Run the test script to verify business-specific behavior:

```bash
python test_pipeline.py
```

Expected output shows different balances, trends, and anomaly counts for each business.

## API Integration

For FastAPI integration, use the `CashflowMLPipeline` class in your endpoints:

```python
from fastapi import FastAPI
from src.ml_pipeline import CashflowMLPipeline

app = FastAPI()
pipeline = CashflowMLPipeline()

@app.post("/analyze")
def analyze_transaction(request: dict):
    return pipeline.analyze_transaction(**request)
```