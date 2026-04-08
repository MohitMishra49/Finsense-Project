# test_pipeline.py
# Simple test script for the improved ML pipeline

import sys
import os
sys.path.append('.')

import pandas as pd
from src.preprocess import clean_text, rule_based_categorize
from src.forecaster import get_business_cashflow, forecast_cashflow, detect_cashflow_anomalies

def test_business_specific_data():
    """Test that different businesses have different data."""
    print("=== Testing Business-Specific Data ===")

    for bid in ['BIZ_001', 'BIZ_002', 'BIZ_003']:
        df = get_business_cashflow(bid)
        if not df.empty:
            balance = df['cumulative_balance'].iloc[-1]
            mean_net = df['net_cashflow'].mean()
            print(f"{bid}: Balance = ₹{balance:,.0f}, Mean Daily Net = ₹{mean_net:,.0f}")
        else:
            print(f"{bid}: No data")

def test_categorization():
    """Test improved categorization."""
    print("\n=== Testing Categorization ===")

    test_descriptions = [
        "bought groceries from supermarket",
        "paid electricity bill",
        "salary payment received",
        "taxi fare to airport",
        "bought medicine from pharmacy"
    ]

    for desc in test_descriptions:
        rule_cat = rule_based_categorize(clean_text(desc))
        print(f"'{desc}' -> {rule_cat}")

def test_forecast():
    """Test business-specific forecasting."""
    print("\n=== Testing Forecast ===")

    for bid in ['BIZ_001', 'BIZ_002']:
        forecast = forecast_cashflow(bid, days=3)
        print(f"{bid}: Start Balance = ₹{forecast['start_balance']:,.0f}, Trend = {forecast['trend']}")
        print(f"  Final Balance: ₹{forecast['final_balance']:,.0f}")

def test_anomalies():
    """Test cashflow anomaly detection."""
    print("\n=== Testing Cashflow Anomalies ===")

    for bid in ['BIZ_001', 'BIZ_002']:
        anomalies = detect_cashflow_anomalies(bid)
        print(f"{bid}: {len(anomalies['anomalies'])} anomalies detected")

if __name__ == '__main__':
    test_business_specific_data()
    test_categorization()
    test_forecast()
    test_anomalies()
    print("\n✓ All tests completed!")