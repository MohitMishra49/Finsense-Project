# test_api.py - Test the FastAPI endpoints

import requests
import json

BASE_URL = "http://localhost:8000"

def test_analyze_endpoint():
    """Test the analyze-transaction endpoint with different inputs."""

    test_cases = [
        {
            "description": "bought groceries from supermarket",
            "amount": 2500.0,
            "user_id": "U001",
            "business_id": "BIZ_001"
        },
        {
            "description": "paid electricity bill",
            "amount": 1800.0,
            "user_id": "U002",
            "business_id": "BIZ_002"
        },
        {
            "description": "received salary payment",
            "amount": 45000.0,
            "user_id": "U003",
            "business_id": "BIZ_003"
        }
    ]

    print("=== Testing /analyze-transaction endpoint ===\n")

    for i, data in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {data['description']} - ₹{data['amount']} - {data['business_id']}")

        try:
            response = requests.post(f"{BASE_URL}/analyze-transaction", json=data)
            response.raise_for_status()
            result = response.json()

            # Extract key results
            category = result.get('category')
            amount = result.get('input', {}).get('amount')
            business_id = result.get('input', {}).get('business_id')
            forecast_balance = result.get('forecast', {}).get('start_balance')
            is_anomaly = result.get('anomaly', {}).get('is_anomaly')

            print(f"✓ Category: {category}")
            print(f"✓ Amount: ₹{amount}")
            print(f"✓ Business: {business_id}")
            print(f"✓ Forecast Balance: ₹{forecast_balance:,.0f}")
            print(f"✓ Anomaly: {is_anomaly}")

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON response: {e}")

        print("-" * 50)

def test_frontend_json_format():
    """Show the correct JSON format for frontend integration."""

    print("\n=== Frontend JSON Format ===")
    print("Send POST request to: http://localhost:8000/analyze-transaction")
    print("Content-Type: application/json")
    print("\nExample request body:")
    print(json.dumps({
        "description": "bought office supplies",
        "amount": 1200.0,
        "user_id": "U001",
        "business_id": "BIZ_001",
        "current_balance": 50000.0,
        "forecast_days": 7
    }, indent=2))

    print("\nResponse will include:")
    print("- category: ML-predicted category")
    print("- anomaly: Anomaly detection results")
    print("- forecast: Business-specific cashflow forecast")
    print("- insights: Personalized insights")
    print("- input: Echoed input values")

if __name__ == "__main__":
    try:
        # Test health endpoint first
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code == 200:
            print("✓ API server is running")
            test_analyze_endpoint()
        else:
            print("✗ API server not responding")

    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server. Make sure it's running:")
        print("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")

    test_frontend_json_format()