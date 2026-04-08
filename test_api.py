import requests

# Test with different inputs
test_cases = [
    {'description': 'bought pizza', 'amount': 400.0, 'business_id': 'BIZ_001'},
    {'description': 'paid electricity bill', 'amount': 2500.0, 'business_id': 'BIZ_002'},
    {'description': 'salary received', 'amount': 50000.0, 'business_id': 'BIZ_003'},
]

for i, data in enumerate(test_cases):
    data.update({'user_id': 'U001'})
    response = requests.post('http://localhost:8000/analyze-transaction', json=data)
    result = response.json()
    category = result.get('category')
    amount = result.get('input', {}).get('amount')
    business_id = result.get('input', {}).get('business_id')
    forecast_balance = result.get('forecast', {}).get('start_balance')
    print(f'Test {i+1}: {data["description"]} - Category: {category}, Amount: {amount}, Business: {business_id}, Balance: {forecast_balance}')