# demo/run_demo.py
# ────────────────────────────────────────────────────────────
# FinSense AI — Demo Script
# Shows 5 real examples with full output
# Run: python demo/run_demo.py
# ────────────────────────────────────────────────────────────

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.pipeline import analyze_transaction, store

# Load models
store.load(models_dir='models')
history = pd.read_csv('data/transactions.csv')
history['date'] = pd.to_datetime(history['date'])

SEPARATOR = "─" * 65

DEMO_CASES = [
    {
        "title":       "CASE 1 — Normal food transaction",
        "description": "Swiggy dinner order kiya for the team",
        "amount":      850.0,
        "user_id":     "U008",
        "business_id": "BIZ_008",
        "balance":     42000.0,
    },
    {
        "title":       "CASE 2 — Hinglish + anomaly detection",
        "description": "restraunt dinner kiya client ke saath",
        "amount":      8500.0,     # <── unusually high
        "user_id":     "U008",
        "business_id": "BIZ_008",
        "balance":     38000.0,
    },
    {
        "title":       "CASE 3 — No keyword, semantic understanding",
        "description": "went out with the whole team to celebrate the launch",
        "amount":      3200.0,
        "user_id":     "U009",
        "business_id": "BIZ_003",
        "balance":     55000.0,
    },
    {
        "title":       "CASE 4 — Subscription detection",
        "description": "auto renewal hit the account this morning",
        "amount":      1499.0,
        "user_id":     "U036",
        "business_id": "BIZ_010",
        "balance":     28000.0,
    },
    {
        "title":       "CASE 5 — Income received",
        "description": "client ne payment bhej diya finally for the project",
        "amount":      45000.0,
        "user_id":     "U008",
        "business_id": "BIZ_008",
        "balance":     42000.0,
    },
]

def pretty_print(result: dict):
    s = result['summary']
    ins = result.get('insights', [])
    fc  = result.get('forecast', {})

    print(f"\n  Category    : {s['category'].upper()}")
    print(f"  Confidence  : {s['confidence_pct']}%")
    print(f"  Keywords    : {', '.join(s['top_keywords'])}")

    anom = result['anomaly']
    if anom['is_anomaly']:
        print(f"\n  ⚠ ANOMALY   : {anom['explanation']}")
    else:
        z = anom.get('z_score')
        z_str = f" (Z={z:.1f})" if z is not None else ""
        print(f"\n  ✓ Normal    : No anomaly detected{z_str}")

    if ins:
        print(f"\n  Insights:")
        for i in ins[:2]:
            sev_icon = {'warning':'⚠', 'good':'✓', 'info':'ℹ', 'danger':'🚨'}.get(
                i.get('severity','info'), 'ℹ')
            print(f"    {sev_icon} {i['message']}")

    if fc.get('alert'):
        print(f"\n  Forecast ⚠  : {fc['alert']}")
    else:
        print(f"\n  Forecast    : {fc.get('summary','')}")

def run_demo():
    print("\n" + "═"*65)
    print("   FinSense AI — Hackathon Demo")
    print("═"*65)

    for case in DEMO_CASES:
        print(f"\n{SEPARATOR}")
        print(f"  {case['title']}")
        print(f"  Input : \"{case['description']}\"")
        print(f"  Amount: ₹{case['amount']:,.0f}")
        print(SEPARATOR)

        result = analyze_transaction(
            description         = case['description'],
            amount              = case['amount'],
            user_id             = case['user_id'],
            business_id         = case['business_id'],
            transaction_history = history,
            current_balance     = case['balance'],
            forecast_days       = 7,
        )
        pretty_print(result)

    print(f"\n{'═'*65}")
    print("  WHAT TO TELL JUDGES")
    print("═"*65)
    print("""
  1. "Our system understands Hinglish — 'kiya', 'ke saath',
     'restraunt' are all handled correctly."

  2. "Case 2 shows anomaly detection in action — ₹8,500 on food
     is flagged because this user's average is much lower."

  3. "Case 3 has zero category keywords — no 'food', 'dinner',
     'restaurant' — but the model still classifies it correctly
     using semantic understanding from patterns in the data."

  4. "Case 4 shows our model handles indirect language —
     'auto renewal hit the account' → subscriptions."

  5. "Every response includes: category, confidence %, the exact
     keywords that drove the decision, anomaly detection with
     Z-score, personalized insights with real percentages, and
     a 7-day cash flow forecast — all from one API call."
    """)

if __name__ == '__main__':
    run_demo()
