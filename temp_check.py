import src.forecaster as fs

for biz in ['BIZ_002','BIZ_007','BIZ_UNKNOWN']:
    f = fs.forecast_cashflow(biz, current_balance=100000.0, days=7)
    print(biz, 'trend:', f['trend'], 'summary:', f['summary'], 'improvement_pct:', f.get('improvement_pct'))

from src.chatbot_engine import append_forecast_insights_to_response
print(append_forecast_insights_to_response('Hello', {'trend':'growing','summary':'Projected balance after 7 days: ₹105,000 (growing)'}))
