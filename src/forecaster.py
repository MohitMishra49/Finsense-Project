# src/forecaster.py
# ══════════════════════════════════════════════════════════════
# Per-business cash flow forecasting
# Replaces the old single-series ARIMA approach.
# Uses daily_cashflow_by_business.csv which has business_id column.
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
from typing import Optional

_cashflow_df: Optional[pd.DataFrame] = None


def _load_cashflow():
    global _cashflow_df
    if _cashflow_df is not None:
        return

    path = os.path.join('data', 'daily_cashflow_by_business.csv')
    if not os.path.exists(path):
        raise FileNotFoundError("daily_cashflow_by_business.csv is required")

    _cashflow_df = pd.read_csv(path)
    _cashflow_df['date'] = pd.to_datetime(_cashflow_df['date'])
    print(f"[Forecaster] Loaded {path} - per-business")


def get_business_cashflow(business_id: str) -> pd.DataFrame:
    """
    Returns the daily cashflow rows for a single business.
    """
    _load_cashflow()
    biz_id = business_id.strip().upper()

    if 'business_id' not in _cashflow_df.columns:
        return pd.DataFrame()

    bdf = _cashflow_df[
        _cashflow_df['business_id'].astype(str).str.upper() == biz_id
    ].copy()
    if bdf.empty:
        return pd.DataFrame()
    return bdf.sort_values('date').reset_index(drop=True)


def forecast_cashflow(
    business_id:     str,
    current_balance: Optional[float] = None,
    income:          Optional[float] = None,
    expense:         Optional[float] = None,
    days:            int = 7,
) -> dict:
    """
    Generate a per-business cash flow forecast.

    Strategy:
      1. Load this business's historical daily cashflow
      2. Use weighted rolling average of last 30 days as predictor
         (recent days weighted more heavily)
      3. Add slight noise for realism
      4. Build running balance projection
      5. Flag alerts if balance drops below threshold

    Args:
        business_id     : e.g. "BIZ_001"
        current_balance : override the balance from CSV (optional)
        days            : forecast horizon (default 7)

    Returns:
        dict with daily forecast, min balance, alerts, summary
    """
    bdf = get_business_cashflow(business_id)

    if bdf.empty:
        # Business not found or no data available: safe fallback
        return _empty_forecast(business_id, current_balance or 0, days)

    # Use last 30 days of history
    history = bdf.tail(30)
    net_series = history['net_cashflow'].values

    # Improved forecasting: exponential smoothing with trend
    if len(net_series) >= 7:
        # Simple exponential smoothing with alpha=0.3
        alpha = 0.3
        smoothed = [net_series[0]]
        for i in range(1, len(net_series)):
            smoothed.append(alpha * net_series[i] + (1 - alpha) * smoothed[-1])
        
        # Calculate trend from last 7 days
        recent_trend = np.polyfit(range(7), smoothed[-7:], 1)[0]
        base_daily_net = float(smoothed[-1] + recent_trend * 0.5)  # slight trend adjustment
    else:
        # Fallback to weighted average
        window  = min(14, len(net_series))
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()
        recent  = net_series[-window:]
        base_daily_net = float(np.sum(recent * weights))

    # Std for realistic noise
    std_daily = float(net_series.std()) if len(net_series) > 3 else abs(base_daily_net) * 0.15
    std_daily = min(std_daily, abs(base_daily_net) * 0.3)  # cap noise at 30%

    if current_balance is not None:
        start_balance = float(current_balance)
    elif not bdf.empty:
        start_balance = float(bdf['cumulative_balance'].iloc[-1])
    else:
        start_balance = 0.0

    # Build forecast
    np.random.seed(42)  # reproducible for same business
    daily_forecasts = []
    running_balance = start_balance

    for i in range(days):
        noise     = np.random.normal(0, std_daily * 0.1)
        daily_net = base_daily_net + noise
        running_balance += daily_net
        daily_forecasts.append({
            'day':           i + 1,
            'net_cashflow':  round(float(daily_net), 2),
            'balance':       round(float(running_balance), 2),
        })

    min_balance    = min(d['balance'] for d in daily_forecasts)
    final_balance  = daily_forecasts[-1]['balance']
    trend          = 'growing' if base_daily_net > 0 else 'declining'

    # Alert thresholds
    alert = None
    if min_balance < 0:
        low_day = next(d['day'] for d in daily_forecasts if d['balance'] == min_balance)
        alert   = (f"Balance turns negative by Day {low_day} "
                   f"(₹{min_balance:,.0f}) — urgent action needed")
    elif min_balance < 10000:
        low_day = next(d['day'] for d in daily_forecasts if d['balance'] == min_balance)
        alert   = (f"Balance may drop to ₹{min_balance:,.0f} "
                   f"by Day {low_day} — consider reducing expenses")

    # Month-over-month context + improvement cap/rounding
    if len(bdf) >= 60:
        last_30 = bdf.tail(30)['net_cashflow'].sum()
        prev_30 = bdf.tail(60).head(30)['net_cashflow'].sum()
        if prev_30 != 0:
            improvement = round((last_30 - prev_30) / abs(prev_30) * 100, 2)
        else:
            improvement = 0.0
        improvement = min(improvement, 300.0)
        mom_str = f" | Cash flow {abs(improvement):.0f}% {'better' if improvement > 0 else 'worse'} than prior month"
    else:
        improvement = 0.0
        mom_str = ""

    # Detect cashflow anomalies
    anomalies = detect_cashflow_anomalies(business_id)

    return {
        'improvement_pct': improvement,
        'business_id':      business_id.upper(),
        'days':             days,
        'start_balance':    round(start_balance, 2),
        'base_daily_net':   round(base_daily_net, 2),
        'trend':            trend,
        'daily':            daily_forecasts,
        'min_balance':      round(min_balance, 2),
        'final_balance':    round(final_balance, 2),
        'alert':            alert,
        'anomalies':        anomalies,
        'summary':          f"Projected balance after {days} days: ₹{int(final_balance):,} ({trend})",
    }


def _derive_forecast_from_transactions(
    business_id: str,
    current_balance: Optional[float],
    days: int,
) -> dict:
    """
    Fallback: derive forecast directly from transactions.csv
    when this business is missing from the cashflow CSV.
    """
    tx_path = os.path.join('data', 'transactions.csv')
    if not os.path.exists(tx_path):
        return _empty_forecast(business_id, current_balance or 0, days)

    tx  = pd.read_csv(tx_path)
    tx['date'] = pd.to_datetime(tx['date'])
    bdf = tx[tx['business_id'].str.upper() == business_id.upper()]

    if bdf.empty:
        return _empty_forecast(business_id, current_balance or 0, days)

    # Last 30 days average daily net
    recent = bdf[bdf['date'] >= bdf['date'].max() - pd.Timedelta(days=30)]
    inc    = float(recent[recent['type']=='income']['amount'].sum())
    exp    = float(recent[recent['type']=='expense']['amount'].sum())
    net30  = inc - exp
    avg_daily_net = net30 / 30

    start  = current_balance or 50000.0
    forecasts = []
    bal    = start
    for i in range(days):
        bal += avg_daily_net
        forecasts.append({'day': i+1, 'net_cashflow': round(avg_daily_net, 2),
                          'balance': round(bal, 2)})

    min_bal = min(d['balance'] for d in forecasts)
    trend   = 'growing' if avg_daily_net > 0 else 'declining'

    return {
        'business_id':   business_id.upper(),
        'days':          days,
        'start_balance': round(start, 2),
        'base_daily_net':round(avg_daily_net, 2),
        'trend':         trend,
        'daily':         forecasts,
        'min_balance':   round(min_bal, 2),
        'final_balance': round(forecasts[-1]['balance'], 2),
        'alert':         "Balance turning negative" if min_bal < 0 else None,
        'summary':       f"Projected balance after {days} days: ₹{int(forecasts[-1]['balance']):,} ({trend})",
    }


def _empty_forecast(business_id, balance, days):
    return {
        'business_id':   business_id,
        'days':          days,
        'start_balance': balance,
        'base_daily_net':0,
        'trend':         'unknown',
        'daily':         [{'day': i+1, 'net_cashflow': 0, 'balance': round(balance, 2)} for i in range(days)],
        'min_balance':   balance,
        'final_balance': balance,
        'alert':         None,
        'summary':       "Not enough data for forecast",
    }


def detect_cashflow_anomalies(business_id: str) -> dict:
    """
    Detect anomalies in cashflow using mean and std deviation.
    """
    bdf = get_business_cashflow(business_id)
    if bdf.empty or len(bdf) < 7:
        return {'anomalies': [], 'message': 'Not enough data for anomaly detection'}
    
    net_series = bdf['net_cashflow'].values
    mean_val = np.mean(net_series)
    std_val = np.std(net_series)
    
    if std_val == 0:
        return {'anomalies': [], 'message': 'No variation in cashflow'}
    
    anomalies = []
    threshold = 2.0  # Z-score threshold
    
    for i, val in enumerate(net_series):
        z_score = abs((val - mean_val) / std_val)
        if z_score > threshold:
            anomalies.append({
                'date': str(bdf['date'].iloc[i].date()),
                'amount': float(val),
                'z_score': round(z_score, 2),
                'deviation': 'high' if val > mean_val else 'low'
            })
    
    return {
        'anomalies': anomalies,
        'mean': round(mean_val, 2),
        'std': round(std_val, 2),
        'message': f'Found {len(anomalies)} anomalies in cashflow'
    }
