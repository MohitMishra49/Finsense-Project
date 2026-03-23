# ============================================================
# FinSense AI — Exploratory Data Analysis
# Run: python notebooks/01_EDA.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#161820',
    'axes.edgecolor':   '#2a2d3a',
    'axes.labelcolor':  '#e8eaf0',
    'xtick.color':      '#9ca3af',
    'ytick.color':      '#9ca3af',
    'text.color':       '#e8eaf0',
    'grid.color':       '#2a2d3a',
    'grid.linewidth':   0.5,
})
PALETTE = ['#00e5a0','#6c63ff','#ff6b6b','#ffb830','#f472b6',
           '#34d399','#a78bfa','#fb923c','#38bdf8','#4ade80',
           '#f87171','#fbbf24','#c084fc','#22d3ee']

# ── Load Data ────────────────────────────────────────────────
print("Loading datasets...")
df      = pd.read_csv('data/transactions.csv')
cf      = pd.read_csv('data/daily_cashflow.csv')
anomaly = pd.read_csv('data/anomaly_data.csv')

df['date'] = pd.to_datetime(df['date'])
cf['date'] = pd.to_datetime(cf['date'])

print(f"Transactions: {len(df):,} rows | {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Businesses: {df['business_id'].nunique()} | Users: {df['user_id'].nunique()}")
print(f"Categories: {df['category'].nunique()}")
print(f"Anomalies labeled: {anomaly['is_anomaly'].sum()}")
print()

# ════════════════════════════════════════════════════════════
# FIGURE 1 — Category Distribution
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Category Analysis', fontsize=16, fontweight='bold', y=1.01)

cat_counts = df['category'].value_counts()
axes[0].barh(cat_counts.index, cat_counts.values,
             color=PALETTE[:len(cat_counts)])
axes[0].set_title('Transaction Count per Category')
axes[0].set_xlabel('Count')
for i, (idx, val) in enumerate(cat_counts.items()):
    axes[0].text(val + 20, i, str(val), va='center', fontsize=9, color='#9ca3af')

cat_amt = df[df['type']=='expense'].groupby('category')['amount'].sum().sort_values(ascending=True)
axes[1].barh(cat_amt.index, cat_amt.values,
             color=PALETTE[:len(cat_amt)])
axes[1].set_title('Total Spend per Category (₹)')
axes[1].set_xlabel('Total Amount (₹)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('notebooks/fig1_category_dist.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✓ Figure 1 saved — category distribution")

# ════════════════════════════════════════════════════════════
# FIGURE 2 — Monthly Spending Trends
# ════════════════════════════════════════════════════════════
df['month_year'] = df['date'].dt.to_period('M').astype(str)

monthly = df.groupby(['month_year','type'])['amount'].sum().unstack(fill_value=0)
monthly = monthly.sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
x = range(len(monthly))
width = 0.35
if 'income' in monthly.columns:
    ax.bar([i - width/2 for i in x], monthly['income'],
           width, label='Income', color='#00e5a0', alpha=0.85)
if 'expense' in monthly.columns:
    ax.bar([i + width/2 for i in x], monthly['expense'],
           width, label='Expense', color='#ff6b6b', alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(monthly.index, rotation=30, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'₹{v/1e6:.1f}M'))
ax.set_title('Monthly Income vs Expense', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig2_monthly_trends.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✓ Figure 2 saved — monthly trends")

# ════════════════════════════════════════════════════════════
# FIGURE 3 — Weekday vs Weekend Spending
# ════════════════════════════════════════════════════════════
df['is_weekend'] = df['day_of_week'].isin(['Saturday','Sunday'])
wkd = df[df['type']=='expense'].groupby('is_weekend')['amount'].agg(['mean','sum'])
wkd.index = ['Weekday','Weekend']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Weekday vs Weekend Spending', fontsize=14, fontweight='bold')

axes[0].bar(wkd.index, wkd['mean'], color=['#6c63ff','#ffb830'])
axes[0].set_title('Average Transaction Amount')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
for i, (idx, row) in enumerate(wkd.iterrows()):
    axes[0].text(i, row['mean'] + 10, f'₹{row["mean"]:,.0f}', ha='center', fontsize=10)

axes[1].bar(wkd.index, wkd['sum'], color=['#6c63ff','#ffb830'])
axes[1].set_title('Total Spend Amount')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'₹{v/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('notebooks/fig3_weekend_pattern.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✓ Figure 3 saved — weekend patterns")

# ════════════════════════════════════════════════════════════
# FIGURE 4 — Business-wise Expense Breakdown
# ════════════════════════════════════════════════════════════
biz_cat = df[df['type']=='expense'].groupby(
    ['business_id','category'])['amount'].sum().unstack(fill_value=0)
biz_cat_pct = biz_cat.div(biz_cat.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(16, 7))
biz_cat_pct.plot(kind='bar', stacked=True, ax=ax,
                 color=PALETTE[:len(biz_cat_pct.columns)],
                 width=0.75, alpha=0.9)
ax.set_title('Business-wise Category Spend Distribution (%)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Business ID')
ax.set_ylabel('% of Total Spend')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig('notebooks/fig4_business_breakdown.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✓ Figure 4 saved — business breakdown")

# ════════════════════════════════════════════════════════════
# FIGURE 5 — Cash Flow Over Time + Rolling Average
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Cash Flow Analysis', fontsize=14, fontweight='bold')

axes[0].fill_between(cf['date'],
    cf['net_cashflow'].clip(lower=0), 0,
    alpha=0.4, color='#00e5a0', label='Positive')
axes[0].fill_between(cf['date'],
    cf['net_cashflow'].clip(upper=0), 0,
    alpha=0.4, color='#ff6b6b', label='Negative')
axes[0].plot(cf['date'], cf['net_cashflow'],
             color='white', linewidth=0.8, alpha=0.6)
axes[0].axhline(0, color='#9ca3af', linewidth=0.8, linestyle='--')
axes[0].set_title('Daily Net Cash Flow')
axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v,_: f'₹{v/1e3:.0f}K'))
axes[0].legend()

axes[1].plot(cf['date'], cf['cumulative_balance'],
             color='#6c63ff', linewidth=2)
axes[1].fill_between(cf['date'], cf['cumulative_balance'],
                     alpha=0.15, color='#6c63ff')
axes[1].set_title('Cumulative Balance Over Time')
axes[1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v,_: f'₹{v/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('notebooks/fig5_cashflow.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✓ Figure 5 saved — cash flow")

# ════════════════════════════════════════════════════════════
# FIGURE 6 — Anomaly Detection Visualization
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
normal  = anomaly[anomaly['is_anomaly']==0]
anom_pts = anomaly[anomaly['is_anomaly']==1]

ax.scatter(range(len(normal)), normal['amount'].values,
           c='#00e5a0', alpha=0.3, s=10, label='Normal')
ax.scatter(anom_pts.index.tolist()[:len(anom_pts)],
           anom_pts['amount'].values,
           c='#ff6b6b', alpha=0.9, s=40,
           marker='X', label=f'Anomaly ({len(anom_pts)})')
ax.set_title('Transaction Amounts — Anomaly Visualization',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Transaction Index')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v,_: f'₹{v/1e3:.0f}K'))
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('notebooks/fig6_anomalies.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✓ Figure 6 saved — anomaly visualization")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS REPORT
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY EDA INSIGHTS FOR PRODUCT")
print("="*60)

top_cat  = df[df['type']=='expense']['category'].value_counts().index[0]
top_amt  = df[df['type']=='expense'].groupby('category')['amount'].sum().idxmax()
wknd_avg = df[(df['type']=='expense') & df['is_weekend']]['amount'].mean()
wkdy_avg = df[(df['type']=='expense') & ~df['is_weekend']]['amount'].mean()
spike_pct = round((wknd_avg - wkdy_avg) / wkdy_avg * 100, 1)
anom_rate = round(anomaly['is_anomaly'].mean() * 100, 1)

print(f"1. Most frequent expense category : {top_cat}")
print(f"2. Highest spend category (total) : {top_amt}")
print(f"3. Weekend spend is {spike_pct}% higher than weekday avg")
print(f"4. Anomaly rate in dataset        : {anom_rate}%")
print(f"5. Total businesses tracked       : {df['business_id'].nunique()}")
print(f"6. Date range covered             : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"7. Avg transaction amount         : ₹{df['amount'].mean():,.0f}")
print(f"8. Median transaction amount      : ₹{df['amount'].median():,.0f}")
print()
print("All EDA figures saved to notebooks/")
