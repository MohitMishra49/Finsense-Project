# src/preprocess.py
# ────────────────────────────────────────────────────────────
# Text cleaning and feature engineering for all models
# ────────────────────────────────────────────────────────────

import re
import pandas as pd
import numpy as np

# Common Hinglish → English mappings
HINGLISH_MAP = {
    'khana':   'food',
    'chai':    'tea food',
    'kiraya':  'rent',
    'bijli':   'electricity',
    'paani':   'water',
    'dawai':   'medicine',
    'dawa':    'medicine',
    'petrol':  'fuel transport',
    'safar':   'travel transport',
    'maal':    'material goods',
    'ghar':    'home',
    'dukaan':  'shop',
    'auto':    'auto transport',
    'nashta':  'breakfast food',
    'sabzi':   'vegetable food',
    'ration':  'grocery food',
}

def normalize_hinglish(text: str) -> str:
    """Replace Hinglish words with English equivalents."""
    for hin, eng in HINGLISH_MAP.items():
        text = re.sub(rf'\b{hin}\b', eng, text, flags=re.IGNORECASE)
    return text

def fix_common_typos(text: str) -> str:
    """Fix frequent OCR / human typos."""
    typos = {
        'restraunt':   'restaurant',
        'resturant':   'restaurant',
        'groccery':    'grocery',
        'grocrey':     'grocery',
        'electricty':  'electricity',
        'medicne':     'medicine',
        'transpotation': 'transportation',
        'maintanence': 'maintenance',
        'miscelanious':'miscellaneous',
        'subscripton': 'subscription',
        'cofee':       'coffee',
        'amazone':     'amazon',
        'shoping':     'shopping',
        'stationary':  'stationery',
    }
    for wrong, right in typos.items():
        text = re.sub(rf'\b{wrong}\b', right, text, flags=re.IGNORECASE)
    return text

def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a description string.
    Steps: lowercase → hinglish → typos → remove noise → strip
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = normalize_hinglish(text)
    text = fix_common_typos(text)
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip() # collapse spaces
    return text

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful ML features to the transactions dataframe.
    Used for anomaly detection and cash-flow modeling.
    """
    df = df.copy()
    df['date']        = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month']       = df['date'].dt.month
    df['week']        = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend']  = df['date'].dt.weekday.isin([5, 6]).astype(int)
    df['is_month_start'] = (df['date'].dt.day <= 5).astype(int)
    df['log_amount']  = np.log1p(df['amount'])

    # Category encode for anomaly model
    df['cat_code'] = df['category'].astype('category').cat.codes
    return df

def load_and_prepare(transactions_path: str,
                     cashflow_path: str) -> tuple:
    """Load datasets and apply full preparation."""
    df = pd.read_csv(transactions_path)
    cf = pd.read_csv(cashflow_path)
    df = engineer_features(df)
    cf['date'] = pd.to_datetime(cf['date'])
    cf = cf.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df):,} transactions | {len(cf)} daily cashflow rows")
    return df, cf
