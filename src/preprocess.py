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

def rule_based_categorize(text: str) -> str:
    """
    Rule-based categorization for common transaction patterns.
    Returns 'misc' if no rule matches.
    """
    text = text.lower()
    
    # Food and dining
    if any(word in text for word in ['food', 'restaurant', 'cafe', 'hotel', 'lunch', 'dinner', 'breakfast', 'snacks', 'meal', 'eat', 'drink', 'chai', 'coffee', 'juice']):
        return 'food'
    
    # Transport
    if any(word in text for word in ['taxi', 'auto', 'bus', 'train', 'flight', 'travel', 'petrol', 'fuel', 'uber', 'ola', 'transport']):
        return 'transport'
    
    # Healthcare
    if any(word in text for word in ['hospital', 'doctor', 'medicine', 'pharmacy', 'health', 'medical', 'clinic', 'dawai']):
        return 'healthcare'
    
    # Utilities
    if any(word in text for word in ['electricity', 'water', 'gas', 'internet', 'phone', 'mobile', 'bill', 'utility']):
        return 'utilities'
    
    # Marketing
    if any(word in text for word in ['ad', 'advertisement', 'promotion', 'social media', 'instagram', 'facebook', 'google', 'seo']):
        return 'marketing'
    
    # Raw materials
    if any(word in text for word in ['material', 'supplies', 'inventory', 'stock', 'maal']):
        return 'raw_material'
    
    # Rent
    if any(word in text for word in ['rent', 'lease', 'kiraya']):
        return 'rent'
    
    # Salary
    if any(word in text for word in ['salary', 'wage', 'payroll']):
        return 'salary'
    
    # Shopping
    if any(word in text for word in ['shopping', 'clothes', 'grocery', 'supermarket', 'amazon', 'flipkart']):
        return 'shopping'
    
    # Education
    if any(word in text for word in ['school', 'college', 'course', 'training', 'book', 'education']):
        return 'education'
    
    # Entertainment
    if any(word in text for word in ['movie', 'theater', 'game', 'entertainment', 'party']):
        return 'entertainment'
    
    # Travel
    if any(word in text for word in ['hotel', 'booking', 'trip', 'vacation', 'travel']):
        return 'travel'
    
    # Subscriptions
    if any(word in text for word in ['subscription', 'netflix', 'prime', 'membership']):
        return 'subscriptions'
    
    return 'misc'
