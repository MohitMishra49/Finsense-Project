# src/explainer.py
# ────────────────────────────────────────────────────────────
# Explainable AI — WHY did the model predict this category?
# Uses TF-IDF feature weights from Logistic Regression
# ────────────────────────────────────────────────────────────

import numpy as np
from typing import Optional


def explain_prediction(
    description: str,
    category: str,
    model,
    vectorizer,
    top_n: int = 5,
) -> dict:
    """
    Extract the top keywords that drove the model's decision.

    Args:
        description : raw or cleaned input text
        category    : predicted category label
        model       : trained LogisticRegression
        vectorizer  : fitted TfidfVectorizer
        top_n       : number of keywords to return

    Returns:
        dict with keys: predicted, confidence, top_keywords,
                        keyword_weights, reasoning_sentence
    """
    from src.preprocess import clean_text

    cleaned     = clean_text(description)
    vec_inp     = vectorizer.transform([cleaned])

    # Probability of each class
    proba       = model.predict_proba(vec_inp)[0]
    class_list  = list(model.classes_)
    confidence  = round(float(proba.max()) * 100, 1)

    # Top-2 alternatives (for richer output)
    top2_idx    = np.argsort(proba)[::-1][:2]
    alternatives = [
        {'category': class_list[i], 'confidence': round(float(proba[i])*100, 1)}
        for i in top2_idx[1:]
    ]

    # Feature weights for the predicted class
    class_idx   = class_list.index(category)
    coef_vector = model.coef_[class_idx]            # shape: (n_features,)

    # Active features in this input
    feature_names  = vectorizer.get_feature_names_out()
    active_indices = vec_inp.toarray()[0].nonzero()[0]

    if len(active_indices) == 0:
        # Fallback: return top global weights for this class
        top_global = np.argsort(coef_vector)[::-1][:top_n]
        keyword_weights = {
            feature_names[i]: round(float(coef_vector[i]), 3)
            for i in top_global
        }
    else:
        # Weight = TF-IDF score × logistic coefficient
        tfidf_scores = vec_inp.toarray()[0][active_indices]
        coef_scores  = coef_vector[active_indices]
        combined     = tfidf_scores * coef_scores

        sorted_idx   = np.argsort(combined)[::-1][:top_n]
        keyword_weights = {
            feature_names[active_indices[i]]: round(float(combined[i]), 3)
            for i in sorted_idx
            if combined[i] > 0         # only positive contributors
        }

    top_keywords = list(keyword_weights.keys())[:top_n]

    # Human-readable reasoning sentence
    if top_keywords:
        kw_str = ', '.join(f'"{w}"' for w in top_keywords[:3])
        reasoning = (
            f"Predicted as {category} because the description contains "
            f"words like {kw_str} which strongly indicate this category."
        )
    else:
        reasoning = (
            f"Predicted as {category} based on overall sentence pattern "
            f"(no single dominant keyword found)."
        )

    return {
        'predicted':          category,
        'confidence':         confidence,
        'top_keywords':       top_keywords,
        'keyword_weights':    keyword_weights,
        'reasoning_sentence': reasoning,
        'alternatives':       alternatives,
    }


def explain_anomaly(
    amount: float,
    category: str,
    history_amounts: list,
    z_score: Optional[float] = None,
) -> dict:
    """
    Human-readable explanation of why a transaction is anomalous.
    """
    if len(history_amounts) < 3:
        return {
            'is_anomaly':  False,
            'z_score':     None,
            'explanation': 'Not enough history to assess anomaly.',
        }

    arr  = np.array(history_amounts)
    mean = np.mean(arr)
    std  = np.std(arr) or 1.0
    z    = z_score if z_score is not None else abs((amount - mean) / std)

    is_anomaly = abs(z) > 2.0
    direction  = 'higher' if amount > mean else 'lower'
    multiple   = round(abs(amount / mean), 1) if mean > 0 else 0

    if is_anomaly:
        explanation = (
            f"₹{amount:,.0f} is {multiple}x {direction} than your usual "
            f"₹{mean:,.0f} on {category} "
            f"(Z-score: {z:.1f}, threshold: 2.0). "
            f"This transaction is statistically unusual."
        )
    else:
        explanation = (
            f"₹{amount:,.0f} is within normal range for {category}. "
            f"Your average is ₹{mean:,.0f} ± ₹{std:,.0f}."
        )

    return {
        'is_anomaly':    is_anomaly,
        'z_score':       round(float(z), 2),
        'amount':        amount,
        'category_mean': round(float(mean), 2),
        'category_std':  round(float(std), 2),
        'explanation':   explanation,
    }
