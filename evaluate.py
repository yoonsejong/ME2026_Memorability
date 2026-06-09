"""
evaluate.py
-----------
Cross-validation evaluation for (feature combination, selector, model) triples.

Metric: Spearman rank correlation between predicted and actual memorability
        scores, averaged over 5 CV folds.

Feature selection is applied *inside* each fold using only the training
partition, preventing any data leakage into validation scores.
"""

import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from feature_loader import build_feature_matrix
from feature_selector import CV_SELECTORS
from models import get_pipeline

N_FOLDS      = 5
RANDOM_STATE = 42


# ── Core CV loop ──────────────────────────────────────────────────────────────

def cv_spearman(
    X: np.ndarray,
    y: np.ndarray,
    selector_name: str,
    model_name: str,
) -> dict:
    """
    Runs N_FOLDS cross-validation.

    Feature selection (selector_name) is applied per fold:
      1. Fit selector on X_train (of that fold).
      2. Apply same indices to X_test.
      3. Fit a fresh pipeline on X_train_sel; predict X_test_sel.

    Parameters
    ----------
    X             : float32 array of shape (n_samples, n_features)
    y             : float32 array of shape (n_samples,)
    selector_name : key in CV_SELECTORS
    model_name    : key in MODEL_FACTORIES

    Returns
    -------
    dict with mean_spearman, std_spearman, min_spearman, max_spearman
    """
    if selector_name not in CV_SELECTORS:
        raise ValueError(f"Unknown selector '{selector_name}'. "
                         f"Choose from {list(CV_SELECTORS)}")

    selector = CV_SELECTORS[selector_name]
    kf       = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores   = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr_sel, X_te_sel = selector(X_tr, X_te, y_tr)

        pipe = get_pipeline(model_name)
        pipe.fit(X_tr_sel, y_tr)
        y_pred = pipe.predict(X_te_sel)

        rho, _ = spearmanr(y_te, y_pred)
        scores.append(0.0 if np.isnan(rho) else float(rho))

    return {
        "mean_spearman": round(float(np.mean(scores)), 4),
        "std_spearman":  round(float(np.std(scores)),  4),
        "min_spearman":  round(float(np.min(scores)),  4),
        "max_spearman":  round(float(np.max(scores)),  4),
    }


# ── High-level convenience wrapper ───────────────────────────────────────────

def evaluate_combo(
    combo_name: str,
    feature_names: list,
    selector_name: str,
    model_name: str,
    verbose: bool = True,
) -> dict:
    """
    Loads features for the given combination, runs CV, and returns a result row.

    Parameters
    ----------
    combo_name    : human-readable label for this combination
    feature_names : list of feature set names (see ALL_FEATURE_SETS)
    selector_name : key in CV_SELECTORS
    model_name    : key in MODEL_FACTORIES
    verbose       : print one-line progress update if True

    Returns
    -------
    dict suitable for appending to a results DataFrame
    """
    X_df, y = build_feature_matrix(feature_names, split="devset")
    X       = X_df.values.astype(np.float32)
    y_arr   = y.values.astype(np.float32)

    scores  = cv_spearman(X, y_arr, selector_name, model_name)

    result = {
        "combination": combo_name,
        "features":    "|".join(feature_names),
        "n_features":  X.shape[1],
        "n_samples":   len(y_arr),
        "selector":    selector_name,
        "model":       model_name,
        **scores,
    }

    if verbose:
        print(
            f"  {combo_name:<25s} | {model_name:<18s} | {selector_name:<14s} | "
            f"n_feat={X.shape[1]:>6d} | "
            f"Spearman={scores['mean_spearman']:.4f} ± {scores['std_spearman']:.4f}"
        )

    return result
