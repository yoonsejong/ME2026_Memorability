"""
kernel_fusion.py
----------------
Multiple Kernel Learning (MKL) fusion: one RBF kernel per feature set,
combined with a parametric weighted sum.

    K_fused = w_1 * K_1 + w_2 * K_2 + ... + w_m * K_m

Default weights are 1.0 per feature set (uniform).

Leakage controls (all paths):
  - StandardScaler fitted on training fold / full devset only.
  - RBF gamma estimated from (scaled) training data via median heuristic.
  - Feature selection (Spearman / PCA) fitted on training fold only.
  - Prediction uses cross-kernel K(X_test, X_train), not K(X_test, X_test).
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from feature_loader import build_feature_matrices_separate
from feature_selector import CV_SELECTORS, FULL_SELECTORS
from models import get_pipeline

N_FOLDS      = 5
RANDOM_STATE = 42


# -- Kernel utilities ---------------------------------------------------------

def _median_gamma(X: np.ndarray, max_samples: int = 500) -> float:
    """
    RBF bandwidth via median pairwise distance heuristic:
        gamma = 1 / (2 * median_dist^2)
    Subsamples up to max_samples rows to cap O(n^2 * d) cost.
    """
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        X   = X[rng.choice(X.shape[0], max_samples, replace=False)]
    D   = euclidean_distances(X)
    med = np.median(D[np.triu_indices_from(D, k=1)])
    return 1.0 / (2.0 * med ** 2) if med > 1e-10 else 1.0


def compute_rbf_kernels(
    X_train: np.ndarray,
    X_test:  np.ndarray = None,
    gamma:   float      = None,
) -> tuple:
    """
    Computes RBF kernel(s) for one feature set.

    gamma is estimated from X_train when None -- no leakage into test data.

    Returns
    -------
    K_train : (n_train, n_train) symmetric kernel
    K_test  : (n_test, n_train)  cross-kernel; None if X_test is None
    gamma   : float used
    """
    if gamma is None:
        gamma = _median_gamma(X_train)
    K_train = rbf_kernel(X_train, gamma=gamma)
    K_test  = rbf_kernel(X_test, X_train, gamma=gamma) if X_test is not None else None
    return K_train, K_test, gamma


def combine_kernels(kernel_list: list, weights=None) -> np.ndarray:
    """
    Weighted sum of kernel matrices.

    Parameters
    ----------
    kernel_list : list of np.ndarray, all same shape
    weights     : list of floats; None -> all 1.0 (uniform)
    """
    if weights is None:
        weights = [1.0] * len(kernel_list)
    if len(weights) != len(kernel_list):
        raise ValueError("len(weights) must equal len(kernel_list)")
    if not kernel_list:
        raise ValueError("kernel_list must not be empty")
    result = weights[0] * kernel_list[0]
    for w, K in zip(weights[1:], kernel_list[1:]):
        result = result + w * K
    return result


# -- Per-fold preprocessing (select -> scale -> kernel) -----------------------

def _fold_kernels(matrices, train_idx, test_idx, y_tr, selector, weights):
    """
    Builds combined training and cross kernels for one CV fold.

    For each feature set:
      1. Feature selection fitted on training fold only.
      2. StandardScaler fitted on training fold only.
      3. RBF gamma estimated from scaled training fold only.
      4. Cross-kernel K(X_test, X_train) for prediction.

    Returns (K_comb_train, K_comb_test).
    """
    kernels_tr = []
    kernels_te = []

    for X_feat in matrices:
        X_tr = X_feat[train_idx]
        X_te = X_feat[test_idx]

        # 1. Feature selection -- indices/transform from X_tr only
        X_tr_sel, X_te_sel = selector(X_tr, X_te, y_tr)

        # 2. Scaling -- scaler fitted on X_tr_sel only
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sel)
        X_te_sc = scaler.transform(X_te_sel)

        # 3. Kernel -- gamma estimated from X_tr_sc only
        K_tr, K_te, _ = compute_rbf_kernels(X_tr_sc, X_te_sc)
        kernels_tr.append(K_tr)
        kernels_te.append(K_te)

    return combine_kernels(kernels_tr, weights), combine_kernels(kernels_te, weights)


# -- CV evaluation ------------------------------------------------------------

def evaluate_combo_kernel(
    combo_name:    str,
    feature_names: list,
    selector_name: str,
    model_name:    str,
    weights=None,
    verbose: bool = True,
) -> dict:
    """
    5-fold CV with per-feature-set RBF kernels fused by weighted sum.
    The fused kernel matrix is used as the feature matrix for model_name.

    Parameters
    ----------
    combo_name    : label for this combination (for CSV and console output)
    feature_names : list of feature set names
    selector_name : key in CV_SELECTORS ("none", "spearman_30", "pca_50", ...)
    model_name    : key in MODEL_FACTORIES
    weights       : per-kernel weights; None = all 1.0

    Returns
    -------
    dict suitable for sweep results DataFrame
    """
    if selector_name not in CV_SELECTORS:
        raise ValueError(f"Unknown selector {selector_name!r}. "
                         f"Choose from {list(CV_SELECTORS)}")

    matrices, _ids, y = build_feature_matrices_separate(feature_names, "devset")
    selector = CV_SELECTORS[selector_name]
    n        = len(y)

    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, test_idx in kf.split(np.arange(n)):
        y_tr = y[train_idx]
        y_te = y[test_idx]

        K_tr, K_te = _fold_kernels(matrices, train_idx, test_idx, y_tr, selector, weights)

        pipe = get_pipeline(model_name)
        pipe.fit(K_tr, y_tr)
        y_pred = pipe.predict(K_te)

        rho, _ = spearmanr(y_te, y_pred)
        scores.append(0.0 if np.isnan(rho) else float(rho))

    stats = {
        "mean_spearman": round(float(np.mean(scores)), 4),
        "std_spearman":  round(float(np.std(scores)),  4),
        "min_spearman":  round(float(np.min(scores)),  4),
        "max_spearman":  round(float(np.max(scores)),  4),
    }
    n_feat_total = sum(m.shape[1] for m in matrices)

    if verbose:
        print(
            f"  {combo_name:<25s} | KF_{model_name:<18s} | {selector_name:<14s} | "
            f"n_feat={n_feat_total:>6d} | "
            f"Spearman={stats['mean_spearman']:.4f} +- {stats['std_spearman']:.4f}"
        )

    return {
        "combination": combo_name,
        "features":    "|".join(feature_names),
        "n_features":  n_feat_total,
        "n_samples":   n,
        "selector":    selector_name,
        "model":       f"KF_{model_name}",
        **stats,
    }


# -- Testset prediction -------------------------------------------------------

def train_and_predict_kernel(
    feature_names: list,
    selector_name: str,
    model_name:    str,
    weights=None,
) -> tuple:
    """
    Trains kernel fusion on the full devset and predicts the testset.
    The fused kernel matrix is used as the feature matrix for model_name.

    For each feature set:
      1. Feature selection fitted on full devset only (leakage-free wrt testset).
      2. StandardScaler fitted on full devset only.
      3. RBF gamma estimated from scaled devset only.
      4. Cross-kernel K(X_test, X_dev) used for prediction.

    Parameters
    ----------
    feature_names : list of feature set names
    selector_name : key in FULL_SELECTORS
    model_name    : key in MODEL_FACTORIES
    weights       : per-kernel weights; None = all 1.0

    Returns
    -------
    (test_ids, y_pred) : list of video IDs, np.ndarray of predictions
    """
    if selector_name not in FULL_SELECTORS:
        raise ValueError(f"Unknown selector {selector_name!r}. "
                         f"Choose from {list(FULL_SELECTORS)}")

    dev_matrices,  _,        y        = build_feature_matrices_separate(feature_names, "devset")
    test_matrices, test_ids, _        = build_feature_matrices_separate(feature_names, "testset")
    selector_fn = FULL_SELECTORS[selector_name]

    kernels_dev  = []
    kernels_test = []

    for X_dev_feat, X_test_feat in zip(dev_matrices, test_matrices):
        # 1. Feature selection on full devset
        X_dev_sel, transform_fn = selector_fn(X_dev_feat, y)
        X_test_sel = transform_fn(X_test_feat) if transform_fn is not None else X_test_feat

        # 2. Scaling -- scaler fitted on devset only
        scaler    = StandardScaler()
        X_dev_sc  = scaler.fit_transform(X_dev_sel)
        X_test_sc = scaler.transform(X_test_sel)

        # 3. Kernel -- gamma estimated from devset only
        K_dev, K_test, _ = compute_rbf_kernels(X_dev_sc, X_test_sc)
        kernels_dev.append(K_dev)
        kernels_test.append(K_test)

    K_dev_comb  = combine_kernels(kernels_dev,  weights)
    K_test_comb = combine_kernels(kernels_test, weights)

    pipe = get_pipeline(model_name)
    pipe.fit(K_dev_comb, y)
    y_pred = pipe.predict(K_test_comb)

    return test_ids, y_pred
