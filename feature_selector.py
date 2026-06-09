"""
feature_selector.py
-------------------
Feature selection strategies used during cross-validation and for
fitting the final model on the full devset.

CV selectors receive (X_train, X_test, y_train) and return
(X_train_selected, X_test_selected).  Selection indices are computed
solely from the training fold so there is no data leakage.

Full-dataset selectors receive (X, y) and return
(X_selected, transform_fn_or_None).  Use these when training a final model
on all devset data before predicting the testset.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spearman_indices(X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Returns indices of the top-n features by |Spearman r| with y.
    Constant features get correlation 0 and are ranked last.
    """
    n_feat = X.shape[1]
    corrs  = np.zeros(n_feat)
    for i in range(n_feat):
        col = X[:, i]
        if np.std(col) > 0:
            val = spearmanr(col, y)[0]
            corrs[i] = 0.0 if np.isnan(val) else abs(val)
    keep = min(n, n_feat)
    return np.argsort(corrs)[-keep:]   # ascending, so last = highest


# ── CV selectors  (X_train, X_test, y_train) -> (X_tr_sel, X_te_sel) ─────────

def cv_none(X_train, X_test, y_train):
    """No selection – pass data through unchanged."""
    return X_train, X_test


def cv_spearman(X_train, X_test, y_train, n: int = 30):
    """Top-n features by |Spearman r| computed on the training fold only."""
    idx = _spearman_indices(X_train, y_train, n)
    return X_train[:, idx], X_test[:, idx]


def cv_pca(X_train, X_test, y_train, n: int = 50):
    """PCA fitted on training fold only, applied to both folds."""
    n_comp = min(n, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X_train), pca.transform(X_test)


# ── Full-dataset selectors  (X, y) -> (X_sel, transform_fn_or_None) ─────────

def full_none(X: np.ndarray, y: np.ndarray):
    """No selection. Returns (X, None)."""
    return X, None


def full_spearman(X: np.ndarray, y: np.ndarray, n: int = 30):
    """Top-n features on the whole dataset. Returns (X_sel, transform_fn)."""
    idx = _spearman_indices(X, y, n)
    return X[:, idx], lambda X_new: X_new[:, idx]


def full_pca(X: np.ndarray, y: np.ndarray, n: int = 50):
    """PCA on the whole dataset. Returns (X_pca, transform_fn)."""
    n_comp = min(n, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X), pca.transform


# ── Public registries ─────────────────────────────────────────────────────────

CV_SELECTORS: dict = {
    "none":        lambda Xtr, Xte, y: cv_none(Xtr, Xte, y),
    "spearman_30": lambda Xtr, Xte, y: cv_spearman(Xtr, Xte, y, n=30),
    "spearman_50": lambda Xtr, Xte, y: cv_spearman(Xtr, Xte, y, n=50),
    "pca_30":      lambda Xtr, Xte, y: cv_pca(Xtr, Xte, y, n=30),
    "pca_50":      lambda Xtr, Xte, y: cv_pca(Xtr, Xte, y, n=50),
}

FULL_SELECTORS: dict = {
    "none":        lambda X, y: full_none(X, y),
    "spearman_30": lambda X, y: full_spearman(X, y, n=30),
    "spearman_50": lambda X, y: full_spearman(X, y, n=50),
    "pca_30":      lambda X, y: full_pca(X, y, n=30),
    "pca_50":      lambda X, y: full_pca(X, y, n=50),
}
