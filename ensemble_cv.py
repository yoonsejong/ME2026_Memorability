"""
ensemble_cv.py
--------------
5-fold cross-validation for an ensemble of configurations.
Each configuration is an independent (combo, model, selector, preprocessing)
triple.  The predictions are fused into one final prediction and the ensemble
Spearman is reported alongside each individual model's score.

Fusion
------
For each model, Spearman rho is measured against ground truth on the training
fold.  If rho < 0 the model's test predictions are negated before averaging,
so all models contribute in the correct rank direction.

Usage
-----
    python ensemble_cv.py
"""

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG  ── edit everything here ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

# Each dict needs: combo, model, selector, preprocessing.
# Optional: "label" (auto-generated from the other fields if absent).
#
# model        : LinearRegression | Ridge | MLP | SVR | GradientBoosting
# selector     : none | spearman_30 | spearman_50 | pca_30 | pca_50
# preprocessing: concat | kernel

CONFIGURATIONS =  [
    {"combo": "R3D",                               "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
    {"combo": "DenseNet121",                       "model": "GradientBoosting", "selector": "none",        "preprocessing": "concat"},
    {"combo": "EmotionSemantic",                   "model": "GradientBoosting", "selector": "spearman_50", "preprocessing": "kernel"},
]

N_FOLDS = 5
SEED    = 42

# ════════════════════════════════════════════════════════════════════════════════
# END CONFIG ─────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from feature_loader import ALL_FEATURE_SETS, FEATURE_GROUPS, build_feature_matrix, build_feature_matrices_separate
from feature_selector import CV_SELECTORS
from models import get_pipeline

# ── Feature combo lookup ──────────────────────────────────────────────────────

_ALL_COMBOS = {name: [name] for name in ALL_FEATURE_SETS}
_ALL_COMBOS.update(FEATURE_GROUPS)


# ── Kernel utilities ──────────────────────────────────────────────────────────

def _median_gamma(X: np.ndarray, max_samples: int = 500) -> float:
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(SEED)
        X   = X[rng.choice(X.shape[0], max_samples, replace=False)]
    D   = euclidean_distances(X)
    med = np.median(D[np.triu_indices_from(D, k=1)])
    return 1.0 / (2.0 * med ** 2) if med > 1e-10 else 1.0


def _build_kernel_fold(matrices, train_idx, test_idx, y_tr, selector: str):
    """Returns (K_train, K_test) for one CV fold."""
    select_fn = CV_SELECTORS[selector]
    kernels_tr, kernels_te = [], []
    for X_feat in matrices:
        X_tr = X_feat[train_idx]
        X_te = X_feat[test_idx]
        X_tr_sel, X_te_sel = select_fn(X_tr, X_te, y_tr)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sel)
        X_te_sc = scaler.transform(X_te_sel)
        gamma   = _median_gamma(X_tr_sc)
        kernels_tr.append(rbf_kernel(X_tr_sc, gamma=gamma))
        kernels_te.append(rbf_kernel(X_te_sc, X_tr_sc, gamma=gamma))
    return sum(kernels_tr), sum(kernels_te)


# ── Per-config prediction for one fold ───────────────────────────────────────

def _predict_one(cfg: dict, data_cache: dict, train_idx, test_idx, y_tr) -> tuple:
    """Returns (p_tr, p_te) for one configuration on one fold."""
    key  = cfg["_cache_key"]
    pipe = get_pipeline(cfg["model"])

    if cfg["preprocessing"] == "kernel":
        K_tr, K_te = _build_kernel_fold(data_cache[key]["matrices"],
                                         train_idx, test_idx, y_tr, cfg["selector"])
        pipe.fit(K_tr, y_tr)
        return pipe.predict(K_tr), pipe.predict(K_te)
    else:
        X = data_cache[key]["X"]
        X_tr, X_te = CV_SELECTORS[cfg["selector"]](X[train_idx], X[test_idx], y_tr)
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_tr), pipe.predict(X_te)


# ── Fusion ────────────────────────────────────────────────────────────────────

def _fuse(preds_te: list) -> np.ndarray:
    """Rank-average fusion: normalize each model's test predictions to [0,1] ranks and average."""
    ranked = [rankdata(p).astype(np.float32) / len(p) for p in preds_te]
    return np.mean(np.stack(ranked), axis=0)


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0 if np.isnan(r) else float(r)


# ── Main CV loop ──────────────────────────────────────────────────────────────

def _cfg_label(cfg: dict) -> str:
    return cfg.get("label") or (
        f"{cfg['combo']}/{cfg['model']}/{cfg['selector']}/{cfg['preprocessing']}"
    )


def run_ensemble_cv() -> None:
    # Validate and annotate configs
    for cfg in CONFIGURATIONS:
        combo = cfg["combo"]
        if combo not in _ALL_COMBOS:
            raise ValueError(f"Unknown combo '{combo}'. "
                             f"Available: {sorted(_ALL_COMBOS.keys())}")
        cfg["_feature_names"] = _ALL_COMBOS[combo]
        cfg["_cache_key"]     = (tuple(cfg["_feature_names"]), cfg["preprocessing"])

    # Load data for each unique (feature_names, preprocessing) pair
    print("Loading data...")
    data_cache: dict = {}
    for cfg in CONFIGURATIONS:
        key = cfg["_cache_key"]
        if key in data_cache:
            continue
        feature_names = cfg["_feature_names"]
        if cfg["preprocessing"] == "kernel":
            matrices, _, y = build_feature_matrices_separate(feature_names, "devset")
            data_cache[key] = {"matrices": matrices, "y": y}
        else:
            X_df, y_s = build_feature_matrix(feature_names, "devset")
            data_cache[key] = {"X": X_df.values.astype(np.float32),
                               "y": y_s.values.astype(np.float32)}
        n_videos = data_cache[key]['X'].shape[0] if 'X' in data_cache[key] else data_cache[key]['matrices'][0].shape[0]
        print(f"  loaded {cfg['combo']} / {cfg['preprocessing']}  ({n_videos} videos)")

    # Use y from the first config (all should align, inner-join handles differences)
    first_key = CONFIGURATIONS[0]["_cache_key"]
    y_all     = data_cache[first_key]["y"]
    n         = len(y_all)

    ENSEMBLE_LABEL = "Ensemble (rank average)"
    label_w = max(
        max(len(_cfg_label(cfg)) for cfg in CONFIGURATIONS),
        len(ENSEMBLE_LABEL),
    ) + 2  # small margin

    total_w = label_w + 36  # label + 4 stats columns
    print(f"\n{'='*total_w}")
    print(f"  Ensemble CV  ({N_FOLDS}-fold)  —  rank average")
    print(f"  Configurations  : {len(CONFIGURATIONS)}")
    for i, cfg in enumerate(CONFIGURATIONS, 1):
        print(f"    [{i}] {_cfg_label(cfg)}")
    print(f"{'='*total_w}")

    kf             = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ensemble_scores = {"spearman": [], "pearson": [], "mse": []}
    per_cfg_scores  = [{"spearman": [], "pearson": [], "mse": []} for _ in CONFIGURATIONS]
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n)), 1):
        print(f"\n  Fold {fold}/{N_FOLDS}")
        y_tr = y_all[train_idx]
        y_te = y_all[test_idx]

        preds_te = []
        for i, cfg in enumerate(CONFIGURATIONS):
            try:
                _, p_te = _predict_one(cfg, data_cache, train_idx, test_idx, y_tr)
            except Exception as exc:
                print(f"    [ERROR] config {i+1} ({_cfg_label(cfg)}): {exc}")
                p_te = np.zeros_like(y_te)

            rho_te, _ = spearmanr(y_te, p_te)
            rho_te    = 0.0 if np.isnan(rho_te) else float(rho_te)
            prs_te    = _safe_pearson(y_te, p_te)
            mse_te    = float(np.mean((y_te - p_te) ** 2))
            per_cfg_scores[i]["spearman"].append(rho_te)
            per_cfg_scores[i]["pearson"].append(prs_te)
            per_cfg_scores[i]["mse"].append(mse_te)
            preds_te.append(p_te)
            print(f"    [{i+1}] {_cfg_label(cfg):<{label_w}s}"
                  f"  Spearman = {rho_te:+.4f}  Pearson = {prs_te:+.4f}  MSE = {mse_te:.4f}")

        y_fused = _fuse(preds_te)
        rho_fused, _ = spearmanr(y_te, y_fused)
        rho_fused = 0.0 if np.isnan(rho_fused) else float(rho_fused)
        prs_fused = _safe_pearson(y_te, y_fused)
        mse_fused = float(np.mean((y_te - y_fused) ** 2))
        ensemble_scores["spearman"].append(rho_fused)
        ensemble_scores["pearson"].append(prs_fused)
        ensemble_scores["mse"].append(mse_fused)
        print(f"    => Ensemble  Spearman = {rho_fused:+.4f}  Pearson = {prs_fused:+.4f}  MSE = {mse_fused:.4f}")

    # Summary
    idx_w = len(str(len(CONFIGURATIONS))) + 3  # "[N] " prefix width
    col_w = label_w + idx_w

    def _summary_block(metric: str, scores_list: list, ensemble_sc: list) -> None:
        print(f"\n  {metric}")
        print(f"  {'Configuration':<{col_w}s}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}")
        print(f"  {'-'*col_w}  {'-------':>7}  {'------':>6}  {'-------':>7}  {'-------':>7}")
        fmt = ".4f" if metric == "MSE" else "+.4f"
        for i, cfg in enumerate(CONFIGURATIONS):
            sc     = scores_list[i][metric.lower()]
            prefix = f"[{i+1}] "
            print(f"  {prefix}{_cfg_label(cfg):<{col_w - len(prefix)}s}"
                  f"  {np.mean(sc):{fmt}}  {np.std(sc):.4f}"
                  f"  {np.min(sc):{fmt}}  {np.max(sc):{fmt}}")
        print(f"  {'─'*col_w}  {'-------':>7}  {'------':>6}  {'-------':>7}  {'-------':>7}")
        print(f"  {ENSEMBLE_LABEL:<{col_w}s}"
              f"  {np.mean(ensemble_sc):{fmt}}  {np.std(ensemble_sc):.4f}"
              f"  {np.min(ensemble_sc):{fmt}}  {np.max(ensemble_sc):{fmt}}")

    print(f"\n{'='*total_w}")
    print(f"  Summary")
    _summary_block("Spearman", per_cfg_scores, ensemble_scores["spearman"])
    _summary_block("Pearson",  per_cfg_scores, ensemble_scores["pearson"])
    _summary_block("MSE",      per_cfg_scores, ensemble_scores["mse"])
    print(f"\n{'='*total_w}")
    print(f"  Per-fold ensemble Spearman : {[round(s, 4) for s in ensemble_scores['spearman']]}")
    print(f"  Per-fold ensemble Pearson  : {[round(s, 4) for s in ensemble_scores['pearson']]}")
    print(f"  Per-fold ensemble MSE      : {[round(s, 4) for s in ensemble_scores['mse']]}")


if __name__ == "__main__":
    run_ensemble_cv()
