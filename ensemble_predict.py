"""
ensemble_predict.py
-------------------
Train the ensemble on the full devset and generate testset predictions.
Uses the same CONFIGURATIONS and rank-average fusion as ensemble_cv.py.
Output: results/predictions/ensemble_YYYYMMDD_HHMMSS.csv
"""

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG  ── edit everything here ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

# Each dict needs: combo, model, selector, preprocessing.
# model        : LinearRegression | Ridge | MLP | SVR | GradientBoosting
# selector     : none | spearman_30 | spearman_50 | pca_30 | pca_50
# preprocessing: concat | kernel

CONFIGURATIONS = [
    {"combo": "Temporal",                          "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
    {"combo": "DenseNet121",                       "model": "GradientBoosting", "selector": "none",        "preprocessing": "concat"},
    {"combo": "EmotionSemantic",                   "model": "GradientBoosting", "selector": "spearman_50", "preprocessing": "kernel"},
]

SEED = 42

# ════════════════════════════════════════════════════════════════════════════════
# END CONFIG ─────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.preprocessing import StandardScaler

from feature_loader import ALL_FEATURE_SETS, FEATURE_GROUPS, build_feature_matrix, build_feature_matrices_separate
from feature_selector import FULL_SELECTORS
from models import get_pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT    = Path(__file__).parent
_OUT_DIR = _ROOT / "results" / "predictions"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def _build_kernel_full(dev_matrices, test_matrices, y_dev, selector: str):
    """Returns (K_dev, K_test)."""
    select_fn = FULL_SELECTORS[selector]
    kernels_dev, kernels_test = [], []
    for X_dev_feat, X_test_feat in zip(dev_matrices, test_matrices):
        X_dev_sel, transform_fn = select_fn(X_dev_feat, y_dev)
        X_test_sel = transform_fn(X_test_feat) if transform_fn is not None else X_test_feat
        scaler    = StandardScaler()
        X_dev_sc  = scaler.fit_transform(X_dev_sel)
        X_test_sc = scaler.transform(X_test_sel)
        gamma     = _median_gamma(X_dev_sc)
        kernels_dev.append(rbf_kernel(X_dev_sc, gamma=gamma))
        kernels_test.append(rbf_kernel(X_test_sc, X_dev_sc, gamma=gamma))
    return sum(kernels_dev), sum(kernels_test)


# ── Per-config predict ────────────────────────────────────────────────────────

def _predict_one(cfg: dict, data_cache: dict) -> tuple:
    """Returns (p_dev, p_test, test_ids)."""
    key  = cfg["_cache_key"]
    pipe = get_pipeline(cfg["model"])

    if cfg["preprocessing"] == "kernel":
        d = data_cache[key]
        K_dev, K_test = _build_kernel_full(d["dev_matrices"], d["test_matrices"],
                                            d["y_dev"], cfg["selector"])
        pipe.fit(K_dev, d["y_dev"])
        return pipe.predict(K_dev), pipe.predict(K_test), d["test_ids"]
    else:
        d = data_cache[key]
        select_fn = FULL_SELECTORS[cfg["selector"]]
        X_dev_sel, transform_fn = select_fn(d["X_dev"], d["y_dev"])
        X_test_sel = transform_fn(d["X_test"]) if transform_fn is not None else d["X_test"]
        pipe.fit(X_dev_sel, d["y_dev"])
        return pipe.predict(X_dev_sel), pipe.predict(X_test_sel), d["test_ids"]


# ── Fusion ────────────────────────────────────────────────────────────────────

def _fuse(preds_test: list) -> np.ndarray:
    """Rank-average fusion: normalize each model's test predictions to [0,1] ranks and average."""
    ranked = [rankdata(p).astype(np.float32) / len(p) for p in preds_test]
    return np.mean(np.stack(ranked), axis=0)


# ── Label helper ─────────────────────────────────────────────────────────────

def _cfg_label(cfg: dict) -> str:
    return cfg.get("label") or (
        f"{cfg['combo']}/{cfg['model']}/{cfg['selector']}/{cfg['preprocessing']}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ensemble(configurations: list) -> Optional[Path]:
    """Train ensemble on full devset and save timestamped testset predictions.

    Returns the path of the saved CSV.
    """
    for cfg in configurations:
        combo = cfg["combo"]
        if combo not in _ALL_COMBOS:
            raise ValueError(f"Unknown combo '{combo}'. Available: {sorted(_ALL_COMBOS.keys())}")
        cfg["_feature_names"] = _ALL_COMBOS[combo]
        cfg["_cache_key"]     = (tuple(cfg["_feature_names"]), cfg["preprocessing"])

    print("Loading data...")
    data_cache: dict = {}
    for cfg in configurations:
        key = cfg["_cache_key"]
        if key in data_cache:
            continue
        feature_names = cfg["_feature_names"]
        if cfg["preprocessing"] == "kernel":
            dev_mats, _, y_dev     = build_feature_matrices_separate(feature_names, "devset")
            test_mats, test_ids, _ = build_feature_matrices_separate(feature_names, "testset")
            data_cache[key] = {"dev_matrices": dev_mats, "test_matrices": test_mats,
                               "y_dev": y_dev, "test_ids": test_ids}
        else:
            X_dev_df, y_dev_s = build_feature_matrix(feature_names, "devset")
            X_test_df, _      = build_feature_matrix(feature_names, "testset")
            y_dev    = y_dev_s.values.astype(np.float32)
            test_ids = X_test_df.index.tolist()
            data_cache[key] = {"X_dev":    X_dev_df.values.astype(np.float32),
                               "X_test":   X_test_df.values.astype(np.float32),
                               "y_dev":    y_dev,
                               "test_ids": test_ids}
        print(f"  {cfg['combo']} / {cfg['preprocessing']}"
              f"  ({len(y_dev)} dev, {len(test_ids)} test)")

    first_key = configurations[0]["_cache_key"]
    y_dev_all = data_cache[first_key]["y_dev"]

    label_w = max(len(_cfg_label(cfg)) for cfg in configurations) + 2
    sep = "=" * (label_w + 22)
    print(f"\n{sep}")
    print(f"  Ensemble predict  ({len(configurations)} configurations)")
    for i, cfg in enumerate(configurations, 1):
        print(f"    [{i}] {_cfg_label(cfg)}")
    print(sep)

    preds_test, all_test_ids = [], None
    for i, cfg in enumerate(configurations):
        print(f"  [{i+1}] {_cfg_label(cfg):<{label_w}s}", end=" ", flush=True)
        try:
            p_dev, p_test, test_ids = _predict_one(cfg, data_cache)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue
        rho_val = spearmanr(y_dev_all, p_dev)[0]
        rho = 0.0 if np.isnan(rho_val) else float(rho_val)
        print(f"dev rho = {rho:+.4f}")
        preds_test.append(p_test)
        if all_test_ids is None:
            all_test_ids = test_ids

    if not preds_test:
        print("No predictions produced.")
        return None

    y_fused = _fuse(preds_test)

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = _OUT_DIR / f"ensemble_{ts}.csv"
    pd.DataFrame({"video": all_test_ids, "mem_score": y_fused}).to_csv(out_csv, index=False)

    print(f"\nSaved {len(all_test_ids)} predictions -> {out_csv}")
    return out_csv


def main() -> None:
    run_ensemble(CONFIGURATIONS)


if __name__ == "__main__":
    main()
