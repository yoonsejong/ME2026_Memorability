"""
batch_ensemble_predict.py
--------------------------
Run multiple ensemble prediction passes, one per entry in CONFIGURATIONS_LIST.
Each entry is an independent list of model configurations passed to run_ensemble().
A separate timestamped CSV is saved for each run.

Add or remove lists in CONFIGURATIONS_LIST to control which ensembles are generated.
"""

from ensemble_predict import run_ensemble

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATIONS_LIST  ── edit below ─────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════
#
# Each entry is a list of dicts with keys: combo, model, selector, preprocessing.
#   model        : LinearRegression | Ridge | MLP | SVR | GradientBoosting
#   selector     : none | spearman_30 | spearman_50 | pca_30 | pca_50
#   preprocessing: concat | kernel

CONFIGURATIONS_LIST = [
    # Run 1 
    [
        {"combo": "R3D",                               "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
        {"combo": "DenseNet121",                       "model": "GradientBoosting", "selector": "none",        "preprocessing": "concat"},
        {"combo": "EmotionSemantic",                   "model": "GradientBoosting", "selector": "spearman_50", "preprocessing": "kernel"},
    ],
    # Run 2 
    [
        {"combo": "R3D",                               "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
        {"combo": "DenseNet121",                       "model": "GradientBoosting", "selector": "none",        "preprocessing": "concat"},
        {"combo": "Audio",                             "model": "MLP",              "selector": "spearman_50", "preprocessing": "concat"},
    ],
    # Run 3 
    [
        {"combo": "All",                               "model": "SVR",              "selector": "none",        "preprocessing": "kernel"},
        {"combo": "HandCrafted+Audio+EmotionSemantic", "model": "SVR",              "selector": "spearman_50", "preprocessing": "kernel"},
        {"combo": "R3D",                               "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
        {"combo": "Audio",                             "model": "MLP",              "selector": "spearman_50", "preprocessing": "concat"},
        {"combo": "HandCrafted+Audio+EmotionSemantic", "model": "Ridge",            "selector": "none",        "preprocessing": "concat"},
        {"combo": "DenseNet121",                       "model": "GradientBoosting", "selector": "none",        "preprocessing": "concat"},
    ],
    # Run 4
    [
        {"combo": "All",                               "model": "SVR",              "selector": "none",        "preprocessing": "kernel"},
        {"combo": "HandCrafted+Audio+EmotionSemantic", "model": "SVR",              "selector": "spearman_50", "preprocessing": "kernel"},
        {"combo": "R3D",                               "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
        {"combo": "Audio",                             "model": "MLP",              "selector": "spearman_50", "preprocessing": "concat"},
        {"combo": "HandCrafted+Audio+EmotionSemantic", "model": "Ridge",            "selector": "none",        "preprocessing": "concat"},
        {"combo": "DeepVisual+Audio",                  "model": "SVR",              "selector": "spearman_30", "preprocessing": "kernel"},
    ],
    # Run 5
    [
        {"combo": "All",                               "model": "SVR",              "selector": "none",        "preprocessing": "kernel"},
        {"combo": "HandCrafted+Audio+EmotionSemantic", "model": "SVR",              "selector": "spearman_50", "preprocessing": "kernel"},
        {"combo": "R3D",                               "model": "SVR",              "selector": "pca_50",      "preprocessing": "kernel"},
        {"combo": "Audio",                             "model": "MLP",              "selector": "spearman_50", "preprocessing": "concat"},
        {"combo": "HandCrafted+Audio+EmotionSemantic", "model": "Ridge",            "selector": "none",        "preprocessing": "concat"},
        {"combo": "DenseNet121",                       "model": "GradientBoosting", "selector": "none",        "preprocessing": "concat"},
        {"combo": "DeepVisual+Audio",                  "model": "SVR",              "selector": "spearman_30", "preprocessing": "kernel"},
    ],
]

# ════════════════════════════════════════════════════════════════════════════════
# END CONFIGURATIONS_LIST ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    n = len(CONFIGURATIONS_LIST)
    print(f"batch_ensemble_predict: {n} ensemble run(s)\n")
    for idx, configurations in enumerate(CONFIGURATIONS_LIST, 1):
        print(f"{'='*60}")
        print(f"  Run {idx}/{n}  ({len(configurations)} configurations)")
        print(f"{'='*60}")
        out_csv = run_ensemble(configurations)
        if out_csv:
            print(f"  -> {out_csv}\n")
        else:
            print(f"  -> run {idx} produced no output\n")
    print("Done.")


if __name__ == "__main__":
    main()
