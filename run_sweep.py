"""
run_sweep.py
------------
Main entry point.  Tests every (feature combination, selector, model) triple
using 5-fold cross-validation on the devset and reports Spearman correlation.

Combinations tested
───────────────────
Individual feature sets (14):
  HSVHistogram, RGBHistogram, LBP, R3D,
  AlexNet, DenseNet121, EfficientNetB3, ResNet50, VGG, ViT,
  tcnj_audio_c, tcnj_audio_o,
  tcnj_emotion, tcnj_semantic

Group combinations (19):
  Single-modality  : HandCrafted, DeepVisual, Audio, Temporal, EmotionSemantic
  Two-modality     : HandCrafted+DeepVisual, HandCrafted+Audio,
                     HandCrafted+EmotionSemantic, DeepVisual+Audio,
                     DeepVisual+EmotionSemantic, Audio+EmotionSemantic
  Three-modality   : VisualAll (HC+T+DV), HandCrafted+DeepVisual+Audio,
                     HandCrafted+DeepVisual+EmotionSemantic,
                     HandCrafted+Audio+EmotionSemantic,
                     DeepVisual+Audio+EmotionSemantic
  Four-modality    : VisualAll+Audio, VisualAll+EmotionSemantic
  All              : every feature set combined

Models:    LinearRegression, Ridge, MLP (sklearn), SVR, GradientBoosting
Selectors: none, spearman_30, spearman_50, pca_30, pca_50

Usage:
    python run_sweep.py
    python run_sweep.py --selectors none spearman_30 spearman_50
    python run_sweep.py --models Ridge
    python run_sweep.py --combos ViT ResNet50 All
"""

import argparse
import json
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path

from feature_loader import ALL_FEATURE_SETS, FEATURE_GROUPS
from evaluate import evaluate_combo
from kernel_fusion import evaluate_combo_kernel

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV  = RESULTS_DIR / "sweep_results.csv"
SWEEP_LOG   = RESULTS_DIR / "sweep_log.json"
PARTIAL_DIR = RESULTS_DIR / "partial"

DEFAULT_MODELS    = ["LinearRegression", "Ridge", "MLP", "SVR", "GradientBoosting"]
DEFAULT_SELECTORS = ["none", "spearman_30", "spearman_50", "pca_30", "pca_50"]

# All combinations: individual feature sets first, then groups
ALL_COMBINATIONS: dict = {name: [name] for name in ALL_FEATURE_SETS}
ALL_COMBINATIONS.update(FEATURE_GROUPS)


# ── Worker (module-level for joblib pickling) ─────────────────────────────────

def _run_task(task_type: str, combo_name: str, feat_list: list,
              model: str, selector: str, partial_dir: Path) -> None:
    """Runs one experiment and atomically saves its result to a per-task CSV."""
    if task_type == "kernel":
        out = partial_dir / f"{combo_name}__KF_{model}__{selector}.csv"
        try:
            result = evaluate_combo_kernel(combo_name, feat_list, selector, model)
        except Exception as exc:
            print(f"  [ERROR] {combo_name}/KF_{model}/{selector}: {exc}")
            return
    else:
        out = partial_dir / f"{combo_name}__{model}__{selector}.csv"
        try:
            result = evaluate_combo(combo_name, feat_list, selector, model)
        except Exception as exc:
            print(f"  [ERROR] {combo_name}/{model}/{selector}: {exc}")
            return
    tmp = out.with_suffix(".tmp")
    pd.DataFrame([result]).to_csv(tmp, index=False)
    tmp.replace(out)


# ── Summary printing ──────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    cols_top  = ["combination", "model", "selector", "n_features",
                 "mean_spearman", "std_spearman"]
    cols_best = ["combination", "model", "selector", "mean_spearman"]

    print("\n-- Top 10 configurations ------------------------------------------")
    print(df.head(10)[cols_top].to_string(index=False))

    print("\n-- Best configuration per combination -----------------------------")
    best = (
        df.loc[df.groupby("combination")["mean_spearman"].idxmax()]
          .sort_values("mean_spearman", ascending=False)
    )
    print(best[cols_best].to_string(index=False))


# ── Main sweep ────────────────────────────────────────────────────────────────

def main(combos: list, models: list, selectors: list, fusion: str = "both", n_jobs: int = -1) -> None:
    cpu_count = os.cpu_count() or 1
    n_jobs = min(n_jobs, cpu_count) if n_jobs > 0 else cpu_count

    # ── Resume or fresh start ─────────────────────────────────────────────────
    if SWEEP_LOG.exists():
        log       = json.loads(SWEEP_LOG.read_text())
        combos    = log["combos"]
        models    = log["models"]
        selectors = log["selectors"]
        fusion    = log["fusion"]
        print(f"[RESUME] Previous run from {log['timestamp']} — restoring parameters.")
        PARTIAL_DIR.mkdir(exist_ok=True)
    else:
        shutil.rmtree(PARTIAL_DIR, ignore_errors=True)
        PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
        SWEEP_LOG.write_text(json.dumps({
            "timestamp": pd.Timestamp.now().isoformat(),
            "fusion":    fusion,
            "combos":    list(combos),
            "models":    list(models),
            "selectors": list(selectors),
        }, indent=2))
        print(f"[LOG] Sweep log written -> {SWEEP_LOG}")

    # ── Resolve combinations ──────────────────────────────────────────────────
    run_combos = {}
    for name in combos:
        if name not in ALL_COMBINATIONS:
            print(f"[WARNING] Unknown combination '{name}' - skipping")
            continue
        run_combos[name] = ALL_COMBINATIONS[name]

    # ── Build pending task list (skip tasks that already have a result file) ──
    tasks = []
    if fusion in ("kernel", "both"):
        for combo_name, feat_list in run_combos.items():
            for model in models:
                for selector in selectors:
                    p = PARTIAL_DIR / f"{combo_name}__KF_{model}__{selector}.csv"
                    if not p.exists():
                        tasks.append(("kernel", combo_name, feat_list, model, selector))
    if fusion in ("concat", "both"):
        for combo_name, feat_list in run_combos.items():
            for model in models:
                for selector in selectors:
                    p = PARTIAL_DIR / f"{combo_name}__{model}__{selector}.csv"
                    if not p.exists():
                        tasks.append(("concat", combo_name, feat_list, model, selector))

    already_done = len(list(PARTIAL_DIR.glob("*.csv")))
    if already_done:
        print(f"[RESUME] Skipping {already_done} already-completed tasks.")
    print(f"Running {len(tasks)} tasks  [n_jobs={n_jobs}]")

    Parallel(n_jobs=n_jobs)(
        delayed(_run_task)(tt, cn, fl, m, s, PARTIAL_DIR)
        for tt, cn, fl, m, s in tasks
    )

    # ── Merge all partial files ───────────────────────────────────────────────
    partial_files = sorted(PARTIAL_DIR.glob("*.csv"))
    if not partial_files:
        print("No results to save.")
        return

    df = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)
    df = df.sort_values("mean_spearman", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nSaved -> {OUTPUT_CSV}")
    _print_summary(df)

    shutil.rmtree(PARTIAL_DIR)
    SWEEP_LOG.unlink(missing_ok=True)
    print("[LOG] Partial results and log cleaned up.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature combination sweep for memorability prediction."
    )
    parser.add_argument(
        "--combos", nargs="+", default=list(ALL_COMBINATIONS.keys()),
        help="Combination names to test (default: all)",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        choices=["LinearRegression", "Ridge", "MLP", "SVR", "GradientBoosting"],
        help="Models to evaluate (default: LinearRegression Ridge MLP SVR GradientBoosting)",
    )
    parser.add_argument(
        "--selectors", nargs="+", default=DEFAULT_SELECTORS,
        choices=["none", "spearman_30", "spearman_50", "pca_30", "pca_50"],
        help="Feature selectors to apply (default: all five)",
    )
    parser.add_argument(
        "--fusion", default="both", choices=["concat", "kernel", "both"],
        help="Feature fusion strategy: concat, kernel, or both (default)",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=-1,
        help="Parallel workers (default: all cores, capped at cpu_count)",
    )
    args = parser.parse_args()
    main(args.combos, args.models, args.selectors, args.fusion, args.jobs)
