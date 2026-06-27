"""
analyze_result.py
-----------------
Reads results/sweep_results.csv and reports the utility of six feature types:
  tcnj_audio_o, tcnj_audio_c, Audio, tcnj_semantic, tcnj_emotion, EmotionSemantic

For each feature type the script splits every row in the CSV into two groups
— combinations that INCLUDE the feature type and those that DO NOT — then
reports the mean +/- std of mean_spearman for each group, the raw difference,
and a two-sample t-test p-value.

Membership logic
----------------
  Single features (tcnj_audio_o, tcnj_audio_c, tcnj_semantic, tcnj_emotion):
      row counts as "with" if that exact feature appears in the pipe-separated
      features column.
  Audio group:
      "with" if tcnj_audio_c OR tcnj_audio_o is present.
  EmotionSemantic group:
      "with" if BOTH tcnj_emotion AND tcnj_semantic are present.

Usage
-----
    python analyze_result.py
    python analyze_result.py --csv path/to/other.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

RESULTS_CSV = Path(__file__).parent / "results" / "sweep_results.csv"

# Each entry: (markers, require_all)
#   require_all=False  -> row is "with" if ANY marker is present (OR)
#   require_all=True   -> row is "with" if ALL markers are present (AND)
FEATURE_MARKERS = {
    "tcnj_audio_o":    (["tcnj_audio_o"],                  False),
    "tcnj_audio_c":    (["tcnj_audio_c"],                  False),
    "Audio":           (["tcnj_audio_c", "tcnj_audio_o"],  True),
    "tcnj_semantic":   (["tcnj_semantic"],                 False),
    "tcnj_emotion":    (["tcnj_emotion"],                  False),
    "EmotionSemantic": (["tcnj_emotion", "tcnj_semantic"], True),
}


def _has_feature(feature_str: str, markers: list, require_all: bool) -> bool:
    parts = set(feature_str.split("|"))
    return all(m in parts for m in markers) if require_all else any(m in parts for m in markers)


def analyze(csv_path: Path) -> None:
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found.\nRun run_sweep.py first.")

    df = pd.read_csv(csv_path)
    scores = df["mean_spearman"].values
    n_total = len(df)

    print(f"Loaded {n_total} rows from {csv_path.name}")
    print(f"Overall  mean_spearman: {scores.mean():+.4f}  std: {scores.std():.4f}\n")

    col = 18
    hdr = (f"  {'Feature':<{col}}  {'logic':^5}  {'n(w/o)':>6}  {'before (w/o)':^17}"
           f"  {'n(with)':>7}  {'after (with)':^17}  {'diff':>7}  p-value")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for feature_name, (markers, require_all) in FEATURE_MARKERS.items():
        logic = "AND" if require_all else "OR"
        mask  = df["features"].apply(lambda f, m=markers, r=require_all: _has_feature(f, m, r))

        with_scores    = df.loc[ mask, "mean_spearman"].values
        without_scores = df.loc[~mask, "mean_spearman"].values

        mean_with    = with_scores.mean()
        std_with     = with_scores.std()
        mean_without = without_scores.mean()
        std_without  = without_scores.std()
        diff         = mean_with - mean_without

        t_stat, p_val = ttest_ind(with_scores, without_scores, equal_var=False)

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "** "
        elif p_val < 0.05:
            sig = "*  "
        else:
            sig = "n.s"

        before_str = f"{mean_without:+.4f} +/- {std_without:.4f}"
        after_str  = f"{mean_with:+.4f} +/- {std_with:.4f}"
        print(f"  {feature_name:<{col}}  {logic:^5}  {len(without_scores):>6}  {before_str:^17}"
              f"  {len(with_scores):>7}  {after_str:^17}  {diff:+.4f}  {p_val:.4f} {sig}")

    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature usefulness from sweep results.")
    parser.add_argument("--csv", default=str(RESULTS_CSV),
                        help="Path to sweep_results.csv (default: results/sweep_results.csv)")
    args = parser.parse_args()
    analyze(Path(args.csv))
