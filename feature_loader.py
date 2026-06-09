"""
feature_loader.py
-----------------
Loads .npy feature files from devset/features or testset/features,
flattens them, and builds aligned feature matrices keyed by video ID.

Feature sets available (devset & testset):
  HandCrafted visual : HSVHistogram, RGBHistogram, LBP
  Temporal           : R3D
  Deep visual        : AlexNet, DenseNet121, EfficientNetB3, ResNet50, VGG, ViT
  Audio              : tcnj_audio_c, tcnj_audio_o
  Emotion            : tcnj_emotion  (shape 19)
  Semantic           : tcnj_semantic (shape 7)

Shapes after flattening:
  (3, D) arrays -> 3*D  (three-frame features padded/truncated to 3 rows)
  (D,)   arrays -> D    (global/1-D features)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT        = Path(__file__).parent
DEVSET_DIR  = ROOT / "devset"
TESTSET_DIR = ROOT / "testset"
TARGET_COL  = "memorability_score"

ALL_FEATURE_SETS = [
    "HSVHistogram", "RGBHistogram", "LBP",
    "R3D",
    "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
    "tcnj_audio_c", "tcnj_audio_o",
    "tcnj_emotion", "tcnj_semantic",
]

# Predefined meaningful combinations (name -> list of feature sets)
FEATURE_GROUPS = {
    # ── Single-modality groups ────────────────────────────────────────────────
    "HandCrafted":    ["HSVHistogram", "RGBHistogram", "LBP"],
    "DeepVisual":     ["AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT"],
    "Audio":          ["tcnj_audio_c", "tcnj_audio_o"],
    "Temporal":       ["R3D"],
    "EmotionSemantic":["tcnj_emotion", "tcnj_semantic"],
    # ── Two-modality groups ───────────────────────────────────────────────────
    "HandCrafted+DeepVisual":        ["HSVHistogram", "RGBHistogram", "LBP",
                                      "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT"],
    "HandCrafted+Audio":             ["HSVHistogram", "RGBHistogram", "LBP",
                                      "tcnj_audio_c", "tcnj_audio_o"],
    "HandCrafted+EmotionSemantic":   ["HSVHistogram", "RGBHistogram", "LBP",
                                      "tcnj_emotion", "tcnj_semantic"],
    "DeepVisual+Audio":              ["AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                      "tcnj_audio_c", "tcnj_audio_o"],
    "DeepVisual+EmotionSemantic":    ["AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                      "tcnj_emotion", "tcnj_semantic"],
    "Audio+EmotionSemantic":         ["tcnj_audio_c", "tcnj_audio_o",
                                      "tcnj_emotion", "tcnj_semantic"],
    # ── Three-modality groups ─────────────────────────────────────────────────
    "VisualAll":                             ["HSVHistogram", "RGBHistogram", "LBP", "R3D",
                                              "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT"],
    "HandCrafted+DeepVisual+Audio":          ["HSVHistogram", "RGBHistogram", "LBP",
                                              "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                              "tcnj_audio_c", "tcnj_audio_o"],
    "HandCrafted+DeepVisual+EmotionSemantic":["HSVHistogram", "RGBHistogram", "LBP",
                                              "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                              "tcnj_emotion", "tcnj_semantic"],
    "HandCrafted+Audio+EmotionSemantic":     ["HSVHistogram", "RGBHistogram", "LBP",
                                              "tcnj_audio_c", "tcnj_audio_o",
                                              "tcnj_emotion", "tcnj_semantic"],
    "DeepVisual+Audio+EmotionSemantic":      ["AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                              "tcnj_audio_c", "tcnj_audio_o",
                                              "tcnj_emotion", "tcnj_semantic"],
    # ── Four-modality groups ──────────────────────────────────────────────────
    "VisualAll+Audio":              ["HSVHistogram", "RGBHistogram", "LBP", "R3D",
                                     "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                     "tcnj_audio_c", "tcnj_audio_o"],
    "VisualAll+EmotionSemantic":    ["HSVHistogram", "RGBHistogram", "LBP", "R3D",
                                     "AlexNet", "DenseNet121", "EfficientNetB3", "ResNet50", "VGG", "ViT",
                                     "tcnj_emotion", "tcnj_semantic"],
    # ── All ───────────────────────────────────────────────────────────────────
    "All":                          list(ALL_FEATURE_SETS),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _flatten(arr: np.ndarray, max_rows: int = 3) -> np.ndarray:
    """Flatten a (rows, cols) array to 1-D, padding to max_rows if needed."""
    if arr.ndim == 1:
        return arr.astype(np.float32)
    rows, cols = arr.shape
    if rows < max_rows:
        arr = np.vstack([arr, np.zeros((max_rows - rows, cols))])
    return arr[:max_rows].flatten().astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def load_feature_set(feature_name: str, split: str = "devset") -> pd.DataFrame:
    """
    Loads and flattens all .npy files for one feature set.

    Parameters
    ----------
    feature_name : one of ALL_FEATURE_SETS
    split        : "devset" or "testset"

    Returns
    -------
    pd.DataFrame  indexed by video ID, columns named <feature_name>_0, _1, …
    """
    base   = DEVSET_DIR if split == "devset" else TESTSET_DIR
    folder = base / "features" / feature_name
    if not folder.exists():
        raise FileNotFoundError(f"Feature folder not found: {folder}")

    rows = {}
    for fpath in sorted(folder.glob("*.npy")):
        arr = np.nan_to_num(np.load(fpath), nan=0.0, posinf=0.0, neginf=0.0)
        rows[fpath.stem] = _flatten(arr)

    if not rows:
        raise FileNotFoundError(f"No .npy files in {folder}")

    n_cols = len(next(iter(rows.values())))
    cols   = [f"{feature_name}_{i}" for i in range(n_cols)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def load_labels() -> pd.Series:
    """Returns memorability_score for devset videos, indexed by video ID."""
    df = pd.read_csv(DEVSET_DIR / "devset_videolist_GT.csv", index_col="id")
    return df[TARGET_COL].astype(np.float32)


def build_feature_matrices_separate(
    feature_names: list,
    split: str = "devset",
) -> tuple:
    """
    Loads feature sets as separate aligned matrices (not concatenated).

    Returns
    -------
    matrices  : list of np.float32 arrays, one per feature name, shape (n_videos, D_i)
    video_ids : list of video ID strings (the shared aligned index)
    y_arr     : np.float32 label array (devset) or None (testset)
    """
    frames = [load_feature_set(name, split) for name in feature_names]

    common_idx = frames[0].index
    for df in frames[1:]:
        common_idx = common_idx.intersection(df.index)

    if split == "devset":
        y          = load_labels()
        common_idx = common_idx.intersection(y.index)
        y_arr      = y.loc[common_idx].values.astype(np.float32)
    else:
        y_arr = None

    matrices  = [df.loc[common_idx].values.astype(np.float32) for df in frames]
    video_ids = common_idx.tolist()
    return matrices, video_ids, y_arr


def build_feature_matrix(
    feature_names: list,
    split: str = "devset",
) -> tuple:
    """
    Inner-joins the requested feature sets and aligns with labels.

    Parameters
    ----------
    feature_names : list of feature set names to concatenate
    split         : "devset" or "testset"

    Returns
    -------
    (X_df, y)  – y is a pd.Series for devset, None for testset.
    X_df rows and y are aligned on video ID.
    """
    frames = [load_feature_set(name, split) for name in feature_names]
    X_df   = pd.concat(frames, axis=1, join="inner").dropna()

    if split == "devset":
        y      = load_labels()
        common = X_df.index.intersection(y.index)
        return X_df.loc[common], y.loc[common]

    return X_df, None
