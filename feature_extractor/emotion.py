"""
emotion.py - Extract emotion scores and language complexity features (Olufela's approach)

Emotion Features (10):
  - joy, anger, fear, sadness, positive, negative, anticipation, disgust, surprise, trust

Derived Emotion Features (5):
  - emotion_intensity
  - dominant_emotion one-hot: joy, anger, fear, sadness (all zeros = neutral)

Language Complexity Features (4):
  - word_count, vocab_size, unique_word_ratio, avg_word_length

Total: 19 numeric features
"""

import re
import numpy as np
from nrclex import NRCLex


EMOTION_LABELS = [
    "joy", "anger", "fear", "sadness",
    "positive", "negative",
    "anticipation", "disgust", "surprise", "trust"
]

CORE_LABELS = ["joy", "anger", "fear", "sadness"]


def clean_text(text: str) -> str:
    """Clean text for analysis."""
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def compute_language_complexity(words: list[str]) -> dict:
    """Compute lexical complexity metrics from words."""
    if not words:
        return {
            "word_count": 0,
            "vocab_size": 0,
            "unique_word_ratio": float("nan"),
            "avg_word_length": float("nan"),
        }

    unique = set(words)
    return {
        "word_count": len(words),
        "vocab_size": len(unique),
        "unique_word_ratio": len(unique) / len(words),
        "avg_word_length": sum(len(w) for w in words) / len(words),
    }


def compute_emotion_scores(text: str) -> dict:
    """Compute emotion scores using NRCLex."""
    scores = {f"{label}_score": 0.0 for label in EMOTION_LABELS}

    if text.strip():
        emotion = NRCLex(text)
        emotion.load_raw_text(text)
        af = emotion.affect_frequencies
        for label in EMOTION_LABELS:
            scores[f"{label}_score"] = af.get(label, 0.0)

    core_scores = [scores[f"{label}_score"] for label in CORE_LABELS]
    scores["dominant_emotion"] = CORE_LABELS[int(np.argmax(core_scores))] if sum(core_scores) > 0 else "none"
    scores["emotion_intensity"] = sum(core_scores)

    return scores


def extract_emotion_features(text: str) -> dict:
    """Extract all emotion and language complexity features.

    Returns:
        Dictionary with 15 numeric features + dominant_emotion (categorical string):
          10 emotion scores, 4 complexity metrics, 1 emotion_intensity
    """
    clean = clean_text(text)
    words = clean.split()

    features = {}
    features.update(compute_language_complexity(words))
    features.update(compute_emotion_scores(clean))

    return features


def extract_emotion_features_array(text: str, precomputed: dict = None) -> np.ndarray:
    """Extract emotion features as numeric array.

    Returns:
        np.array of 19 features:
          10 emotion scores + 4 complexity + 1 intensity + 4 dominant_emotion one-hot
    """
    features = precomputed if precomputed is not None else extract_emotion_features(text)

    emotion_scores = [
        features[f"{label}_score"]
        for label in EMOTION_LABELS
    ]

    complexity_scores = [
        features["word_count"],
        features["vocab_size"],
        features["unique_word_ratio"],
        features["avg_word_length"]
    ]

    emotion_derived = [
        features["emotion_intensity"],
    ]

    dominant_onehot = [
        1 if features["dominant_emotion"] == label else 0
        for label in CORE_LABELS
    ]

    return np.array(emotion_scores + complexity_scores + emotion_derived + dominant_onehot)


if __name__ == "__main__":
    import sys

    results = [0, 0]

    def check(name, condition, detail=""):
        if condition:
            print(f"[pass] {name}")
            results[0] += 1
        else:
            print(f"[FAIL] {name}" + (f"  {detail}" if detail else ""))
            results[1] += 1

    # --- clean_text ---
    check("clean_text: strips brackets",    clean_text("[music] hello world") == "hello world")
    check("clean_text: strips punctuation", clean_text("hello, world!") == "hello world")
    check("clean_text: lowercases",         clean_text("HELLO World") == "hello world")

    # --- compute_language_complexity ---
    lc = compute_language_complexity([])
    check("complexity empty: word_count=0",          lc["word_count"] == 0)
    check("complexity empty: unique_word_ratio=nan", np.isnan(lc["unique_word_ratio"]))

    lc = compute_language_complexity(["hello", "world", "hello"])
    check("complexity: word_count=3",      lc["word_count"] == 3)
    check("complexity: vocab_size=2",      lc["vocab_size"] == 2)
    check("complexity: unique_word_ratio", abs(lc["unique_word_ratio"] - 2/3) < 1e-9)
    check("complexity: avg_word_length=5", abs(lc["avg_word_length"] - 5.0) < 1e-9)

    # --- neutral text: dominant_emotion should be "none" ---
    neutral = compute_emotion_scores("the building stood quietly beside the road")
    check("neutral: dominant_emotion='none'", neutral["dominant_emotion"] == "none",
          neutral["dominant_emotion"])
    check("neutral: emotion_intensity=0",     neutral["emotion_intensity"] == 0.0)

    # --- emotional text ---
    joyful = compute_emotion_scores("I am so happy and joyful today love everything")
    check("joyful: joy_score > 0",        joyful["joy_score"] > 0,        str(joyful["joy_score"]))
    check("joyful: emotion_intensity > 0", joyful["emotion_intensity"] > 0)

    # --- extract_emotion_features_array: feature breakdown ---
    sample   = "I am so happy and joyful today because the weather is wonderful and beautiful"
    feat_dict = extract_emotion_features(sample)
    feat_arr  = extract_emotion_features_array(sample, precomputed=feat_dict)

    emotion_scores    = [feat_dict[f"{l}_score"] for l in EMOTION_LABELS]
    complexity_scores = [feat_dict["word_count"], feat_dict["vocab_size"],
                         feat_dict["unique_word_ratio"], feat_dict["avg_word_length"]]
    emotion_derived   = [feat_dict["emotion_intensity"]]

    print(f"\n--- emotion_scores ({len(emotion_scores)}) ---")
    for label, val in zip(EMOTION_LABELS, emotion_scores):
        print(f"  {label}: {val:.4f}")

    print(f"\n--- complexity_scores ({len(complexity_scores)}) ---")
    for key, val in zip(["word_count", "vocab_size", "unique_word_ratio", "avg_word_length"], complexity_scores):
        print(f"  {key}: {val}")

    print(f"\n--- emotion_derived ({len(emotion_derived)}) ---")
    print(f"  emotion_intensity: {emotion_derived[0]:.4f}")

    dominant_onehot = [1 if feat_dict["dominant_emotion"] == l else 0 for l in CORE_LABELS]
    print(f"\n--- dominant_emotion one-hot ({len(dominant_onehot)}) ---")
    for label, val in zip(CORE_LABELS, dominant_onehot):
        print(f"  {label}: {val}  (dominant: {feat_dict['dominant_emotion']})")

    print(f"\ntotal: {len(emotion_scores)} + {len(complexity_scores)} + {len(emotion_derived)} + {len(dominant_onehot)} = {len(feat_arr)}")

    check(f"array length = {len(feat_arr)} (expect 19)", feat_arr.shape == (19,), str(feat_arr.shape))
    check("all elements finite or nan",
          all(np.isfinite(x) or np.isnan(x) for x in feat_arr))

    # joyful text: one-hot should have joy=1, rest=0
    joy_dict = extract_emotion_features("I am so happy and joyful today love everything")
    joy_arr  = extract_emotion_features_array("", precomputed=joy_dict)
    check("joyful: dominant one-hot joy=1",    joy_arr[-4] == 1, str(joy_arr[-4:]))
    check("joyful: dominant one-hot others=0", np.all(joy_arr[-3:] == 0), str(joy_arr[-3:]))

    # neutral text: one-hot should be all zeros
    neutral_dict = extract_emotion_features("the building stood quietly beside the road")
    neutral_arr  = extract_emotion_features_array("", precomputed=neutral_dict)
    check("neutral: dominant one-hot all zeros", np.all(neutral_arr[-4:] == 0), str(neutral_arr[-4:]))

    # empty text: shape should match
    empty_arr = extract_emotion_features_array("")
    check("empty: shape (19,)", empty_arr.shape == (19,), str(empty_arr.shape))

    print(f"\n{results[0]} passed, {results[1]} failed")
    sys.exit(0 if results[1] == 0 else 1)
