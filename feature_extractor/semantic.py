"""
semantic.py - Extract 7 semantic/text features (Conner's approach)

Features:
  - noun_ratio, verb_ratio, adj_ratio
  - lexical_diversity
  - avg_word_length
  - avg_sentence_length
  - rarity (word frequency)
"""

import spacy
import numpy as np
from pathlib import Path
from collections import Counter


nlp = spacy.load("en_core_web_sm")


def build_word_frequency(text_dirs):
    """Build word frequency counter from all text files in one or more directories."""
    if isinstance(text_dirs, (str, Path)):
        text_dirs = [text_dirs]

    word_counter = Counter()

    for text_dir in text_dirs:
        for path in Path(text_dir).glob("*.txt"):
            doc = nlp.tokenizer(path.read_text(encoding="utf-8").strip())
            word_counter.update(t.text.lower() for t in doc if t.is_alpha)

    return dict(word_counter)


def extract_text_features(text, word_freq):
    """Extract 7 semantic features from text.
    
    Args:
        text: Input text string
        word_freq: Dictionary of word frequencies
    
    Returns:
        np.array of 7 features
    """
    doc = nlp(text)

    words = []
    noun_count = verb_count = adj_count = 0
    for t in doc:
        if t.is_alpha:
            words.append(t.text.lower())
        pos = t.pos_
        if pos == "NOUN":
            noun_count += 1
        elif pos == "VERB":
            verb_count += 1
        elif pos == "ADJ":
            adj_count += 1

    num_words = len(words)

    if num_words == 0:
        return np.zeros(7)

    noun_ratio = noun_count / num_words
    verb_ratio = verb_count / num_words
    adj_ratio = adj_count / num_words

    unique_words = len(set(words))
    lexical_diversity = unique_words / num_words

    avg_word_length = sum(len(w) for w in words) / num_words

    sentences = list(doc.sents)
    avg_sentence_length = num_words / len(sentences) if sentences else 0

    corpus_size = sum(word_freq.values())
    freq_values = np.array([word_freq.get(w, 1) / corpus_size for w in words])
    rarity = float(-np.mean(np.log(freq_values)))

    return np.array([
        noun_ratio,
        verb_ratio,
        adj_ratio,
        lexical_diversity,
        avg_word_length,
        avg_sentence_length,
        rarity
    ])


if __name__ == "__main__":
    import sys
    import tempfile
    import os

    results = [0, 0]

    def check(name, condition, detail=""):
        if condition:
            print(f"[pass] {name}")
            results[0] += 1
        else:
            print(f"[FAIL] {name}" + (f"  {detail}" if detail else ""))
            results[1] += 1

    CORPUS = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore on a sunny afternoon.",
        "Scientists discovered a beautiful new species deep in the rainforest.",
    ]
    SAMPLE = "The curious scientist carefully examined the rare ancient artifact."

    # --- build_word_frequency ---
    tmp_dir = tempfile.mkdtemp()
    try:
        for i, text in enumerate(CORPUS):
            Path(tmp_dir, f"doc{i}.txt").write_text(text, encoding="utf-8")

        word_freq = build_word_frequency(tmp_dir)
        check("word_freq: non-empty",        len(word_freq) > 0,           str(len(word_freq)))
        check("word_freq: 'the' present",    "the" in word_freq)
        check("word_freq: 'the' count >= 3", word_freq.get("the", 0) >= 3, str(word_freq.get("the")))

        # --- extract_text_features ---
        features = extract_text_features(SAMPLE, word_freq)
        labels = ["noun_ratio", "verb_ratio", "adj_ratio", "lexical_diversity",
                  "avg_word_length", "avg_sentence_length", "rarity"]

        print("\n--- feature breakdown ---")
        for label, val in zip(labels, features):
            print(f"  {label}: {val:.4f}")
        print(f"total array length: {len(features)}")

        check("features: shape (7,)",              features.shape == (7,),       str(features.shape))
        check("features: noun_ratio >= 0",         features[0] >= 0,             f"{features[0]:.3f}")
        check("features: verb_ratio >= 0",         features[1] >= 0,             f"{features[1]:.3f}")
        check("features: adj_ratio >= 0",          features[2] >= 0,             f"{features[2]:.3f}")
        check("features: lexical_diversity (0,1]", 0 < features[3] <= 1,         f"{features[3]:.3f}")
        check("features: avg_word_length > 0",     features[4] > 0,              f"{features[4]:.3f}")
        check("features: avg_sentence_length > 0", features[5] > 0,              f"{features[5]:.3f}")
        check("features: rarity is finite",        np.isfinite(features[6]),     str(features[6]))

        # empty text
        features_empty = extract_text_features("", word_freq)
        check("empty: shape (7,)",  features_empty.shape == (7,),        str(features_empty.shape))
        check("empty: all zeros",   np.all(features_empty == 0),         str(features_empty))

    finally:
        for fname in os.listdir(tmp_dir):
            os.unlink(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)

    print(f"\n{results[0]} passed, {results[1]} failed")
    sys.exit(0 if results[1] == 0 else 1)
