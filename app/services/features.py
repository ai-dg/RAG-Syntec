"""Features for the abstention classifier, computed from retrieval signals only.

Every feature is available before generation and depends only on the question and
the distances returned by the vector search, never on the golden label.
Distances are squared L2 between normalised vectors: lower means closer.
"""

import re

import numpy as np

FEATURE_K = 10
FEATURE_NAMES = (
    "best_distance",
    "mean_top5",
    "std_top5",
    "gap_1_2",
    "gap_1_5",
    "gap_1_10",
    "n_within_0_05_of_best",
    "question_words",
)


def extract_features(distances: list[float], question: str) -> dict[str, float]:
    """`distances` are the FEATURE_K best distances, nearest first."""
    if not distances:
        raise ValueError("at least one distance is needed")
    d = np.asarray(sorted(distances)[:FEATURE_K], dtype=float)
    padded = np.pad(d, (0, FEATURE_K - len(d)), constant_values=d[-1])
    top5 = padded[:5]
    return {
        "best_distance": float(padded[0]),
        "mean_top5": float(top5.mean()),
        "std_top5": float(top5.std()),
        "gap_1_2": float(padded[1] - padded[0]),
        "gap_1_5": float(padded[4] - padded[0]),
        "gap_1_10": float(padded[-1] - padded[0]),
        "n_within_0_05_of_best": float(np.sum(padded <= padded[0] + 0.05)),
        "question_words": float(len(re.findall(r"\w+", question))),
    }


def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
