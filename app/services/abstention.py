"""Answer-or-refuse decision from a trained logistic regression.

The model is stored as plain JSON (feature names, scaling, weights, intercept,
operating threshold), so inference needs only numpy and loading a model file
cannot execute code. If the file is missing or unreadable, the caller falls back
to the distance threshold.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.services.features import FEATURE_NAMES, feature_vector

logger = logging.getLogger(__name__)


class AbstentionModel:
    def __init__(self, spec: dict):
        if tuple(spec["feature_names"]) != FEATURE_NAMES:
            raise ValueError("model features do not match the feature extractor")
        self.mean = np.asarray(spec["mean"], dtype=float)
        self.scale = np.asarray(spec["scale"], dtype=float)
        self.weights = np.asarray(spec["weights"], dtype=float)
        self.intercept = float(spec["intercept"])
        self.threshold = float(spec["threshold"])
        self.version = spec.get("version", "unknown")

    def probability(self, features: dict[str, float]) -> float:
        """Probability that the question should be answered."""
        z = (feature_vector(features) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.weights + self.intercept))))

    def should_answer(self, features: dict[str, float]) -> tuple[bool, float]:
        p = self.probability(features)
        return p >= self.threshold, p


def to_spec(mean, scale, weights, intercept, threshold, version) -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": [float(x) for x in mean],
        "scale": [float(x) for x in scale],
        "weights": [float(x) for x in weights],
        "intercept": float(intercept),
        "threshold": float(threshold),
        "version": version,
    }


@lru_cache
def load_model(path: str) -> AbstentionModel | None:
    try:
        return AbstentionModel(json.loads(Path(path).read_text()))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("abstention model unavailable (%s): falling back to the threshold", exc)
        return None
