"""Train the abstention classifier and compare it with the distance threshold.

Model selection uses repeated stratified cross-validation on the visible split
only. The chosen model is then fitted on the whole visible split and evaluated
once on the held-out split. The operating point is fixed from out-of-fold
predictions before the held-out split is read: answer when the probability is at
least the cut that refuses at most MAX_FALSE_REFUSAL of the answerable questions.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.abstention import to_spec
from app.services.features import FEATURE_NAMES

DATASET = ROOT / "eval" / "abstention_dataset.csv"
MODEL_PATH = ROOT / "models" / "abstention.json"
SEED = 0
MAX_FALSE_REFUSAL = 0.10


def load(split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    with open(DATASET) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    return X, y, rows


def logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
    )


def boosting():
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1, random_state=SEED
    )


def decision_metrics(y: np.ndarray, answer: np.ndarray) -> dict:
    """Answering is the positive class; false refusal = answerable refused,
    false acceptance = should-refuse answered."""
    return {
        "precision": precision_score(y, answer, zero_division=0),
        "recall": recall_score(y, answer, zero_division=0),
        "f1": f1_score(y, answer, zero_division=0),
        "false_refusal": float(np.mean(answer[y == 1] == 0)) if (y == 1).any() else 0.0,
        "false_acceptance": (
            float(np.mean(answer[y == 0] == 1)) if (y == 0).any() else 0.0
        ),
    }


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
    }


def out_of_fold(make_model, X, y, repeats=10):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    oof = np.zeros((repeats, len(y)))
    for i, (train, test) in enumerate(splitter.split(X, y)):
        model = make_model().fit(X[train], y[train])
        oof[i // 5, test] = model.predict_proba(X[test])[:, 1]
    return oof


def operating_point(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Highest cut that refuses at most MAX_FALSE_REFUSAL of the answerable questions."""
    positives = np.sort(probabilities[y == 1])
    allowed = int(np.floor(MAX_FALSE_REFUSAL * len(positives)))
    return float(positives[allowed]) if len(positives) else 0.5


def fmt(metrics: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in metrics.items())


def main() -> None:
    threshold = get_settings().relevance_threshold
    X, y, _ = load("visible")
    best = X[:, FEATURE_NAMES.index("best_distance")]
    print(
        f"visible split: {len(y)} rows, {int(y.sum())} to answer, {int(len(y) - y.sum())} to refuse"
    )

    print(f"\nbaseline, distance threshold {threshold}:")
    print(
        "  ",
        fmt(ranking_metrics(y, -best)),
        " ",
        fmt(decision_metrics(y, (best <= threshold).astype(int))),
    )

    oofs = {}
    for name, make_model in (
        ("logistic regression", logistic),
        ("gradient boosting", boosting),
    ):
        oof = out_of_fold(make_model, X, y)
        oofs[name] = oof
        auc = np.mean([roc_auc_score(y, o) for o in oof])
        pr = np.mean([average_precision_score(y, o) for o in oof])
        print(
            f"\n{name}, 5-fold CV x10: roc_auc {auc:.3f} (sd {np.std([roc_auc_score(y, o) for o in oof]):.3f})  pr_auc {pr:.3f}"
        )

    oof_mean = oofs["logistic regression"].mean(axis=0)
    cut = operating_point(y, oof_mean)
    print(
        f"\nlogistic regression operating point (out-of-fold, at most {MAX_FALSE_REFUSAL:.0%} answerable refused): {cut:.3f}"
    )
    print(
        "   out-of-fold decisions:",
        fmt(decision_metrics(y, (oof_mean >= cut).astype(int))),
    )

    model = logistic().fit(X, y)
    scaler, regression = (
        model.named_steps["standardscaler"],
        model.named_steps["logisticregression"],
    )
    print(
        "\nstandardised coefficients (visible split, positive = more likely to answer):"
    )
    for name, w in sorted(
        zip(FEATURE_NAMES, regression.coef_[0]), key=lambda t: -abs(t[1])
    ):
        print(f"   {name:24s} {w:+.3f}")

    Xh, yh, rows_h = load("held_out")
    best_h = Xh[:, FEATURE_NAMES.index("best_distance")]
    proba_h = model.predict_proba(Xh)[:, 1]
    print(f"\nheld-out split, read once: {len(yh)} rows, {int(yh.sum())} to answer")
    print(
        "   threshold :",
        fmt(ranking_metrics(yh, -best_h)),
        " ",
        fmt(decision_metrics(yh, (best_h <= threshold).astype(int))),
    )
    print(
        "   classifier:",
        fmt(ranking_metrics(yh, proba_h)),
        " ",
        fmt(decision_metrics(yh, (proba_h >= cut).astype(int))),
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    spec = to_spec(
        scaler.mean_,
        scaler.scale_,
        regression.coef_[0],
        regression.intercept_[0],
        cut,
        version="logreg-v1-visible55",
    )
    MODEL_PATH.write_text(json.dumps(spec, indent=2))
    print(f"\nwrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
