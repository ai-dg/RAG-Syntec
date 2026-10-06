import numpy as np
import pytest

pytest.importorskip("sklearn")

from scripts.train_abstention import decision_metrics, logistic, operating_point, out_of_fold


def test_decision_metrics_treat_answering_as_the_positive_class():
    y = np.array([1, 1, 0, 0])
    answer = np.array([1, 0, 1, 0])

    m = decision_metrics(y, answer)

    assert m["false_refusal"] == 0.5
    assert m["false_acceptance"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_operating_point_refuses_at_most_the_allowed_share_of_answerable_questions():
    y = np.array([1] * 10 + [0] * 5)
    probabilities = np.array([0.1 * i for i in range(10)] + [0.05] * 5)

    cut = operating_point(y, probabilities)

    refused = np.sum(probabilities[y == 1] < cut)
    assert refused <= 1
    assert cut == pytest.approx(0.1)


def test_cross_validation_is_deterministic_with_a_fixed_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = np.array([0, 1] * 15)

    first = out_of_fold(logistic, X, y, repeats=2)
    second = out_of_fold(logistic, X, y, repeats=2)

    assert np.array_equal(first, second)
