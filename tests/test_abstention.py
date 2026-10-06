import json

import numpy as np
import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.services import retrieval
from app.services.abstention import AbstentionModel, load_model, to_spec
from app.services.features import (
    FEATURE_K,
    FEATURE_NAMES,
    extract_features,
    feature_vector,
)


def test_features_come_from_distances_and_question_only():
    features = extract_features([0.5, 0.3, 0.9, 0.4, 0.6, 0.7], "Quel est le préavis ?")

    assert set(features) == set(FEATURE_NAMES)
    assert features["best_distance"] == pytest.approx(0.3)
    assert features["gap_1_2"] == pytest.approx(0.1)
    assert features["gap_1_5"] == pytest.approx(0.4)
    assert features["n_within_0_05_of_best"] == 1
    assert features["question_words"] == 4
    assert not any(np.isnan(v) for v in features.values())


def test_short_result_lists_are_padded_with_the_last_distance():
    features = extract_features([0.4], "q")

    assert features["gap_1_10"] == 0.0
    assert features["std_top5"] == 0.0


def test_features_need_at_least_one_distance():
    with pytest.raises(ValueError):
        extract_features([], "q")


def test_only_the_feature_k_nearest_are_used():
    near = [0.1] * FEATURE_K
    assert extract_features(near + [5.0], "q") == extract_features(near, "q")


def spec(weights, intercept=0.0, threshold=0.5):
    n = len(FEATURE_NAMES)
    return to_spec(np.zeros(n), np.ones(n), weights, intercept, threshold, "test")


def test_probability_is_a_logistic_of_the_scaled_features():
    weights = np.zeros(len(FEATURE_NAMES))
    weights[0] = -2.0
    model = AbstentionModel(spec(weights, intercept=1.0))
    features = extract_features([0.5], "q")

    expected = 1 / (1 + np.exp(-(1.0 - 2.0 * 0.5)))
    assert model.probability(features) == pytest.approx(expected)


def test_should_answer_applies_the_operating_threshold():
    model = AbstentionModel(spec(np.zeros(len(FEATURE_NAMES)), threshold=0.6))

    answer, p = model.should_answer(extract_features([0.5], "q"))

    assert p == pytest.approx(0.5)
    assert answer is False


def test_a_model_trained_on_other_features_is_rejected():
    bad = spec(np.zeros(len(FEATURE_NAMES)))
    bad["feature_names"] = ["something_else"] * len(FEATURE_NAMES)

    with pytest.raises(ValueError):
        AbstentionModel(bad)


def test_a_missing_model_file_falls_back_to_none(tmp_path):
    load_model.cache_clear()

    assert load_model(str(tmp_path / "missing.json")) is None


def test_a_saved_model_loads_and_predicts(tmp_path):
    load_model.cache_clear()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(spec(np.zeros(len(FEATURE_NAMES)))))

    model = load_model(str(path))

    assert model.should_answer(extract_features([0.3, 0.4], "q"))[1] == pytest.approx(
        0.5
    )
    assert np.allclose(
        feature_vector(extract_features([0.3], "q")).shape, (len(FEATURE_NAMES),)
    )


class RecordingStore:
    def __init__(self, distances):
        self.results = [
            (
                Document(
                    page_content=f"t{i}",
                    metadata={"chunk_id": f"c{i}", "in_force": True},
                ),
                d,
            )
            for i, d in enumerate(distances)
        ]
        self.requested_k = None

    def similarity_search_with_score(self, question, k, filter=None):
        self.requested_k = k
        return self.results[:k]


@pytest.fixture
def classifier_mode(monkeypatch, tmp_path):
    def use(threshold):
        weights = np.zeros(len(FEATURE_NAMES))
        weights[0] = -10.0
        path = tmp_path / "model.json"
        path.write_text(json.dumps(spec(weights, intercept=5.0, threshold=threshold)))
        monkeypatch.setenv("GUARDRAIL_MODE", "classifier")
        monkeypatch.setenv("ABSTENTION_MODEL_PATH", str(path))
        get_settings.cache_clear()
        load_model.cache_clear()

    yield use
    load_model.cache_clear()


def test_classifier_mode_reads_enough_distances_for_its_features(
    monkeypatch, classifier_mode
):
    classifier_mode(threshold=0.5)
    store = RecordingStore([0.3 + 0.05 * i for i in range(12)])
    monkeypatch.setattr(retrieval, "_vector_store", store)

    result = retrieval.retrieve("q")

    assert store.requested_k == FEATURE_K
    assert result["guardrail_mode"] == "classifier"
    assert len(result["chunks"]) == 3


def test_classifier_mode_can_answer_beyond_the_distance_threshold(
    monkeypatch, classifier_mode
):
    classifier_mode(threshold=0.0)
    monkeypatch.setattr(retrieval, "_vector_store", RecordingStore([0.95, 0.96, 0.97]))

    result = retrieval.retrieve("q")

    assert result["context_found"] is True
    assert result["refusal_reason"] is None
    assert 0.0 <= result["confidence"] <= 1.0


def test_classifier_refusal_states_its_reason_and_confidence(
    monkeypatch, classifier_mode
):
    classifier_mode(threshold=1.0)
    monkeypatch.setattr(retrieval, "_vector_store", RecordingStore([0.2, 0.3]))

    result = retrieval.retrieve("q")

    assert result["context_found"] is False
    assert result["refusal_reason"] == "classifier_refused"
    assert result["confidence"] is not None
    assert result["best_score"] == pytest.approx(0.2)


def test_a_missing_model_falls_back_to_the_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDRAIL_MODE", "classifier")
    monkeypatch.setenv("ABSTENTION_MODEL_PATH", str(tmp_path / "absent.json"))
    get_settings.cache_clear()
    load_model.cache_clear()
    monkeypatch.setattr(retrieval, "_vector_store", RecordingStore([1.5]))

    result = retrieval.retrieve("q")

    assert result["guardrail_mode"] == "threshold"
    assert result["refusal_reason"] == "below_relevance_threshold"


def test_threshold_mode_states_why_it_refused(monkeypatch):
    monkeypatch.setattr(retrieval, "_vector_store", RecordingStore([1.5]))

    result = retrieval.retrieve("q")

    assert result["refusal_reason"] == "below_relevance_threshold"
    assert result["confidence"] is None
