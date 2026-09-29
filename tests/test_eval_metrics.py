from eval.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mrr,
    false_refusal_rate,
    false_acceptance_rate,
    confusion_matrix,
)


def test_recall_at_k():
    result = recall_at_k(["c1", "c5", "c9"], ["c5", "c9", "c20"], k=3)
    assert result == 2 / 3


def test_precision_at_k():
    result = precision_at_k(
        ["c1", "c5", "c9", "c30", "c40"], ["c5", "c9", "c20", "c99"], k=5
    )
    assert result == 0.4


def test_precision_at_k_fewer_results_than_k():
    result = precision_at_k(["c1", "c5", "c9"], ["c5", "c9", "c20"], k=5)
    assert result == 2 / 3


def test_reciprocal_rank_finds_match():
    result = reciprocal_rank(["c1", "c2", "c3", "c4"], ["c4"])
    assert result == 0.25


def test_reciprocal_rank_no_match_returns_zero():
    result = reciprocal_rank(["c1", "c2", "c3", "c4"], ["c5"])
    assert result == 0


def test_mrr():
    result = mrr([["c1", "c2", "c3"], ["c5", "c6"]], [["c3"], ["c99"]])
    assert result == (1 / 3 + 0) / 2


def test_false_refusal_rate():
    predictions = [
        {"true_class": "in_topic_answerable", "guardrail_passed": True},
        {"true_class": "in_topic_unanswerable", "guardrail_passed": False},
        {"true_class": "off_topic", "guardrail_passed": False},
        {"true_class": "adversarial", "guardrail_passed": True},
    ]
    assert false_refusal_rate(predictions) == 0.5


def test_false_acceptance_rate():
    predictions = [
        {"true_class": "in_topic_answerable", "guardrail_passed": True},
        {"true_class": "in_topic_unanswerable", "guardrail_passed": False},
        {"true_class": "off_topic", "guardrail_passed": False},
        {"true_class": "adversarial", "guardrail_passed": True},
    ]
    assert false_acceptance_rate(predictions) == 0.5


def test_refuse_everything_is_a_trap():
    predictions = [
        {"true_class": "in_topic_answerable", "guardrail_passed": False},
        {"true_class": "in_topic_unanswerable", "guardrail_passed": False},
        {"true_class": "off_topic", "guardrail_passed": False},
        {"true_class": "adversarial", "guardrail_passed": False},
    ]

    assert false_acceptance_rate(predictions) == 0.0
    assert false_refusal_rate(predictions) == 1.0


def test_confusion_matrix():

    predictions = [
        {"true_class": "in_topic_answerable", "guardrail_passed": True},
        {"true_class": "in_topic_unanswerable", "guardrail_passed": False},
        {"true_class": "off_topic", "guardrail_passed": False},
        {"true_class": "adversarial", "guardrail_passed": True},
    ]

    matrix = confusion_matrix(predictions)
    assert matrix["adversarial"]["passed"] == 1
