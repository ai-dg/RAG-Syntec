from eval.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mrr,
    false_refusal_rate,
    false_acceptance_rate,
    confusion_matrix,
    sentence_overlap_proxy,
    faithfulness_proxy,
)
from eval.judge import judge_agreement_rate

CONTEXT = "le salarié a droit à un préavis de deux mois"


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


def test_sentence_overlap_proxy_partial_overlap():
    result = sentence_overlap_proxy("le salarié a droit", "le salarié a un préavis")
    assert result == 0.75


def test_sentence_overlap_proxy_full_and_no_overlap():
    assert sentence_overlap_proxy("a b c", "a b c") == 1.0
    assert sentence_overlap_proxy("x y z", "a b c") == 0.0


def test_sentence_overlap_proxy_empty_sentence_returns_zero():
    assert sentence_overlap_proxy("", CONTEXT) == 0


def test_faithfulness_proxy_counts_grounded_sentences_not_the_inverse():
    answer = (
        "Le salarié a droit à un préavis. "
        "Il peut aussi demander une prime exceptionnelle de dix mille euros."
    )
    # 1 grounded sentence out of 2: a flipped division would give 2.0
    assert faithfulness_proxy(answer, CONTEXT) == 0.5


def test_faithfulness_proxy_bounds():
    assert faithfulness_proxy("Le salarié a droit à un préavis.", CONTEXT) == 1.0
    assert faithfulness_proxy("Prime de dix mille euros.", CONTEXT) == 0.0


def test_faithfulness_proxy_empty_answer_returns_zero():
    assert faithfulness_proxy("", CONTEXT) == 0.0


def test_faithfulness_proxy_threshold_is_inclusive():
    # 3 of 5 words overlap -> score exactly 0.6
    assert faithfulness_proxy("a b c x y", "a b c", threshold=0.6) == 1.0
    assert faithfulness_proxy("a b c x y", "a b c", threshold=0.61) == 0.0


def test_judge_agreement_rate():
    assert judge_agreement_rate([True, False, True, True], [True, False, False, True]) == 0.75


def test_judge_agreement_rate_unparseable_verdict_counts_as_disagreement():
    assert judge_agreement_rate([True, None], [True, False]) == 0.5


def test_judge_agreement_rate_empty_returns_zero():
    assert judge_agreement_rate([], []) == 0


def test_judge_agreement_rate_sixty_percent_case():
    judge = [True] * 12 + [False] * 8
    human = [True] * 20
    assert judge_agreement_rate(judge, human) == 0.6
