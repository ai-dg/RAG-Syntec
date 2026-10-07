import pytest

from eval import compare_runs
from eval.compare_runs import (
    compare,
    faithfulness_at_k,
    faithfulness_full_context,
    format_value,
    retrieval_at_k,
    truncate_prediction,
)

TEXTS = {
    "c1": "Le préavis est de deux mois.",
    "c2": "Les titres-restaurant sont maintenus en télétravail.",
    "c3": "Le contingent annuel est de cent trente heures.",
    "c4": "La majoration de nuit est de vingt-cinq pour cent.",
}
GIST = "préavis deux mois"


def make_prediction(**overrides):
    prediction = {
        "id": "q1",
        "true_class": "in_topic_answerable",
        "guardrail_passed": True,
        "best_distance": 0.4,
        "retrieved_chunk_ids": ["c2", "c3", "c1"],
        "relevant_chunk_ids": ["c1"],
        "context": "\n\n".join(TEXTS[c] for c in ["c2", "c3", "c1"]),
        "answer": "Le préavis est de deux mois.",
        "retrieval_s": 0.1,
        "generation_s": 1.0,
    }
    prediction.update(overrides)
    return prediction


def make_result(predictions, top_k=3, label="run"):
    return {
        "metadata": {"label": label, "split": "visible", "top_k": top_k},
        "metrics": {
            "guardrail": {"false_refusal_rate": 0.0, "false_acceptance_rate": 0.0},
            "latency": {
                "retrieval_s": {"p50": 0.1, "p95": 0.2},
                "generation_s": {"p50": 1.0, "p95": 2.0},
            },
        },
        "predictions": predictions,
    }


def test_retrieval_at_k_cuts_the_ranking_at_k():
    prediction = make_prediction(retrieved_chunk_ids=["c2", "c3", "c4", "c1"])

    assert retrieval_at_k([prediction], k=3)["recall"] == 0.0
    assert retrieval_at_k([prediction], k=4)["recall"] == 1.0


def test_mrr_is_cut_at_k_too():
    prediction = make_prediction(retrieved_chunk_ids=["c2", "c1"])

    assert retrieval_at_k([prediction], k=2)["mrr"] == 0.5
    assert retrieval_at_k([prediction], k=1)["mrr"] == 0.0


def test_retrieval_ignores_questions_that_are_not_answerable():
    off_topic = make_prediction(true_class="off_topic", retrieved_chunk_ids=[])

    result = retrieval_at_k([off_topic], k=3)

    assert result == {"n": 0, "recall": None, "hit": None, "mrr": None}


def test_faithfulness_at_k_only_sees_the_first_k_chunks():
    prediction = make_prediction(
        answer="La majoration de nuit est de vingt-cinq pour cent.",
        retrieved_chunk_ids=["c1", "c2", "c3", "c4"],
    )

    assert faithfulness_at_k([prediction], TEXTS, k=3)["mean"] == 0.0
    assert faithfulness_at_k([prediction], TEXTS, k=4)["mean"] == 1.0


def test_faithfulness_skips_blocked_and_non_answerable_questions():
    blocked = make_prediction(answer=None, guardrail_passed=False)
    off_topic = make_prediction(true_class="off_topic")

    assert faithfulness_at_k([blocked, off_topic], TEXTS, k=3) == {"n": 0, "mean": None}
    assert faithfulness_full_context([blocked, off_topic]) == {"n": 0, "mean": None}


def test_faithfulness_full_context_uses_the_context_of_the_run():
    prediction = make_prediction(
        answer="La majoration de nuit est de vingt-cinq pour cent.",
        context=TEXTS["c4"],
    )

    assert faithfulness_full_context([prediction])["mean"] == 1.0


def test_truncate_prediction_rebuilds_the_context_without_mutating_the_original():
    prediction = make_prediction(retrieved_chunk_ids=["c1", "c2", "c3"])

    cut = truncate_prediction(prediction, TEXTS, k=2)

    assert cut["retrieved_chunk_ids"] == ["c1", "c2"]
    assert cut["context"] == TEXTS["c1"] + "\n\n" + TEXTS["c2"]
    assert cut["answer"] == prediction["answer"]
    assert prediction["retrieved_chunk_ids"] == ["c1", "c2", "c3"]


@pytest.fixture
def synthetic_gists(monkeypatch):
    monkeypatch.setattr(compare_runs, "load_gists", lambda split: {"q1": GIST})


def rows_by_label(rows):
    return {label.strip(): (a, b) for label, a, b in rows}


def test_comparing_a_run_with_itself_changes_nothing(synthetic_gists):
    result = make_result([make_prediction()])

    rows = compare(result, result, TEXTS, k=3)

    for label, a, b in rows:
        assert a == b, label


def test_a_wider_run_is_compared_at_the_same_k(synthetic_gists):
    narrow = make_result(
        [make_prediction(retrieved_chunk_ids=["c2", "c3", "c4"])], top_k=3
    )
    wide = make_result(
        [make_prediction(retrieved_chunk_ids=["c2", "c3", "c4", "c1"])], top_k=4
    )

    at_three = rows_by_label(compare(narrow, wide, TEXTS, k=3))
    at_four = rows_by_label(compare(narrow, wide, TEXTS, k=4))

    assert at_three["recall@3 (answerable, n=1)"] == (0.0, 0.0)
    assert at_four["recall@4 (answerable, n=1)"] == (0.0, 1.0)


def test_failures_are_classified_on_the_first_k_chunks_of_each_run(synthetic_gists):
    abstention = "Je ne sais pas."
    short = make_result(
        [make_prediction(retrieved_chunk_ids=["c2", "c3", "c4"], answer=abstention)]
    )
    long = make_result(
        [
            make_prediction(
                retrieved_chunk_ids=["c2", "c3", "c4", "c1"], answer=abstention
            )
        ],
        top_k=4,
    )

    counts = rows_by_label(compare(short, long, TEXTS, k=3))

    assert counts["retrieval"] == (1, 1)


def test_format_value():
    assert format_value(None) == "n/a"
    assert format_value(0.5) == "0.500"
    assert format_value(3) == "3"


def test_relabel_scores_a_saved_run_against_the_current_labels():
    saved = make_result([make_prediction(id="q1", relevant_chunk_ids=["old"])])

    relabelled = compare_runs.relabel(saved, {"q1": ["c1"]})

    assert relabelled["predictions"][0]["relevant_chunk_ids"] == ["c1"]
    assert saved["predictions"][0]["relevant_chunk_ids"] == ["old"]


def test_relabel_keeps_the_saved_label_for_an_unknown_question():
    saved = make_result([make_prediction(id="gone", relevant_chunk_ids=["c9"])])

    relabelled = compare_runs.relabel(saved, {"q1": ["c1"]})

    assert relabelled["predictions"][0]["relevant_chunk_ids"] == ["c9"]


def test_hit_rate_counts_a_question_once_even_with_several_relevant_chunks():
    prediction = make_prediction(
        retrieved_chunk_ids=["c1", "c2", "c3"], relevant_chunk_ids=["c1", "c4"]
    )

    result = retrieval_at_k([prediction], k=3)

    assert result["recall"] == 0.5
    assert result["hit"] == 1.0


def test_each_run_is_read_with_its_own_chunk_texts(synthetic_gists):
    texts_a = {"c1": "Le préavis est de deux mois.", "c2": "x", "c3": "y"}
    texts_b = {"c1": "Texte sans rapport.", "c2": "x", "c3": "y"}
    run = make_result([make_prediction(retrieved_chunk_ids=["c1", "c2", "c3"])])

    rows = rows_by_label(compare(run, run, texts_a, k=3, texts_b=texts_b))

    a, b = rows["faithfulness proxy vs first 3 chunks (biased down)"]
    assert a == 1.0
    assert b == 0.0


def test_absolute_chunk_ids_are_compared_from_the_corpus_folder():
    absolute = "/home/user/project/data/converted/a.md#chunk_3"

    assert compare_runs.normalize_chunk_id(absolute) == "data/converted/a.md#chunk_3"
    assert (
        compare_runs.normalize_chunk_id("data/converted/a.md#chunk_3")
        == "data/converted/a.md#chunk_3"
    )
    assert compare_runs.normalize_chunk_id("other.md#chunk_1") == "other.md#chunk_1"
