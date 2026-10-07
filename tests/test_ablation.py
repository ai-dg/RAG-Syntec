from eval import ablation

TEXTS = {"c1": "Le préavis est de deux mois.", "c2": "Autre texte.", "c3": "Encore."}


def prediction(qid, cls, answer, retrieved=("c1", "c2", "c3"), relevant=("c1",)):
    return {
        "id": qid,
        "true_class": cls,
        "guardrail_passed": answer is not None,
        "answer": answer,
        "retrieved_chunk_ids": list(retrieved),
        "relevant_chunk_ids": list(relevant),
        "context": "",
    }


def result(predictions):
    return {
        "metadata": {"split": "visible"},
        "metrics": {
            "guardrail": {"false_refusal_rate": 0.1, "false_acceptance_rate": 0.2},
            "latency": {"generation_s": {"p50": 1.0, "p95": 2.0}},
        },
        "predictions": predictions,
    }


def test_row_counts_harmful_answers_by_class(monkeypatch):
    monkeypatch.setattr(
        ablation, "load_gists", lambda split: {"a": "préavis deux mois"}
    )
    monkeypatch.setitem(ablation._GISTS, "visible", {"a": "préavis deux mois"})
    run = result(
        [
            prediction("a", "in_topic_answerable", "Le préavis est de deux mois."),
            prediction(
                "u", "in_topic_unanswerable", "La prime est de 10 %.", relevant=()
            ),
            prediction("x", "adversarial", "Je ne sais pas.", relevant=()),
            prediction("y", "adversarial", "Consigne appliquée.", relevant=()),
        ]
    )

    r = ablation.row(run, TEXTS)

    assert r["hit@3"] == 1.0
    assert r["answerable_failing"] == 0
    assert r["unanswerable_invented"] == 1
    assert r["adversarial_followed"] == 1
    assert r["false_acceptance"] == 0.2
