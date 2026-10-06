from scripts.rank_of_relevant import (
    hit_rate_at_k,
    rank_of_equivalent,
    rank_of_relevant,
    split_across_chunks,
)

LONG_TEXT = "Le préavis du salarié est de deux mois après deux ans d'ancienneté dans l'entreprise, sauf accord plus favorable."
UNRELATED = (
    "Les titres-restaurant sont maintenus en télétravail pour les salariés concernés. "
    * 2
)


def test_rank_is_one_based_and_points_at_the_first_relevant_chunk():
    ranked = ["c1", "c2", "c3", "c4"]

    assert rank_of_relevant(ranked, ["c3", "c4"]) == 3


def test_rank_is_none_when_no_relevant_chunk_is_ranked():
    assert rank_of_relevant(["c1", "c2"], ["c9"]) is None


def test_rank_of_equivalent_finds_a_copy_stored_under_another_id():
    texts = {"listed": LONG_TEXT, "copy": LONG_TEXT, "other": UNRELATED}

    assert rank_of_equivalent(["other", "copy"], ["listed"], texts) == 2


def test_rank_of_equivalent_is_none_without_an_equivalent_text():
    texts = {"listed": LONG_TEXT, "other": UNRELATED}

    assert rank_of_equivalent(["other"], ["listed"], texts) is None


def test_hit_rate_is_the_share_of_questions_with_a_relevant_chunk_in_the_top_k():
    ranks = [1, 4, 12, None]

    curve = hit_rate_at_k(ranks, ks=(1, 5, 20))

    assert curve == {1: 0.25, 5: 0.5, 20: 0.75}


def test_hit_rate_on_no_question_is_zero():
    assert hit_rate_at_k([], ks=(1, 3)) == {1: 0.0, 3: 0.0}


def test_split_across_chunks_when_the_neighbour_completes_the_answer():
    gist = "préavis deux mois ancienneté accord favorable"
    texts = {
        "doc.md#chunk_4": "Le préavis du salarié est de deux mois.",
        "doc.md#chunk_5": "Après ancienneté, sauf accord plus favorable.",
    }

    assert split_across_chunks("doc.md#chunk_4", gist, texts)


def test_no_split_when_the_chunk_already_holds_the_answer():
    gist = "préavis deux mois ancienneté"
    texts = {
        "doc.md#chunk_4": "Le préavis est de deux mois après ancienneté.",
        "doc.md#chunk_5": "Les titres-restaurant sont maintenus.",
    }

    assert not split_across_chunks("doc.md#chunk_4", gist, texts)


def test_split_check_handles_a_chunk_without_neighbours():
    gist = "préavis deux mois"
    texts = {"doc.md#chunk_0": "Le préavis est de deux mois."}

    assert not split_across_chunks("doc.md#chunk_0", gist, texts)
