from scripts.measure_duplicates import (
    distinct_texts,
    duplicate_groups,
    duplicate_stats,
    normalize,
    text_equivalent_recall,
    texts_equivalent,
)

LONG_TEXT = "Le préavis du salarié est de deux mois après deux ans d'ancienneté dans l'entreprise, sauf accord plus favorable."


def test_normalize_ignores_case_and_whitespace():
    assert normalize("  Le   Préavis\n est  ") == "le préavis est"


def test_duplicate_groups_finds_exact_twins_after_normalization():
    texts = {
        "a.md#chunk_0": "Même texte",
        "b.md#chunk_3": "même   texte",
        "a.md#chunk_1": "Texte unique",
    }

    groups = duplicate_groups(texts)

    assert [sorted(g) for g in groups] == [["a.md#chunk_0", "b.md#chunk_3"]]


def test_duplicate_stats_counts_copies_and_cross_document_groups():
    texts = {
        "a.md#chunk_0": "x",
        "b.md#chunk_0": "x",
        "b.md#chunk_1": "x",
        "a.md#chunk_1": "y",
        "a.md#chunk_2": "y",
        "a.md#chunk_3": "unique",
    }

    stats = duplicate_stats(texts)

    assert stats["n_chunks"] == 6
    assert stats["n_groups"] == 2
    assert stats["chunks_in_groups"] == 5
    assert stats["redundant_copies"] == 3
    assert stats["groups_spanning_documents"] == 1
    assert stats["largest_group"] == 3
    assert stats["share_in_groups"] == 5 / 6


def test_duplicate_stats_on_an_index_without_duplicates():
    stats = duplicate_stats({"a.md#chunk_0": "x", "a.md#chunk_1": "y"})

    assert stats["n_groups"] == 0
    assert stats["share_in_groups"] == 0.0
    assert stats["largest_group"] == 0


def test_distinct_texts_counts_repeated_texts_once():
    texts = {"c1": "même texte", "c2": "Même  texte", "c3": "autre texte"}

    assert distinct_texts(["c1", "c2", "c3"], texts) == 2


def test_texts_equivalent_when_equal_after_normalization():
    assert texts_equivalent(LONG_TEXT, "  " + LONG_TEXT.upper() + "\n")


def test_texts_equivalent_when_a_long_chunk_is_contained_in_another():
    with_heading = "Engagement et contrat de travail\n" + LONG_TEXT

    assert texts_equivalent(LONG_TEXT, with_heading)


def test_short_containment_is_not_equivalence():
    assert not texts_equivalent("## Article", "## Article 2 : le préavis est de deux mois")


def test_unrelated_texts_are_not_equivalent():
    assert not texts_equivalent(LONG_TEXT, "Les titres-restaurant sont maintenus en télétravail. " * 3)


def test_text_equivalent_recall_counts_a_copy_under_another_id():
    texts = {"listed": LONG_TEXT, "copy": LONG_TEXT, "other": "Sans rapport. " * 10}

    assert text_equivalent_recall(["other", "copy"], ["listed"], texts, k=3) == 1.0


def test_text_equivalent_recall_is_a_fraction_of_the_relevant_chunks():
    second = "Autre article sur les congés payés, avec une règle précise et des conditions détaillées. " * 2
    texts = {"r1": LONG_TEXT, "r2": second, "other": "Sans rapport. " * 10}

    assert text_equivalent_recall(["r1", "other"], ["r1", "r2"], texts, k=3) == 0.5


def test_text_equivalent_recall_ignores_chunks_ranked_beyond_k():
    texts = {"listed": LONG_TEXT, "copy": LONG_TEXT, "other": "Sans rapport. " * 10}

    assert text_equivalent_recall(["other", "copy"], ["listed"], texts, k=1) == 0.0
