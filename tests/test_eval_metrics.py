from eval.metrics import recall_at_k, precision_at_k, reciprocal_rank, mrr


def test_recall_at_k():
    result = recall_at_k(["c1", "c5", "c9"], ["c5", "c9", "c20"], k=3)
    assert result == 2 / 3


def test_precision_at_k():
    result = precision_at_k(["c1", "c5", "c9", "c30", "c40"], ["c5", "c9", "c20", "c99"], k=5)
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
    assert result == (1/3 + 0) / 2
