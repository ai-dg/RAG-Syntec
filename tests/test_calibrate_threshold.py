import pytest

from scripts.calibrate_threshold import errors_at, recommend_threshold


def test_separable_groups_get_the_midpoint_of_the_gap():
    result = recommend_threshold([0.3, 0.5, 0.6], [0.8, 1.0])

    assert result["separable"] is True
    assert result["threshold"] == pytest.approx(0.7)
    assert result["errors"] == (0, 0)


def test_overlapping_groups_get_the_cut_with_fewest_errors():
    in_scores = [0.3, 0.5, 0.75, 0.9]
    off_scores = [0.7, 0.85, 1.0, 1.1]

    result = recommend_threshold(in_scores, off_scores)

    assert result["separable"] is False
    assert sum(result["errors"]) == 2
    assert result["errors"] == errors_at(result["threshold"], in_scores, off_scores)


def test_ties_prefer_fewer_false_acceptances():
    result = recommend_threshold([0.4, 0.8], [0.6, 1.0])

    assert result["errors"] == (1, 0)


def test_errors_at_counts_both_kinds_with_lower_distance_being_more_relevant():
    assert errors_at(0.5, [0.4, 0.6], [0.45, 0.9]) == (1, 1)
