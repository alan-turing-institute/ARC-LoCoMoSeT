"""
Test for functions for aggregating metric for metrics scores
(src/locomoset/metrics/aggregate_metric_for_metrics.py)
"""

import numpy as np
import pytest

from locomoset.metrics.aggregate_metric_for_metrics import aggregate_metric_scores


def test_aggregate_metric_for_metrics_perfect_scores_spearmans(
    dummy_perfect_metric_scores, dummy_validation_scores, test_n_features
):
    """Test for aggregation of metric of metrics over datasets or models for perfect
    metric scores
    """
    collected_metric_scores = np.array(
        [dummy_perfect_metric_scores for _ in range(test_n_features)]
    )
    collected_validation_scores = np.array(
        [dummy_validation_scores for _ in range(test_n_features)]
    )
    assert aggregate_metric_scores(
        collected_metric_scores,
        collected_validation_scores,
        "spearmans",
        by_dataset=True,
    ) == pytest.approx(100)


def test_aggregate_metric_for_metrics_random_scores_spearmans(
    dummy_random_metric_scores, dummy_validation_scores, test_n_features
):
    """Test for aggregation of metric of metrics over datasets or models for perfect
    metric scores
    """
    collected_metric_scores = np.array(
        [dummy_random_metric_scores for _ in range(test_n_features)]
    )
    collected_validation_scores = np.array(
        [dummy_validation_scores for _ in range(test_n_features)]
    )
    assert aggregate_metric_scores(
        collected_metric_scores,
        collected_validation_scores,
        "spearmans",
        by_dataset=True,
    ) == pytest.approx(0, abs=1)


def test_aggregate_metric_for_metrics_perfect_scores_pearsons(
    dummy_perfect_metric_scores, dummy_validation_scores, test_n_features
):
    """Test for aggregation of metric of metrics over datasets or models for perfect
    metric scores
    """
    collected_metric_scores = np.array(
        [dummy_perfect_metric_scores for _ in range(test_n_features)]
    )
    collected_validation_scores = np.array(
        [dummy_validation_scores for _ in range(test_n_features)]
    )
    assert aggregate_metric_scores(
        collected_metric_scores,
        collected_validation_scores,
        "pearsons",
        by_dataset=True,
    ) == pytest.approx(100, abs=1)


def test_aggregate_metric_for_metrics_random_scores_pearsons(
    dummy_random_metric_scores, dummy_validation_scores, test_n_features
):
    """Test for aggregation of metric of metrics over datasets or models for perfect
    metric scores
    """
    collected_metric_scores = np.array(
        [dummy_random_metric_scores for _ in range(test_n_features)]
    )
    collected_validation_scores = np.array(
        [dummy_validation_scores for _ in range(test_n_features)]
    )
    assert aggregate_metric_scores(
        collected_metric_scores,
        collected_validation_scores,
        "pearsons",
        by_dataset=True,
    ) == pytest.approx(0, abs=1)


def test_aggregate_metric_for_metrics_perfect_scores_kendall_tau(
    dummy_perfect_metric_scores, dummy_validation_scores, test_n_features
):
    """Test for aggregation of metric of metrics over datasets or models for perfect
    metric scores
    """
    collected_metric_scores = np.array(
        [dummy_perfect_metric_scores for _ in range(test_n_features)]
    )
    collected_validation_scores = np.array(
        [dummy_validation_scores for _ in range(test_n_features)]
    )
    assert aggregate_metric_scores(
        collected_metric_scores,
        collected_validation_scores,
        "kendall_tau",
        by_dataset=True,
    ) == pytest.approx(100)


def test_aggregate_metric_for_metrics_random_scores_kendall_tau(
    dummy_random_metric_scores, dummy_validation_scores, test_n_features
):
    """Test for aggregation of metric of metrics over datasets or models for perfect
    metric scores
    """
    collected_metric_scores = np.array(
        [dummy_random_metric_scores for _ in range(test_n_features)]
    )
    collected_validation_scores = np.array(
        [dummy_validation_scores for _ in range(test_n_features)]
    )
    assert aggregate_metric_scores(
        collected_metric_scores,
        collected_validation_scores,
        "kendall_tau",
        by_dataset=True,
    ) == pytest.approx(0, abs=1)
