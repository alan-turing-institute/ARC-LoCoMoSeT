"""
Test functions for metrics for metrics (src/locmoset/metrics/metrics_for_metrics.py)
"""

import pytest

from locomoset.metrics.metrics_for_metrics import (
    kendall_tau_correlation,
    pearsons_correlation,
    spearmans_rank_correlation,
)


def test_spearmans_perfect_score(dummy_validation_scores, dummy_perfect_metric_scores):
    """Test the spearmans rank correlation metric for metrics for perfect ranking"""
    assert spearmans_rank_correlation(
        dummy_perfect_metric_scores, dummy_validation_scores
    ) == pytest.approx(100)


def test_spearmans_wrong_ranking_score(
    dummy_validation_scores, dummy_random_metric_scores
):
    """Test the spearmans rank correlation metric for metrics for random ranking"""
    assert spearmans_rank_correlation(
        dummy_random_metric_scores, dummy_validation_scores
    ) == pytest.approx(0, abs=2)


def test_pearsons_perfect_score(dummy_validation_scores, dummy_perfect_metric_scores):
    """Test the pearson's correlation metric for metrics for perfect ranking"""
    assert pearsons_correlation(
        dummy_perfect_metric_scores, dummy_validation_scores
    ) == pytest.approx(100, abs=1)


def test_pearsons_wrong_ranking_score(
    dummy_validation_scores, dummy_random_metric_scores
):
    """Test the spearmans rank correlation metric for metrics for random ranking"""
    assert pearsons_correlation(
        dummy_random_metric_scores, dummy_validation_scores
    ) == pytest.approx(0, abs=2)


def test_kendall_tau_perfect_score(
    dummy_validation_scores, dummy_perfect_metric_scores
):
    """Test the Kendall Tau correlation metric for metrics for perfect ranking"""
    assert kendall_tau_correlation(
        dummy_perfect_metric_scores, dummy_validation_scores
    ) == pytest.approx(100)


def test_kendall_tau_wrong_ranking_score(
    dummy_validation_scores, dummy_random_metric_scores
):
    """Test the spearmans rank correlation metric for metrics for random ranking"""
    assert kendall_tau_correlation(
        dummy_random_metric_scores, dummy_validation_scores
    ) == pytest.approx(0, abs=2)
