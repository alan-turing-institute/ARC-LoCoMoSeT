"""
Test functions for metrics for metrics
"""
import pytest

from locomoset.metrics.metrics_for_metrics import spearmans_rank_correlation


def test_spearmans_perfect_score(dummy_validation_scores):
    """Test the spearmans rank correlation metrics for metrics for perfect ranking"""
    assert spearmans_rank_correlation(
        dummy_validation_scores, dummy_validation_scores
    ) == pytest.approx(100)
