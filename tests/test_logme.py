"""
Test functions for the LogME metric (src/locomoset/metrics/logme.py)
"""

import pytest

from locomoset.metrics.logme import logme


@pytest.fixture
def logme_n_samples():
    return 3500


def test_logme_perfect_features(dummy_features_perfect, dummy_labels, logme_n_samples):
    """Test that LogME returns a value greater than zero for perfect features, with less
    than zero only seen for random features."""
    assert logme(dummy_features_perfect, dummy_labels) > 0


def test_logme_random_features(dummy_features_random, dummy_labels, logme_n_samples):
    """Test that LogME returns a value less than zero for random features, with greater
    than zero only seen for good features"""
    assert logme(dummy_features_random, dummy_labels) < 0
