"""
Test functions for the LogME metric (src/locomoset/metrics/logme.py)
"""


from locomoset.metrics.logme import logme


def test_logme_perfect_features(dummy_features_perfect, dummy_labels):
    """Test that LogME returns a value greater than zero for perfect features, with less
    than zero only seen for random features."""
    assert logme(dummy_features_perfect, dummy_labels) > 0


def test_logme_random_features(dummy_features_random, dummy_labels):
    """Test that LogME returns a value less than zero for random features, with greater
    than zero only seen for good features"""
    assert logme(dummy_features_random, dummy_labels) < 0
