"""
Test functions for the LogME metric (src/locomoset/metrics/logme.py)
"""

import numpy as np
import pytest

from locomoset.metrics.logme import logme


def test_logme_n_samples_catch(dummy_features_perfect, dummy_labels):
    """Test that the LogME metric catches that the sample size isn't great enough to
    compute a consistent score
    """
    with pytest.raises(AssertionError):
        logme(dummy_features_perfect, dummy_labels)


def test_logme_perfect_features(dummy_features_perfect, dummy_labels):
    """Test that LogME returns a value greater than zero for perfect features, with less
    than zero only seen for random features"""
    features = np.concatenate(
        [
            np.asarray(dummy_features_perfect)
            for _ in range((3500 // dummy_features_perfect.shape[0] + 1))
        ],
        axis=0,
    )
    labels = np.concatenate(
        [
            dummy_labels.reshape(1, -1)
            for _ in range((3500 // dummy_labels.shape[0] + 1))
        ],
        axis=1,
    )
    assert logme(features, labels) > 0


def test_logme_random_features(dummy_features_random, dummy_labels):
    """Test that LogME returns a value less than zero for random features, with greater
    than zero only seen for good features"""
    features = np.concatenate(
        [
            np.asarray(dummy_features_random)
            for _ in range((3500 // dummy_features_random.shape[0] + 1))
        ],
        axis=0,
    )
    labels = np.concatenate(
        [
            dummy_labels.reshape(1, -1)
            for _ in range((3500 // dummy_labels.shape[0] + 1))
        ],
        axis=1,
    )
    assert logme(features, labels) < 0
