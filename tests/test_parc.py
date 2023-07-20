"""
Test functions for the PARC score (src/locomoset/metrics/parc.py).
"""
import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from locomoset.metrics.parc import parc


def test_parc_perfect_features():
    """
    Test that the PARC score is 100 if the features give perfect information about
    the labels.

    NB: This is without applying feature reduction to PARC.
    """
    n_classes = 3
    n_samples = 100
    labels = np.random.randint(0, n_classes, n_samples)
    # use one hot encoded labels as the input features (giving the classifier perfect
    # information to distinguish between classes)
    features = OneHotEncoder(sparse_output=False).fit_transform(
        labels.reshape((n_samples, 1))
    )
    assert parc(features, labels) == pytest.approx(100)


def test_parc_random_features():
    """
    Test that the PARC score is 0 if the features are random noise.
    """
    n_classes = 3
    n_features = 5
    n_samples = 1000
    labels = np.random.randint(0, n_classes, n_samples)
    features = np.random.normal(size=(n_samples, n_features))
    assert parc(features, labels) == pytest.approx(0.0, abs=0.3)
