"""
Test functions for the PARC score (src/locomoset/metrics/parc.py).
"""
import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from locomoset.metrics.parc import _feature_reduce, _lower_tri_arr, parc


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
    assert parc(features, labels, feat_red_dim=None) == pytest.approx(100)


def test_parc_random_features():
    """
    Test that the PARC score is 0 if the features are random noise.
    """
    n_classes = 3
    n_features = 5
    n_samples = 1000
    rng = np.random.default_rng(42)
    labels = rng.integers(0, n_classes, n_samples)
    features = rng.normal(size=(n_samples, n_features))
    assert parc(features, labels) == pytest.approx(0.0, abs=0.3)


def test_lower_tri():
    """
    Test that the lower triangular values (offset from diag by one) values of a square
    matrix are being correctly pulled out.

    NB: This actually pulls out the upper triangular values but applies to a symmetric
    matrix by definition.
    """
    n = 5

    # create n^2 array of numbers from 1 -> 25
    arr = np.array([i + 1 for i in range(n**2)]).reshape((n, n))

    # analytical sum of upper triangular values for above matrix
    s = (n**2 * (n**2 + 1)) / 2 - sum(
        [(k * (k * (2 * n + 1) - 2 * n + 1)) / 2 for k in range(1, n + 1)]
    )

    assert s == sum(_lower_tri_arr(arr))


def test_feature_reduce():
    """
    Test that the feature reduction method returns the features if f = None is given and
    raises an exception if f > min(features.shape)
    """
    n_features = 5
    n_samples = 1000
    rng = np.random.default_rng(42)
    features = rng.normal(size=(n_samples, n_features))

    assert np.all(_feature_reduce(features, 42, None)) == np.all(features)

    with pytest.raises(ValueError):
        _feature_reduce(features, 42, 10000)
