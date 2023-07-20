"""
Test functions for the Renggli score (src/locomoset/metrics/renggli.py).
"""
import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from locomoset.metrics.renggli import renggli_score


def test_renggli_perfect_features():
    """
    Test that the Renggli score is 1 if the features give perfect information
    about the labels.
    """
    n_classes = 3
    n_samples = 100
    labels = np.random.randint(0, n_classes, n_samples)
    # use one hot encoded labels as the input features (giving the classifier perfect
    # information to distinguish between classes)
    features = OneHotEncoder(sparse_output=False).fit_transform(
        labels.reshape((n_samples, 1))
    )
    assert renggli_score(features, labels) == 1


def test_renggli_random_features():
    """
    Test that the Renggli score is around 1/n_classes (i.e. same as random) if the
    features are just random noise.
    """
    n_classes = 3
    n_features = 5
    n_samples = 1000
    labels = np.random.randint(0, n_classes, n_samples)
    features = np.random.normal(size=(n_samples, n_features))
    assert renggli_score(features, labels) == pytest.approx(1 / n_classes, rel=0.3)
