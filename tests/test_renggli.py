"""
Test functions for the Renggli score (src/locomoset/metrics/renggli.py).
"""
import pytest

from locomoset.metrics.renggli import renggli


def test_renggli_perfect_features(dummy_features_perfect, dummy_labels):
    """
    Test that the Renggli score is 1 if the features give perfect information
    about the labels.
    """
    assert renggli(dummy_features_perfect, dummy_labels) == 1


def test_renggli_random_features(dummy_features_random, dummy_labels, dummy_n_classes):
    """
    Test that the Renggli score is around 1/n_classes (i.e. same as random) if the
    features are just random noise.
    """
    assert renggli(dummy_features_random, dummy_labels) == pytest.approx(
        1 / dummy_n_classes, rel=0.3
    )
