# Collected PARC tests
# Author: Edmund Dable-Heath
"""
    Collected tests for the PARC metric:
        - Testing perfect PARC response by basing features on labels.
        - Test random PARC response with uniform random features.
"""

import numpy as np
import pytest

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
    labs = np.zeros((n_samples, n_classes))
    for iter, item in enumerate(labels):
        labs[iter][item] = 1.0
    assert parc(labs, labs) == pytest.approx(100)


def test_parc_random_features():
    """
    Test that the PARC score is 0 if the features are random noise.
    """
    n_classes = 3
    n_features = 5
    n_samples = 1000
    labels = np.zeros((n_samples, n_classes))
    for itr, itm in enumerate(np.random.randint(0, n_classes, n_samples)):
        labels[itr][itm] = 1.0
    features = np.random.normal(size=(n_samples, n_features))
    assert parc(features, labels) == pytest.approx(0.0, abs=0.3)
