"""
Test functions for the NCE metric (src/locomoset/metrics/nce.py)
"""

import numpy as np
import pytest

from locomoset.metrics.nce import nce


def test_nce_perfect_features(dummy_labels):
    """Test that NCE returns approximately zero for perfect features"""
    assert nce(dummy_labels, dummy_labels) == pytest.approx(0)


def test_nce_random_features(dummy_labels, dummy_n_classes, test_n_samples):
    """Test that NCE returns a negative value

    New rng defined to ensure different labels sampled"""
    rng = np.random.default_rng(43)
    random_labels = rng.integers(0, dummy_n_classes, test_n_samples)
    assert nce(dummy_labels, random_labels) < -1
