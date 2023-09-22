"""
Script for computing the LogME metric within our framework, from papers:

- LogME: Practical Assessment of Pre-trained Models for Transfer Learning
- Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs

Relies on the implementation by the authors.
"""

from numpy.typing import ArrayLike

from LogME.LogME import LogME


def logme(features: ArrayLike, labels: ArrayLike, random_state=None):
    """Comput the LogME metric based on features and labels.

    Args:
        features: Input features of shape (num_samples, num_features)
        labels: Input labels of shape (num_samples, 1)
    """
    # Check there enough samples
    assert features.shape[0] >= 3500, ValueError(
        "Requires >3500 samples to return a consistent result"
    )

    metric = LogME()
    return metric.fit(features, labels)
