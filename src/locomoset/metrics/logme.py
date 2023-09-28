"""
Script for computing the LogME metric within our framework, from papers:

- LogME: Practical Assessment of Pre-trained Models for Transfer Learning
- Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs

Relies on the implementation by the authors.
"""
import warnings

from locomoset.LogME.LogME import LogME
from numpy.typing import ArrayLike


def logme(features: ArrayLike, labels: ArrayLike, random_state=None):
    """Comput the LogME metric based on features and labels.

    NB: LogME gives innacurate results for smaller test sizes, from empirical tests
    we recommend num_samples >= 3500

    Args:
        features: Input features of shape (num_samples, num_features)
        labels: Input labels of shape (num_samples, 1)

    Returns:
        LogME metric value.
    """
    if features.shape[0] <= 3500:
        warnings.warn("LogME gives innacurate results for smaller sample sizes.")
    metric = LogME()
    return metric.fit(features, labels)
