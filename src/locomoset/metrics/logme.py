"""
Script for computing the LogME metric within our framework, from papers:

- LogME: Practical Assessment of Pre-trained Models for Transfer Learning
- Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs

Relies on the implementation by the authors.
"""
import warnings

from numpy.typing import ArrayLike

from locomoset.LogME.LogME import LogME


def logme(model_input: ArrayLike, dataset_input: ArrayLike, **metric_kwargs) -> float:
    """Comput the LogME metric based on features and labels.

    NB: LogME gives innacurate results for smaller test sizes, from empirical tests
    we recommend num_samples >= 3500

    Args:
        features: Input features of shape (num_samples, num_features)
        labels: Input labels of shape (num_samples, 1)

    Returns:
        LogME metric value.
    """
    LogME_bound = (
        metric_kwargs["LogME_bound"] if "LogME_bound" in metric_kwargs.keys() else 3500
    )
    if model_input.shape[0] <= LogME_bound:
        warnings.warn("LogME gives innacurate results for smaller sample sizes.")
    metric = LogME()
    return metric.fit(model_input, dataset_input)
