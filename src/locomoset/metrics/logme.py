"""
Script for computing the LogME metric within our framework, from papers:

- LogME: Practical Assessment of Pre-trained Models for Transfer Learning
- Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs

Relies on the implementation by the authors.
"""
import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import LabelEncoder

from locomoset.LogME.LogME import LogME
from locomoset.metrics.classes import TaskSpecificMetric


class LogMEMetric(TaskSpecificMetric):
    """Wrapper around the author's implementation of the LogME metric from the papers:
    - LogME: Practical Assessment of Pre-trained Models for Transfer Learning
        (K. You et al., 2021)
    - Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs
        (K. You et al., 2021)
    """

    def __init__(self, logme_bound: int = 3500, random_state=None) -> None:
        """
        Args:
            logme_bound: LogME may give innacurate results with small sample sizes,
                print a warning if the no. of samples is smaller than this value (set
                from empirical tests on ImageNet).
        """
        super().__init__(metric_name="LogME", inference_type="features")
        self.logme_bound = logme_bound

    def metric_function(self, features: ArrayLike, labels: ArrayLike) -> float:
        """Compute the LogME metric.

        Args:
            features: Input features of shape (num_samples, num_features)
            labels: Input labels of shape (num_samples, 1)

        Returns:
            LogME metric value.
        """
        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        if isinstance(labels[0], str):
            labels = LabelEncoder().fit_transform(labels)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        if features.shape[0] <= self.logme_bound:
            warnings.warn("LogME gives innacurate results for smaller sample sizes.")
        metric = LogME()
        return metric.fit(features, labels)
