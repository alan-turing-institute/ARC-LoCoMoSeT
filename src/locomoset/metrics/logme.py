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
    """LogME metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "LogME"
        super().__init__(metric_name, **metric_kwargs)
        self.inference_type = "features"
        self.dataset_dependent = True

    def metric_function(
        self, model_input: ArrayLike, dataset_input: ArrayLike, logme_bound: int = 3500
    ) -> float:
        """Compute the LogME metric based on features and labels.

        NB: LogME gives innacurate results for smaller test sizes, from empirical tests
        we recommend num_samples >= 3500

        Args:
            features: Input features of shape (num_samples, num_features)
            labels: Input labels of shape (num_samples, 1)

        Returns:
            LogME metric value.
        """
        if not isinstance(model_input, np.ndarray):
            model_input = np.asarray(model_input)
        if isinstance(dataset_input[0], str):
            dataset_input = LabelEncoder().fit_transform(dataset_input)
        if model_input.shape[0] <= logme_bound:
            warnings.warn("LogME gives innacurate results for smaller sample sizes.")
        metric = LogME()
        return metric.fit(model_input, dataset_input)
