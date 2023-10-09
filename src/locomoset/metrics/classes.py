"""
Base metric class for unifying the method.
"""
from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Base class for metrics.

    Attributes:
        metric_name: Name of the metric
        inference_type: Type of inference - either 'features', 'model' or None:
            - 'features': The model_input passed to the metric is features generated
                    by the model on the test dataset
            - 'model': The model_input passed to the metric is the model itself
                (with its classification head removed)
            - None: model_input is set to None.
        dataset_dependent: Whether the metric requires information about the dataset or
            only uses the model.
        random_state: Random seed to set prior to metric computation.
    """

    def __init__(
        self,
        metric_name: str,
        inference_type: str | None,
        dataset_dependent: bool,
        random_state: int | None = None,
    ) -> None:
        self.metric_name = metric_name
        self.inference_type = inference_type
        self.dataset_dependent = dataset_dependent
        self.random_state = random_state

    def set_random_state(self) -> None:
        """Set the random state (should be called in fit_metric before calling
        metric_function)
        """
        np.random.seed(self.random_state)

    @abstractmethod
    def metric_function(self, *args, **kwargs) -> float | int:
        """Base metric function, reimplement for each metric class. Can include
        different number arguments depending on the metric subclass."""

    @abstractmethod
    def fit_metric(self, model_input, dataset_input) -> float | int:
        """Base fit metric function, reimplement for each subclass (metric category).
        Must always take two inputs, and takes care of setting random seeds and calling
        metric_function correctly for each subclass."""


class TaskAgnosticMetric(Metric):
    """Base class for task agnostic metrics, for which the metric function input is
    solely the model function."""

    def __init__(
        self,
        metric_name: str,
        inference_type: str | None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            metric_name,
            inference_type=inference_type,
            dataset_dependent=False,
            random_state=random_state,
        )

    @abstractmethod
    def metric_function(self, model_input) -> float | int:
        """TaskAgnosticMetric classes should implement a metric_function that takes
        only a a model, or model-derived quantity/function, as input."""

    def fit_metric(self, model_input, dataset_input=None) -> float | int:
        """Compute the metric value. The dataset_input is not used for
        TaskAgnosticMetric instances but included for compatibility with the parent
        Metric class."""
        self.set_random_state()
        return self.metric_function(model_input)


class TaskSpecificMetric(Metric):
    """Base class for task specific metric, for which the input is of shape:
    (model_input, dataset_input, **kwargs)
    """

    def __init__(
        self, metric_name: str, inference_type: str, random_state: int | None = None
    ) -> None:
        super().__init__(
            metric_name=metric_name,
            inference_type=inference_type,
            dataset_dependent=True,
            random_state=random_state,
        )

    @abstractmethod
    def metric_function(self, model_input, dataset_input) -> float | int:
        """TaskSpecificMetric classes should implement a metric_function that takes both
        a model-derived input (e.g. features generated on new samples) and a dataset
        input (e.g. ground truth labels for each sample)."""

    def fit_metric(self, model_input, dataset_input) -> float | int:
        self.set_random_state()
        return self.metric_function(model_input, dataset_input)
