"""
    Base metric class for unifying the method.
"""

from abc import ABC, abstractmethod


class Metric(ABC):

    """Base class for metrics"""

    def __init__(self, metric_name, **metric_kwargs) -> None:
        self.metric_name = metric_name
        self.inference_type = None
        self.metric_kwargs = metric_kwargs

    @abstractmethod
    def metric_function(self, *args, **kwargs):
        """Base metric function, reimplement for each metric class"""
        raise NotImplementedError("Implement metric function method!")

    @abstractmethod
    def fit_metric(self, *args, **kwargs):
        """Base fit metric function, reimplement for each subclass"""
        raise NotImplementedError("Implement fit metric function!")


class TaskAgnosticMetric(Metric):

    """Base class for task agnostic metrics, for which the metric function input is
    solely the model function."""

    def __init__(self, metric_name, **metric_kwargs) -> None:
        super().__init__(metric_name, **metric_kwargs)
        self.dataset_dependent = False

    def fit_metric(self, model_fn):
        return self.metric_function(model_fn)


class TaskSpecificMetric(Metric):

    """Base class for task specific metric, for which the input is of shape:

    (model_input, dataset_input, **kwargs)

    """

    def __init__(self, metric_name, **metric_kwargs) -> None:
        super().__init__(metric_name, **metric_kwargs)
        self.dataset_dependent = True

    def fit_metric(self, model_input, dataset_input):
        return self.metric_function(model_input, dataset_input, **self.metric_kwargs)
