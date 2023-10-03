"""
    Base metric class for unifying the method.
"""

from locomoset.metrics.library import METRIC_FUNCTIONS


class Metric:

    """Base class for metrics"""

    def __init__(self, metric_name, **metric_kwargs) -> None:
        self.metric_name = metric_name
        self.inference_type = None
        self.metric_function = METRIC_FUNCTIONS[self.metric_name]
        self.dataset_dependent = False
        self.metric_kwargs = metric_kwargs

    def get_inference_type(self):
        return self.inference_type

    def fit_metric(self, model_fn, model_input, dataset_input):
        if not self.dataset_dependent:
            return self.metric_function(
                model_input, dataset_input, **self.metric_kwargs
            )
        else:
            return self.metric_function(model_fn, **self.metric_kwargs)


class NoParsMetric(Metric):

    """Number of parameters metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "n_pars"
        super().__init__(metric_name, **metric_kwargs)


class RenggliMetric(Metric):

    """Renggli metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "renggli"
        super().__init__(metric_name, **metric_kwargs)
        self.inference_type = "features"


class PARCMetric(Metric):

    """PARC metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "parc"
        super().__init__(metric_name, **metric_kwargs)
        self.inference_type = "features"


class LogMEMetric(Metric):

    """LogME metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "LogME"
        super().__init__(metric_name, **metric_kwargs)
        self.inference_type = "features"
