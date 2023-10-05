"""
Functions for the task-agnostic metrics. In particular the number of parameters of a
model as a metric.
"""

from transformers.modeling_utils import PreTrainedModel

from locomoset.metrics.classes import TaskAgnosticMetric


class NumParsMetric(TaskAgnosticMetric):

    """Number of parameters metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "n_pars"
        super().__init__(metric_name, **metric_kwargs)

    def metric_function(self, model: PreTrainedModel) -> int:
        """Return the number of parameters of a model as a metric.

        Args:
            model: pre-loaded hugging face model from which to return the number of
            parameters.

        Returns:
            number of paramters of said model.
        """
        return model.num_parameters()
