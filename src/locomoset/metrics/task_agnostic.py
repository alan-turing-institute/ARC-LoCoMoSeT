"""
Functions for the task-agnostic metrics. In particular the number of parameters of a
model as a metric.
"""
from transformers.modeling_utils import PreTrainedModel

from locomoset.metrics.classes import TaskAgnosticMetric


class NumParsMetric(TaskAgnosticMetric):
    """Number of parameters metric"""

    def __init__(self, random_state: int | None = None) -> None:
        super().__init__(
            metric_name="n_pars", inference_type="model", random_state=random_state
        )

    def metric_function(self, model: PreTrainedModel) -> int:
        """Returns the number of parameters in a model.

        Args:
            model: Pre-loaded hugging face model

        Returns:
            Number of paramters in said model.
        """
        return model.num_parameters()
