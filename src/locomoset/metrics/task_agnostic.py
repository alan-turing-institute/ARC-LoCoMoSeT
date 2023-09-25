"""
Functions for the task-agnostic metrics. In particular the number of parameters of a
model as a metric.
"""

from transformers.modeling_utils import PreTrainedModel


def num_params_metric(model: PreTrainedModel) -> int:
    """Return the number of parameters of a model as a metric.

    Args:
        model: pre-loaded hugging face model from which to return the number of
        parameters.

    Returns:
        number of paramters of said model.
    """
    return model.num_parameters()
