"""
Functions for the task-agnostic metrics. In particular the number of parameters of a
model as a metric.

This also includes functions for reweighting a set of metric values by the inclusion of
a task agnostic metric as a proxy for 'capactity to learn', as introduced in the
Pairwise Annotation Representation Comparison (PARC) metric for transferability paper:

Bolya, Daniel, Rohit Mittapalli, and Judy Hoffman. "Scalable diverse model selection
for accessible transfer learning." Advances in Neural Information Processing Systems
34 (2021): 19301-19312.
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
