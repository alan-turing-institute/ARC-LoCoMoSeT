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

import numpy as np
from transformers import AutoModel


def num_params_metric(model: AutoModel) -> int:
    """Return the number of parameters of a model as a metric.

    Args:
        model: pre-loaded hugging face model from which to return the number of
        parameters.

    Returns:
        number of paramters of said model.
    """
    return model.num_parameters()


def rescale_by_params(models_and_metrics: dict) -> dict:
    """Rescale the metric value for a set of models with associated metric values by the
    number of parameters in the model, as a proxy for the models capacity to learn.

    â_i = (a_i - mean(a_i)) / std(a_i) + l_i / max(l)

    where a_i are the metric values and l_i the number of parameters. The a_i values
    would typically be Z-normed with respect to the values varied over different
    datasets, however for a single dataset here we normalise over the models.
    Normalisation is still required to make this an effective method.

    Args:
        models_and_metrics: dictionary containing 'model_name': metric_value pairs.

    Returns:
        dictionary containgin 'model_name': scaled_metric_value paris.
    """
    n_pars = {}
    for model in models_and_metrics.keys():
        model_fn = AutoModel.from_pretrained(model, num_labels=0)
        n_pars[model] = num_params_metric(model_fn)
    mean_metric = np.mean(list(models_and_metrics.values()))
    std_metric = np.std(list(models_and_metrics.values()))
    max_n_pars = np.max(list(n_pars.values()))
    return {
        model: (models_and_metrics[model] - mean_metric) / std_metric
        + n_pars[model] / max_n_pars
        for model in n_pars.keys()
    }
