"""
Functions for aggregating the metric for metrics score over different datasets
considered, as found in the PARC paper:

Bolya, Daniel, Rohit Mittapalli, and Judy Hoffman. "Scalable diverse model selection
for accessible transfer learning." Advances in Neural Information Processing Systems
34 (2021): 19301-19312.
"""
import numpy as np

from locomoset.metrics.library import METRIC_FOR_METRIC_FUNCTIONS


def aggregate_metric_scores(
    metric_scores: np.ndarray,
    validation_scores: np.ndarray,
    metric_for_metrics: str,
    by_dataset: bool = True,
):
    """Aggregate the metric scores by taking the mean over either varying datasets:

    (1/|T|) * sum_T corr([S], [V])

    for set of task datsets T, and models S, or over models:

    (1/|S|) * sum_M corr([T], [V])

    Args:
        metric_scores: metric scores for each model for all datasets considered,
                        (T,S) for T datasets and S models
        validation_scores: validation accuracy scores for each model, (T, S) for T
                            datasets and S models
        metric_for_metrics: which of the above metric for metrics to use
        by_dataset: controls if the aggregation is over varying datasets or varying
                    models
    """
    if not by_dataset:
        metric_scores = metric_scores.T
        validation_scores = validation_scores.T
    return np.sum(
        [
            METRIC_FOR_METRIC_FUNCTIONS[metric_for_metrics](
                metric_vals, validation_scores[idx]
            )
            for idx, metric_vals in enumerate(metric_scores)
        ]
    ) * (1 / metric_scores.shape[0])
