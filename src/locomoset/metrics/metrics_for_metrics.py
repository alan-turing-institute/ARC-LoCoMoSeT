"""
Metrics for rating how effective the metrics we're considering are, with the three most
common metrics from this seen in the literature are:

- Spearman's rank correlation
- Pearson's correlation
- Kendall Tau correlation

We also include the an aggregate (mean) correlation over datasets considered for each of
these.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import kendalltau, pearsonr, spearmanr

from locomoset.metrics.library import METRIC_FOR_METRIC_FUNCTIONS


def spearmans_rank_correlation(metric_scores: ArrayLike, validation_scores: ArrayLike):
    """Return the Spearman's rank correlation between the scores for a collection of
    models from a particular metric and the post-fine tuning validation scores for those
    same models.

    Args:
        metric_scores: metric scores for each model, (s, ) for s models
        validation_scores: validation accuracy scores for each model, (s, ) for s models
    """
    return spearmanr(metric_scores, validation_scores)[0] * 100


def pearsons_correlation(metric_scores: ArrayLike, validation_scores: ArrayLike):
    """Return Pearson's correlation between the scores for a collection of models from a
    particular metric and the post-fine tuning validation scores for those same models.

    Args:
        metric_scores: metric scores for each model, (s, ) for s models
        validation_scores: validation accuracy scores for each model, (s, ) for s models
    """
    return pearsonr(metric_scores, validation_scores)[0] * 100


def kendall_tau_correlation(metric_scores: ArrayLike, validation_scores: ArrayLike):
    """Return the Kendall Tau correlation between the scores for a collection of models
    from a particular metric and the post-fine tuning validation scores for those same
    models.

    Args:
        metric_scores: metric scores for each model, (s, ) for s models
        validation_scores: validation accuracy scores for each model, (s, ) for s models
    """
    return kendalltau(metric_scores, validation_scores)[0] * 100


def aggregate_metric_scores(
    metric_scores: ArrayLike,
    validation_scores: ArrayLike,
    metric_for_metrics: str,
    by_dataset: bool = True,
):
    """Aggregate the metric scores by taking the mean over either varying datasets:

    (1/|T|) * sum_T corr([S], [V])

    for set of task datsets T, and models S, or over models:

    (1/|S|) * sum_M corr([T], [V])

    Args:
        metric_scores: metric scores for each model for all datasets considered, (T, S)
                        for T datasets and S models
        validation_scores: validation accuracy scores for each model, (S, ) for S models
        metric_for_metrics: which of the above metric for metrics to use
        by_dataset: controls if the aggregation is over varying datasets or varying
                    models
    """
    if not by_dataset:
        metric_scores = metric_scores.T
    return np.sum(
        [
            METRIC_FOR_METRIC_FUNCTIONS[metric_for_metrics](
                metric_vals, validation_scores
            )
            for metric_vals in metric_scores
        ]
    ) * (1 / metric_scores.shape[0])
