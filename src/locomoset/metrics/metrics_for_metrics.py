"""
Metrics for rating how effective the metrics we're considering are, with the three most
common metrics from this seen in the literature are:

- Spearman's rank correlation
- Pearson's correlation
- Kendall Tau correlation

We also include the an aggregate (mean) correlation over datasets considered for each of
these.
"""

from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr


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
