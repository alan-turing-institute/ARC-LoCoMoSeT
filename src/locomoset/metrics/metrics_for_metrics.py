"""
Metrics for rating how effective the metrics we're considering are.
"""

from scipy.stats import spearmanr


def spearmans_rank_correlation(
    metric_scores: list(float), validation_scores: list(float)
):
    """Return the spearman's rank correlation between the scores for a collection of
    models from a particular metric and the post-fine tuning validation scores for those
    same models.

    Args:
        metric_scores: metric scores for each model
        validation_scores: validation accuracy scores for each model
    """
    return spearmanr(metric_scores, validation_scores)[0] * 100
