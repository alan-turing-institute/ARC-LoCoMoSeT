"""
Script for computing the NCE metric within our framework, from paper:

- Transferability and Hardness of Supervised Classification Tasks

Relies on the implementation by the LogME authors (which corrects the original
implementation).
"""
from numpy.typing import ArrayLike

from locomoset.LogME.NCE import NCE


def nce(pred_labels: ArrayLike, labels: ArrayLike, random_state=None) -> float:
    """Function for computing the negative conditionaly entropy metric.

    Args:
        pred_labels: predicted labels from model, (n_samples, 1)
        labels: ground truth labels, (n_samples, 1)
        random_state: not necessary, here to fit the pipeline

    Returns:
        NCE metric value
    """
    return NCE(labels, pred_labels)
