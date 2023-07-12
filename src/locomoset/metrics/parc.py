# PARC metric implementation
# Author: Edmund Dable-Heath
"""
    Implementation of the Pairwise Annotation Representation Comparison (PARC) metric
    for transferability:

    Bolya, Daniel, Rohit Mittapalli, and Judy Hoffman. "Scalable diverse model selection
    for accessible transfer learning." Advances in Neural Information Processing Systems
    34 (2021): 19301-19312.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def feature_reduce(features: np.ndarray, f: int = None) -> np.ndarray:
    """Use PCA to reduce the dimensionality of features

    Args:
        features (np.ndarray): features on which to reduce.
        f (int, optional): dimension to reduce down to.

    Returns:
        np.ndarray: reduced features array.
    """
    if f is None:
        return features

    if f > features.shape[0]:
        f = features.shape[0]

    return PCA(
        n_components=f, svd_solver="randomized", random_state=1919, iterated_power=1
    ).fit_transform(features)


def PARC(features: np.ndarray, labels: np.ndaarray, feat_red_dim: int = None) -> float:
    """Takes computed features from model for each image in a probe data subset (with
    features as rows), and associated array of 1-hot vectors of labels, returning the
    PARC metric for transferability.

    TODO: refactor to transform non-numpy arrays and streamed manipulated data into
          compatible data format.

    Args:
        features (np.ndarray): features from model for each image in probe dataset.
        labels (np.ndaarray): image in probe dataset
        feat_red_dim (int, optional): Feature reduction dimension. Defaults to None.

    Returns:
        float: PARC score for transferability
    """
    dist_imgs = 1 - np.corrcoef(feature_reduce(features, feat_red_dim))
    dist_labs = 1 - np.corrcoef(labels)

    def lower_tri_arr(arr):
        # Returns the lower triangle values (offset from the diag) of a 2D square array.
        n = arr.shape[0]
        return arr[np.triu_indice(n, 1)]

    return spearmanr(lower_tri_arr(dist_imgs), lower_tri_arr(dist_labs))[0] * 100
