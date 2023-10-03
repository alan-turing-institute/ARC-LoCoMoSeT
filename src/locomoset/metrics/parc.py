"""
Functions for calculating the Pairwise Annotation Representation Comparison (PARC)
metric for transferability:

Bolya, Daniel, Rohit Mittapalli, and Judy Hoffman. "Scalable diverse model selection
for accessible transfer learning." Advances in Neural Information Processing Systems
34 (2021): 19301-19312.
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _feature_reduce(features: np.ndarray, random_state: int, f: int = 32) -> np.ndarray:
    """Use PCA to reduce the dimensionality of features.

    Args:
        features: features on which to reduce.
        f: dimension to reduce down to.

    Returns:
        Reduced features array.
    """
    if f is None:
        return features

    if f > min(features.shape):
        raise ValueError(
            "Reduced dimension should not be more than minimum features dimension."
        )

    return PCA(
        n_components=f,
        svd_solver="randomized",
        iterated_power=1,
        random_state=random_state,
    ).fit_transform(features)


def _lower_tri_arr(arr: ArrayLike):
    """Takes a square 2 dimensional array and returns the lower triangular values as a
    1 dimensional array (offset from the diagonal by 1 (i.e. no diagonal values))

    Args:
        arr : 2 dimensional square array for value extraction.

    Returns:
        1 dimensional array of offset lower diagonal values from input array.
    """
    n = arr.shape[0]
    return arr[np.triu_indices(n, 1)]


def parc(
    features: ArrayLike,
    labels: ArrayLike,
    feat_red_dim: int = 32,
    random_state: int = None,
    scale_features: bool = True,
) -> float:
    """Takes computed features from model for each image in a probe data subset (with
    features as rows), and associated array of 1-hot vectors of labels, returning the
    PARC metric for transferability.

    Args:
        features: Features from model for each image in probe dataset of
            shape (num_samples, num_features).
        labels: Input labels of shape (num_samples, 1).
        feat_red_dim: If set, feature reduction dimension.
        random_state: Random state for dimensionality reduction.
        scale_features: If True, use StandardScaler to convert features to have mean
            zero and standard deviation one before computing PARC.

    Returns:
        PARC score for transferability.
    """
    np.random.seed(random_state)
    if not isinstance(features, np.ndarray):
        features = np.asarray(features)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    labels = OneHotEncoder(sparse_output=False).fit_transform(
        labels.reshape((len(labels), 1))
    )
    if scale_features:
        features = StandardScaler().fit_transform(features)
    dist_imgs = 1 - np.corrcoef(_feature_reduce(features, random_state, f=feat_red_dim))
    dist_labs = 1 - np.corrcoef(labels)

    return spearmanr(_lower_tri_arr(dist_imgs), _lower_tri_arr(dist_labs))[0] * 100


def parc_class_function(
    model_input: ArrayLike, dataset_input: ArrayLike, **metric_kwargs
) -> float:
    """Takes computed features from model for each image in a probe data subset (with
    features as rows), and associated array of 1-hot vectors of labels, returning the
    PARC metric for transferability.

    Args:
        features: Features from model for each image in probe dataset of
            shape (num_samples, num_features).
        labels: Input labels of shape (num_samples, 1).
        metric_kwargs:
            feat_red_dim: If set, feature reduction dimension.
            random_state: Random state for dimensionality reduction.
            scale_features: If True, use StandardScaler to convert features to have mean
                zero and standard deviation one before computing PARC.

    Returns:
        PARC score for transferability.
    """
    random_state = (
        metric_kwargs["random_state"]
        if "random_state" in metric_kwargs.keys()
        else None
    )
    np.random.seed(random_state)

    if not isinstance(model_input, np.ndarray):
        model_input = np.asarray(model_input)
    if not isinstance(dataset_input, np.ndarray):
        dataset_input = np.asarray(dataset_input)
    dataset_input = OneHotEncoder(sparse_output=False).fit_transform(
        dataset_input.reshape((len(dataset_input), 1))
    )

    scale_features = (
        metric_kwargs["scale_features"]
        if "scale_features" in metric_kwargs.keys()
        else True
    )
    if scale_features:
        model_input = StandardScaler().fit_transform(model_input)

    feat_red_dim = (
        metric_kwargs["feat_red_dim"] if "feat_red_dim" in metric_kwargs.keys() else 32
    )
    dist_imgs = 1 - np.corrcoef(
        _feature_reduce(model_input, random_state, f=feat_red_dim)
    )
    dist_labs = 1 - np.corrcoef(dataset_input)

    return spearmanr(_lower_tri_arr(dist_imgs), _lower_tri_arr(dist_labs))[0] * 100
