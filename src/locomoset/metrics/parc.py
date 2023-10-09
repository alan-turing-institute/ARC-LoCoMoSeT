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

from locomoset.metrics.classes import TaskSpecificMetric


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


def _lower_tri_arr(arr: np.ndarray) -> np.ndarray:
    """Takes a square 2 dimensional array and returns the lower triangular values as a
    1 dimensional array (offset from the diagonal by 1 (i.e. no diagonal values))

    Args:
        arr : 2 dimensional square array for value extraction.

    Returns:
        1 dimensional array of offset lower diagonal values from input array.
    """
    n = arr.shape[0]
    return arr[np.triu_indices(n, 1)]


class PARCMetric(TaskSpecificMetric):
    """PARC metric class"""

    def __init__(self, **metric_kwargs) -> None:
        metric_name = "parc"
        super().__init__(metric_name, **metric_kwargs)
        self.inference_type = "features"
        self.dataset_dependent = True

    def metric_function(
        self,
        features: ArrayLike,
        labels: ArrayLike,
        feat_red_dim: int = 32,
        random_state: int = None,
        scale_features: bool = True,
    ) -> float:
        """Takes computed features from model for each image in a probe data subset
        (with features as rows), and associated array of 1-hot vectors of labels,
        returning the PARC metric for transferability.

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
        dist_imgs = 1 - np.corrcoef(
            _feature_reduce(features, random_state, f=feat_red_dim)
        )
        dist_labs = 1 - np.corrcoef(labels)

        return spearmanr(_lower_tri_arr(dist_imgs), _lower_tri_arr(dist_labs))[0] * 100
