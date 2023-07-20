"""
Functions for calculating the metric from the paper Renggli, Cedric, et al. "Which model
to transfer? finding the needle in the growing haystack." Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 2022.
"""
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def renggli_score(
    features: ArrayLike,
    labels: ArrayLike,
    clf: BaseEstimator = Pipeline(
        [("scaler", StandardScaler()), ("logistic", LogisticRegression())]
    ),
    metric: Callable = accuracy_score,
    test_size: float = 0.25,
    random_state: float = None,
) -> float:
    """Trains and evaluates a classifier on input features and labels.

    In the Renggli paper the features are from a classifier with its head removed, and
    the trained classifier is a LogisticRegression model.

    See Renggli, Cedric, et al. "Which model to transfer? finding the needle in the
    growing haystack." Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition. 2022.

    Args:
        features: Input features of shape (num_samples, num_features).
        labels: Input labels of shape (num_samples, 1).
        clf: Type of classifier to fit.
        metric: Metric used to evaluate the classifier on the test set.
        test_size: Size of test set (fraction of features and labels to exclude from
            training for evaluation).
        random_state: Random state to use for the train/test split.

    Returns:
        Metric score of the trained classifier on the test set.
    """

    np.random.seed(random_state)
    feats_train, feats_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    clf.fit(feats_train, labels_train)
    pred_test = clf.predict(feats_test)
    return metric(labels_test, pred_test)
