"""
Functions for calculating the metric from the paper Renggli, Cedric, et al. "Which model
to transfer? finding the needle in the growing haystack." Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 2022.
"""
from typing import Callable

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from locomoset.metrics.classes import TaskSpecificMetric


class RenggliMetric(TaskSpecificMetric):
    """Implements the metric from Renggli, Cedric, et al. "Which model to transfer?
    Finding the needle in the growing haystack." Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition. 2022.

    In short, this metric trains a linear classifier on the features of a pretrained
    model on images from the new dataset.
    """

    def __init__(
        self,
        clf: BaseEstimator | None = None,
        clf_metric: Callable = accuracy_score,
        test_size: float = 0.25,
        random_state: int = None,
    ) -> None:
        """
        Args:
            clf: Type of classifier to fit. If None defaults to a LogisticRegression
                model.
            clf_metric: Metric used to evaluate the classifier on the test set.
            test_size: Size of test set (fraction of features and labels to exclude from
                training for evaluation).
            random_state: Random state to use for the train/test split.
        """
        super().__init__(
            metric_name="renggli", inference_type="features", random_state=random_state
        )

        if clf is None:
            clf = Pipeline(
                (
                    ("scaler", StandardScaler()),
                    ("logistic", LogisticRegression(random_state=self.random_state)),
                )
            )
        self.clf = clf
        self.clf_metric = clf_metric
        self.test_size = test_size

    def metric_function(self, features: ArrayLike, labels: ArrayLike) -> float:
        """Trains and evaluates a classifier on input features and labels. In the
        Renggli paper the features are from a classifier with its head removed.

        Args:
            features: Input features of shape (num_samples, num_features).
            labels: Input labels of shape (num_samples, 1).

        Returns:
            Metric score of the trained classifier on the test set.
        """
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.clf.fit(feats_train, labels_train)
        pred_test = self.clf.predict(feats_test)
        return self.clf_metric(labels_test, pred_test)
