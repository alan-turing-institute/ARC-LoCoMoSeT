"""
Test functions for the Renggli score (src/locomoset/metrics/renggli.py).
"""
import pytest

from locomoset.metrics.renggli import RenggliMetric


def test_renggli_class_perfect_features(
    dummy_features_perfect, dummy_labels, test_seed
):
    """Test the Renggli metric class for perfect features"""
    metric = RenggliMetric(**{"random_state": test_seed})
    assert metric.metric_name == "renggli"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert (
        metric.fit_metric(
            dummy_features_perfect,
            dummy_labels,
        )
        == 1
    )


def test_renggli_class_random_features(
    dummy_features_random, dummy_labels, dummy_n_classes, test_seed
):
    """Test the Renggli metric class for random features"""
    metric = RenggliMetric(**{"random_state": test_seed})
    assert metric.metric_name == "renggli"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(dummy_features_random, dummy_labels) == pytest.approx(
        1 / dummy_n_classes, rel=0.2
    )
