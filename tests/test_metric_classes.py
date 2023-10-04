"""
    Test functions for metric classes (src/locomoset/metric/metric_classes.py)
"""

import pytest

from locomoset.metrics.metric_classes import (
    LogMEMetric,
    NumParsMetric,
    PARCMetric,
    RenggliMetric,
)


def test_n_pars_class(dummy_model_head, dummy_features_perfect, dummy_labels):
    """Test the task agonistic metric class"""
    metric = NumParsMetric()
    assert metric.metric_name == "n_pars"
    assert metric.inference_type is None
    assert metric.dataset_dependent is False
    assert (
        metric.fit_metric(dummy_model_head, dummy_features_perfect, dummy_labels)
        == 1686
    )


def test_renggli_class_perfect_features(
    dummy_model_head, dummy_features_perfect, dummy_labels, test_seed
):
    """Test the Renggli metric class for perfect features"""
    metric = RenggliMetric(**{"random_state": test_seed})
    assert metric.metric_name == "renggli"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert (
        metric.fit_metric(
            dummy_model_head,
            dummy_features_perfect,
            dummy_labels,
        )
        == 1
    )


def test_renggli_class_random_features(
    dummy_model_head, dummy_features_random, dummy_labels, dummy_n_classes, test_seed
):
    """Test the Renggli metric class for random features"""
    metric = RenggliMetric(**{"random_state": test_seed})
    assert metric.metric_name == "renggli"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(
        dummy_model_head, dummy_features_random, dummy_labels
    ) == pytest.approx(1 / dummy_n_classes, rel=0.2)


def test_parc_class_perfect_features(
    dummy_model_head, dummy_features_perfect, dummy_labels, test_seed
):
    """Test PARC metric class for perfect features"""
    metric = PARCMetric(
        **{"random_state": test_seed, "feat_red_dim": None, "scale_features": False}
    )
    assert metric.metric_name == "parc"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(
        dummy_model_head, dummy_features_perfect, dummy_labels
    ) == pytest.approx(100)


def test_parc_class_random_features(
    dummy_model_head, dummy_features_random, dummy_labels, test_seed
):
    """Test PARC metric class for random features"""
    metric = PARCMetric(
        **{"random_state": test_seed, "feat_red_dim": None, "scale_features": False}
    )
    assert metric.metric_name == "parc"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(
        dummy_model_head, dummy_features_random, dummy_labels
    ) == pytest.approx(0.0, abs=0.2)


def test_logme_class_perfect_features(
    dummy_model_head, dummy_features_perfect, dummy_labels
):
    """Test the LogME metric class for perfect features"""
    metric = LogMEMetric()
    assert metric.metric_name == "LogME"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(dummy_model_head, dummy_features_perfect, dummy_labels) > 0


def test_logme_class_random_features(
    dummy_model_head, dummy_features_random, dummy_labels
):
    """Test the LogME metric class for random features"""
    metric = LogMEMetric()
    assert metric.metric_name == "LogME"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(dummy_model_head, dummy_features_random, dummy_labels) < 0
