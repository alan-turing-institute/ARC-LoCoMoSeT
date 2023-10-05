"""
Test functions for the LogME metric (src/locomoset/metrics/logme.py)
"""


from locomoset.metrics.logme import LogMEMetric


def test_logme_class_perfect_features(dummy_features_perfect, dummy_labels):
    """Test the LogME metric class for perfect features"""
    metric = LogMEMetric()
    assert metric.metric_name == "LogME"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(dummy_features_perfect, dummy_labels) > 0


def test_logme_class_random_features(dummy_features_random, dummy_labels):
    """Test the LogME metric class for random features"""
    metric = LogMEMetric()
    assert metric.metric_name == "LogME"
    assert metric.inference_type == "features"
    assert metric.dataset_dependent is True
    assert metric.fit_metric(dummy_features_random, dummy_labels) < 0
