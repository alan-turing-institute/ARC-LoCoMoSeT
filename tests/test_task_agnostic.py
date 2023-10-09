"""
Test functions for the task_agonstic.py metrics.
"""

from locomoset.metrics.task_agnostic import NumParsMetric


def test_n_pars_class(dummy_model_head):
    """Test the task agonistic metric class"""
    metric = NumParsMetric()
    assert metric.metric_name == "n_pars"
    assert metric.inference_type == "model"
    assert metric.fit_metric(dummy_model_head) == 1686
