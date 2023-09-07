"""
Test functions for the task_agonstic.py metrics.
"""

from locomoset.metrics.task_agnostic import num_params_metric


def test_num_params_metric(dummy_model_classifier):
    """
    Test that the number of parameters metric returns the correct value.
    """
    assert num_params_metric(dummy_model_classifier) == 1698
