from locomoset import datasets, metrics, models
from locomoset.metrics.parc import parc
from locomoset.metrics.renggli import renggli_score

__all__ = ("models", "metrics", "datasets")


def export_metric_func(metric_name):
    metric_functions = {
        "parc": parc,
        "renggli": renggli_score,
    }
    return metric_functions[metric_name]
