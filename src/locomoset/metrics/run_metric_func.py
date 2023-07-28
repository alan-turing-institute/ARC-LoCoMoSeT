"""
    Functions for running a metric experiment.
"""

from itertools import product
from time import time

from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split

from locomoset import export_metric_func


def nest_string_in_list(string):
    """Nest an element of a list in a list if it is a string, to convert it to an
    element of an iterable rather than an iterable itself

    Args:
        string: string to be converted
    """
    return [string] if isinstance(string, str) else string


def parameter_sweep_dicts(config: dict) -> list[dict]:
    """Generate all sub config dicts for parameter sweep of experiment parameters.

    Args:
        config: Config file for particular experiment

    Returns:
        list[dict]: List of all config dictionaries containing unique combination of
                    parameters.
    """
    config_keys, config_vals = zip(*config.items())
    return [
        dict(zip(config_keys, v))
        for v in product(*list(map(nest_string_in_list, config_vals)))
    ]


def compute_metric(config: dict, features: ArrayLike, labels: ArrayLike):
    """Compute the results of a metric experiment

    Args:
        config: config for specific experiment instance
        features: precomputed features from model
        labels: related labels for the features
    """
    if config["num_samples"] < len(labels):
        run_features, _, run_labels, _ = train_test_split(
            features,
            labels,
            train_size=config["num_samples"],
            random_state=config["random_state"],
        )
    else:
        run_features = features
        run_labels = labels
    metric_start = time()
    metric_funcion = export_metric_func(config["metric"])
    if "parc_test" in list(config.keys()):
        result = metric_funcion(
            run_features,
            run_labels,
            random_state=config["random_state"],
            feat_red_dim=None,
        )
    else:
        result = metric_funcion(
            run_features, run_labels, random_state=config["random_state"]
        )
    return result, time() - metric_start
