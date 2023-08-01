"""
    Tests for the run metrics script
"""

import numpy as np
import pytest
from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder

from locomoset.metrics.run import (
    compute_metric,
    nest_string_in_list,
    parameter_sweep_dicts,
)


def test_parameter_sweep():
    """
    Test the parameter sweep config dicts generator function
    """
    test_dict = {
        "name_1": "name_1",
        "name_2": ["name_21", "name_22"],
        "par_1": [1, 2, 3],
        "par_2": [4, 5, 6],
    }

    par_dicts = parameter_sweep_dicts(test_dict)
    par_dict_pars = [list(d.values()) for d in par_dicts]

    possible_par_sets = [
        ["name_1", "name_21", 1, 4],
        ["name_1", "name_21", 1, 5],
        ["name_1", "name_21", 1, 6],
        ["name_1", "name_21", 2, 4],
        ["name_1", "name_21", 2, 5],
        ["name_1", "name_21", 2, 6],
        ["name_1", "name_21", 3, 4],
        ["name_1", "name_21", 3, 5],
        ["name_1", "name_21", 3, 6],
        ["name_1", "name_22", 1, 4],
        ["name_1", "name_22", 1, 5],
        ["name_1", "name_22", 1, 6],
        ["name_1", "name_22", 2, 4],
        ["name_1", "name_22", 2, 5],
        ["name_1", "name_22", 2, 6],
        ["name_1", "name_22", 3, 4],
        ["name_1", "name_22", 3, 5],
        ["name_1", "name_22", 3, 6],
    ]

    assert par_dict_pars == possible_par_sets


def test_nest_string_in_list():
    """
    Test the function that nests a string in a list to avoid iterable confounding.
    """
    assert type(nest_string_in_list("test")) == list
    assert type(nest_string_in_list(4)) == int


def test_compute_metric():
    """
    Test the metric computation function.
    """
    n_classes = 3
    n_features = 5
    n_samples = 1000
    rng = np.random.default_rng(42)
    labels = rng.integers(0, n_classes, n_samples)
    perf_features = OneHotEncoder(sparse_output=False).fit_transform(
        labels.reshape((n_samples, 1))
    )
    rand_features = rng.normal(size=(n_samples, n_features))

    parc_test_config = {
        "num_samples": 500,
        "random_state": 42,
        "metric": "parc",
        "parc_test": True,
    }
    renggli_test_config = {"num_samples": 500, "random_state": 42, "metric": "renggli"}

    assert compute_metric(parc_test_config, perf_features, labels)[0] == pytest.approx(
        100
    )
    assert compute_metric(parc_test_config, rand_features, labels)[0] == pytest.approx(
        0.0, abs=0.3
    )
    assert compute_metric(renggli_test_config, perf_features, labels)[0] == 1
    assert compute_metric(renggli_test_config, rand_features, labels)[
        0
    ] == pytest.approx(1 / n_classes, rel=0.3)


def test_load_dataset(dummy_dataset_name, dummy_split):
    load_dataset(dummy_dataset_name, split=dummy_split)
