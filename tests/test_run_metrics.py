"""
Tests for the run metrics script
"""

from locomoset.metrics.run import (
    compute_metric,
    nest_var_in_list,
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
        "par_3": 7,
    }

    par_dicts = parameter_sweep_dicts(test_dict)
    par_dict_pars = [list(d.values()) for d in par_dicts]

    possible_par_sets = [
        ["name_1", "name_21", 1, 4, 7],
        ["name_1", "name_21", 1, 5, 7],
        ["name_1", "name_21", 1, 6, 7],
        ["name_1", "name_21", 2, 4, 7],
        ["name_1", "name_21", 2, 5, 7],
        ["name_1", "name_21", 2, 6, 7],
        ["name_1", "name_21", 3, 4, 7],
        ["name_1", "name_21", 3, 5, 7],
        ["name_1", "name_21", 3, 6, 7],
        ["name_1", "name_22", 1, 4, 7],
        ["name_1", "name_22", 1, 5, 7],
        ["name_1", "name_22", 1, 6, 7],
        ["name_1", "name_22", 2, 4, 7],
        ["name_1", "name_22", 2, 5, 7],
        ["name_1", "name_22", 2, 6, 7],
        ["name_1", "name_22", 3, 4, 7],
        ["name_1", "name_22", 3, 5, 7],
        ["name_1", "name_22", 3, 6, 7],
    ]

    assert par_dict_pars == possible_par_sets


def test_nest_var_in_list():
    """
    Test the function that nests a string in a list to avoid iterable confounding.
    """
    assert type(nest_var_in_list("test")) == list
    assert type(nest_var_in_list(["test"])) == list
    assert type(nest_var_in_list(4)) == list
    assert type(nest_var_in_list([4])) == list


def test_compute_metric(dummy_config):
    """
    Test the metric computation function runs.
    """
    result = compute_metric(dummy_config)
    assert isinstance(result["result"]["score"], float)
    assert isinstance(result["result"]["time"], float)
