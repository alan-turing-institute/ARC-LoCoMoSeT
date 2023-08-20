"""
Tests for the run metrics script
"""
import glob
import json
import tempfile

from locomoset.metrics.run import (
    compute_metric,
    nest_var_in_list,
    parameter_sweep_dicts,
    run,
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
    assert (
        isinstance(result["result"]["score"], float) and result["result"]["score"] != 0
    )
    assert isinstance(result["result"]["time"], float) and result["result"]["time"] != 0


def test_run(dummy_config):
    """
    Test an end to end run of computing metrics with multiple configs.
    """
    # adjust dummy config to give multiple results. Setting the 2nd value to 5000 also
    # verifies that results will still be computed if the number of samples requested is
    # greater than the number of images in the dataset (the whole dataset is used in
    # that case).
    test_samples = [25, 5000]
    dummy_config["n_samples"] = test_samples

    with tempfile.TemporaryDirectory() as temp_dir:
        # set results to be saved to the temporary directory
        dummy_config["save_dir"] = temp_dir

        run(dummy_config)

        # check the correct number of results files were saved
        results_files = glob.glob(f"{temp_dir}/*.json")
        assert len(results_files) == 2

        actual_samples = set()
        for path in results_files:
            with open(path) as rf:
                results = json.load(rf)
                # check a non-zero metric score was saved
                assert results["result"]["score"] != 0
                actual_samples.add(results["n_samples"])

        # verify there was a result for each n_samples value
        assert actual_samples == set(test_samples)
