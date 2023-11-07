"""
    Test functions for the config classes (src/locomoset/run/classes)
"""

import os

import yaml

from locomoset.metrics.classes import TopLevelMetricConfig
from locomoset.models.classes import TopLevelFineTuningConfig


def test_bask_script_creation_metrics(dummy_top_level_config, test_n_samples):
    # generate test bask script
    config = TopLevelMetricConfig.from_dict(dummy_top_level_config)
    config.config_type = "metrics"
    config.generate_sub_configs()
    config.save_sub_configs()
    config.create_bask_job_script(test_n_samples)
    config_path = f"{config.config_dir}/{config.config_gen_dtime}"
    file_name = f"{config.config_type}_jobscript_{config.config_gen_dtime}"

    # turn .sh into .yaml for testing purposes
    os.rename(f"{config_path}/{file_name}.sh", f"{config_path}/{file_name}.yaml")
    with open(f"{config_path}/{file_name}.yaml", "r") as f:
        bask_script = yaml.safe_load(f)

    assert bask_script["job_name"] == str(config.bask[config.config_type]["job_name"])
    assert bask_script["walltime"] == str(config.bask[config.config_type]["walltime"])
    assert bask_script["node_number"] == str(
        config.bask[config.config_type]["node_number"]
    )
    assert bask_script["gpu_number"] == str(
        config.bask[config.config_type]["gpu_number"]
    )
    assert bask_script["cpu_per_gpu"] == str(
        config.bask[config.config_type]["cpu_per_gpu"]
    )
    assert bask_script["config_path"] == str(
        f"{config.config_dir}/{config.config_gen_dtime}"
    )
    assert bask_script["array_number"] == str(test_n_samples)
    assert bask_script["config_type"] == str(config.config_type)


def test_bask_script_creation_training(dummy_top_level_config, test_n_samples):
    # generate test bask script
    config = TopLevelFineTuningConfig.from_dict(dummy_top_level_config)
    config.config_type = "train"
    config.generate_sub_configs()
    config.save_sub_configs()
    config.create_bask_job_script(test_n_samples)
    config_path = f"{config.config_dir}/{config.config_gen_dtime}"
    file_name = f"{config.config_type}_jobscript_{config.config_gen_dtime}"

    # turn .sh into .yaml for testing purposes
    os.rename(f"{config_path}/{file_name}.sh", f"{config_path}/{file_name}.yaml")
    with open(f"{config_path}/{file_name}.yaml", "r") as f:
        bask_script = yaml.safe_load(f)

    assert bask_script["job_name"] == str(config.bask[config.config_type]["job_name"])
    assert bask_script["walltime"] == str(config.bask[config.config_type]["walltime"])
    assert bask_script["node_number"] == str(
        config.bask[config.config_type]["node_number"]
    )
    assert bask_script["gpu_number"] == str(
        config.bask[config.config_type]["gpu_number"]
    )
    assert bask_script["cpu_per_gpu"] == str(
        config.bask[config.config_type]["cpu_per_gpu"]
    )
    assert bask_script["config_path"] == str(
        f"{config.config_dir}/{config.config_gen_dtime}"
    )
    assert bask_script["array_number"] == str(test_n_samples)
    assert bask_script["config_type"] == str(config.config_type)
