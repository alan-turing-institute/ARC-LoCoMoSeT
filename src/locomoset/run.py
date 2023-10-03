"""
    Entry point for running experiments per model for each metric stated in config.
"""

import argparse
from copy import copy

import yaml

from locomoset.models.model_experiment_classes import ModelExperiment


def model_experiment_multiplicity(config: dict) -> list(dict):
    """For a config with multiple models submitted for experiment, create a new list of
    of config dicts with a single model for each config.

    Args:
        config: config dictionary containing multiplicity of models.
    """
    model_configs = []
    for model in config["models"]:
        new_config = copy(config)
        new_config["model_name"] = model
        del new_config["models"]
        model_configs.append(new_config)
    return model_configs


def random_state_multiplicity(config: dict) -> list(dict):
    """For a config with multiple random states submitted for experiment, create a new
    list of config dicts with a single random state for each config.

    Args:
        config: config dictionary containing multiplicity of random stages.
    """
    model_configs = []
    for rstate in config["random_state"]:
        new_config = copy(config)
        new_config["random_state"] = rstate
        model_configs.append(new_config)
    return model_configs


def run(config: dict):
    """Run comparative metric experiment for a given pair (model, dataset) for stated
    metrics. Results saved to file path of form results/results_YYYYMMDD-HHMMSS.json by
    default.

    Args:
        config: Loaded configuration dictionary including the following keys:
            - models: a list of HuggingFace model names to experiment with.
            - dataset_name: Name of HuggingFace dataset to use.
            - dataset_split: Dataset split to use.
            - metrics: Which metrics to experiment on.
            - n_samples: List of how many samples (images) to compute the metric with.
            - random_state: List of random seeds to compute the metric with (used for
                subsetting the data and dimensionality reduction).
            - feat_red_dim: List of feature dimensions to compute the metric with (PCA
                is used for dimensionality reduction).
            - (Optional) save_dir: Directory to save results, "results" if not set.
    """

    model_configs = random_state_multiplicity(config)
    model_configs = sum(
        [model_experiment_multiplicity(model_config) for model_config in model_configs],
        [],
    )

    for model_config in model_configs:
        model_experiment = ModelExperiment(model_config)
        model_experiment.run_experiment()
        model_experiment.save_results()


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics scans with various parameter values"
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    run(config)


if __name__ == "__main__":
    main()
