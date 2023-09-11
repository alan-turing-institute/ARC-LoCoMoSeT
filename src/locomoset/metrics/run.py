"""
Entry-point for computing metric scores for various configurations.
"""

import argparse
import json
import os
from datetime import datetime
from itertools import product
from time import time
from typing import Any, Iterable

import yaml
from datasets import load_dataset
from tqdm import tqdm

from locomoset.metrics.library import METRIC_FUNCTIONS
from locomoset.models.features import get_features
from locomoset.models.load import get_model_and_processor


def nest_var_in_list(var: Any) -> list[Any]:
    """Nest a variable in a list if it is not an iterable, or is a string.

    Args:
        var: variable to be converted

    Returns:
        list containing the variable
    """
    return [var] if not isinstance(var, Iterable) or isinstance(var, str) else var


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
        for v in product(*list(map(nest_var_in_list, config_vals)))
    ]


def compute_metric(config: dict) -> dict:
    """Compute the results of a metric experiment

    Args:
        config: config for specific experiment instance

    Returns:
        Dict including original config and an additional key "results" containing the
            metric score and time taken to compute it.
    """
    results = config
    results["time"] = {}

    model_head, processor = get_model_and_processor(config["model_name"], num_labels=0)
    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])

    print("Generating data sample...")
    if config["n_samples"] < dataset.num_rows:
        dataset = dataset.train_test_split(
            train_size=config["n_samples"], shuffle=True, seed=config["random_state"]
        )["train"]
    labels = dataset["label"]

    print("Extracting features...")
    features_start = time()
    features = get_features(dataset, processor, model_head)
    results["time"]["features"] = time() - features_start

    print("Computing metric...")
    metric_start = time()
    metric_function = METRIC_FUNCTIONS[config["metric"]]
    score = metric_function(features, labels, random_state=config["random_state"])
    results["result"] = {
        "score": score,
        "time": time() - metric_start,
    }

    return results


def run(config: dict):
    """Run comparative metric experiment for given pair (dataset, model). Results saved
    to file path of form results/results_YYYYMMDD-HHMMSS.json by default.

    Args:
        config: Loaded configuration dictionary including the following keys:
            - model_name: Name of HuggingFace model to use.
            - dataset_name: Name of HuggingFace dataset to use.
            - dataset_split: Dataset split to use.
            - metric: Which metric to experiment on.
            - n_samples: List of how many samples (images) to compute the metric with.
            - random_state: List of random seeds to compute the metric with (used for
                subsetting the data and dimensionality reduction).
            - feat_red_dim: List of feature dimensions to compute the metric with (PCA
                is used for dimensionality reduction).
            - (Optional) save_dir: Directory to save results, "results" if not set.
    """
    save_dir = config.get("save_dir", "results")
    os.makedirs(save_dir, exist_ok=True)

    # creates all experiment variants
    config_variants = parameter_sweep_dicts(config)

    for config_var in tqdm(config_variants):
        print(f"Starting computation for {config_var}...")
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        save_path = f"{save_dir}/results_{date_str}.json"
        results = compute_metric(config_var)
        with open(save_path, "w") as f:
            json.dump(results, f, default=float)
        print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics scans with various parameter values"
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    run(config)
