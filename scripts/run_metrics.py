"""
    Main script for running comparative experiment given a config file.
"""

import argparse
import json
import os
from datetime import datetime
from itertools import product
from time import time

import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from locomoset.datasets.preprocess import preprocess
from locomoset.metrics.parc import parc
from locomoset.metrics.renggli import renggli_score
from locomoset.models.features import get_features
from locomoset.models.load import get_model_and_processor

metric_funcs = {"parc": parc, "renggli": renggli_score}


def run_metric(config: dict):
    """Run comparative metric experiment for given pair (dataset, model). Results saved
    to file path of form results/results_YYYYMMDD-HHMMSS.json by default.

    Args:
        config: Loaded configuration dictionary including the following keys:
            - model_name: Name of HuggingFace model to use.
            - dataset_name: Name of HuggingFace dataset to use.
            - dataset_split: Dataset split to use.
            - metric: Which metric to experiment on.
            - n_samples: List of how many samples (images) to compute the metric with.
            - random_states: List of random seeds to compute the metric with (used for
                subsetting the data and dimensionality reduction).
            - feat_red_dims: List of feature dimensions to compute the metric with (PCA
                is used for dimensionality reduction).
            - (Optional) save_dir: Directory to save results, "results" if not set.
    """
    save_dir = config.get("save_dir", "results")
    os.makedirs(save_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"{save_dir}/results_{date_str}.json"

    # Removes feat dim reduction for now (wrap in if statement for future WPs)
    del config["feat_red_dims"]

    # creates all experiment variants
    config_keys, config_vals = zip(*config.items())
    config_variants = [dict(config_keys, v) for v in product(*config_vals)]

    results = config
    results["results"] = []
    results["time"] = {}

    model_head, processor = get_model_and_processor(config["model_name"], num_labels=0)
    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])

    print("Preprocessing...")
    process_start = time()
    dataset = preprocess(dataset, processor)
    results["time"]["preprocess"] = time() - process_start

    print("Extracting features...")
    features_start = time()
    features = get_features(dataset, model_head, batched=True, batch_size=4)
    results["time"]["features"] = time() - features_start

    labels = dataset["label"]

    for config_var in tqdm(config_variants):
        if config_var["num_samples"] < len(labels):
            run_features, _, run_labels, _ = train_test_split(
                features,
                labels,
                train_size=config_var["num_samples"],
                random_state=config_var["random_state"],
            )
        else:
            run_features = features
            run_labels = labels
        metric_start = time()
        result = metric_funcs[config_var["metric"]](
            run_features, run_labels, random_state=config_var["random_state"]
        )
        results["result"].append(
            {
                "score": result,
                "random_state": config_var["random_state"],
                "n_samples": config_var["n_samples"],
                "metric_time": time() - metric_start,
            }
        )
        with open(save_path, "w") as f:
            json.dump(results, f, default=float)

    print(f"DONE! Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PARC metrics")
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    run_metric(config)
