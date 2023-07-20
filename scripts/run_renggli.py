import argparse
import json
import os
from datetime import datetime
from time import time

import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from locomoset.datasets.preprocess import preprocess
from locomoset.metrics.renggli import renggli_score
from locomoset.models.features import get_features
from locomoset.models.load import get_model_and_processor


def run_renggli(config: dict):
    """Compute Renggli et al metric for a single dataset and model, but for a number
    of different data subset sizes and random seeds.

    Results saved to file path of form results/results_20230711-143025.json by default.

    Args:
        config: Loaded configuration dictionary including the following keys:
            - model_name: Name of HuggingFace model to use
            - dataset_name: Name of HuggingFace dataset to use
            - dataset_split: Dataset split to use
            - n_samples: List of how many samples (images) to compute the metric with
            - random_states: List of random seeds to compute the metric with (used for
                subsetting the data)
            - (Optional) save_dir: Directory to save results, "results" if not set
    """
    save_dir = config.get("save_dir", "results")
    os.makedirs(save_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"{save_dir}/results_{date_str}.json"

    results = config
    results["results"] = []
    results["time"] = {}
    results["metric"] = "renggli"

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

    print("Starting Renggli computation loop...")
    for samples in tqdm(config["n_samples"]):
        for state in tqdm(config["random_states"]):
            if samples < len(labels):
                run_features, _, run_labels, _ = train_test_split(
                    features, labels, train_size=samples, random_state=state
                )
            else:
                run_features = features
                run_labels = labels
            metric_start = time()
            result = renggli_score(run_features, run_labels, random_state=state)
            results["results"].append(
                {
                    "score": result,
                    "random_state": state,
                    "n_samples": samples,
                    "metric_time": time() - metric_start,
                }
            )
            with open(save_path, "w") as f:
                json.dump(results, f, default=float)

    print(f"DONE! Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Renggli metrics")
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    run_renggli(config)
