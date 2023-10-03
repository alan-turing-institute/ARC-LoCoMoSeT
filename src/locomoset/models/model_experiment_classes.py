"""
    Classes for the model experiments. This is the overarching class that takes as an
    input the name of the model, a dataset, a list of metrics to score the model by, and
    any parameters required by said metrics.

    Model inference here is done by pipline.
"""

from time import time

import numpy as np
from datasets import load_dataset
from torch import Tensor

from locomoset.metrics.library import METRIC_CLASSES
from locomoset.models.features import get_features
from locomoset.models.load import get_model_and_processor

# from datasets import Dataset


class ModelExperiment:

    """Model experiment base class."""

    def __init__(self, **config) -> None:
        """Initialise model experiment class for given config with following args:

        Args:
            model_name: name of model to be computed (str)
            dataset_name: name of dataset to be scored by (str)
            dataset_split: dataset split (str)
            n_samples: number of samples for a metric experiment (int)
            random_state: random seed for variation of experiments (int)
            metrics: list of metrics to score (model, dataset) pair by (list(str))
            param_sweep: whether to do a parameter sweep over the metric parameters
                        (bool). Defaults to False.
            metric_kwargs: dictionary of entries {metric_name: **metric_kwargs} containg
                            parameters for each metric.
        """
        self.model_name = config["model_name"]
        self.dataset_name = config["dataset_name"]
        self.n_samples = config["n_samples"]
        self.random_state = config["random_state"]
        print("Generating data sample...")
        self.dataset = load_dataset(self.dataset_name, split=config["dataset_split"])
        if self.n_samples < self.dataset.num_rows:
            self.dataset = self.dataset.train_test_split(
                train_size=self.n_samples, shuffle=True, seed=self.random_state
            )["train"]
        self.labels = self.dataset["labels"]
        self.metrics = {
            metric: {
                "metric_fn": METRIC_CLASSES[metric](config["metric_kwargs"][metric])
            }
            for metric in config["metrics"]
        }
        for metric in self.metrics.keys():
            self.metrics[metric]["inference_type"] = self.metrics[metric][
                "metric_fn"
            ].get_inference_type()
        self.inference_types = np.unique(
            [self.metrics[metric]["inference_type"] for metric in config["metrics"]]
        )
        self.results = {"time": {}}

    def get_model_name(self) -> str:
        """Get the model name

        Returns:
            model name
        """
        return self.model_name

    def get_dataset_name(self) -> str:
        """Get dataset name

        Returns:
            dataset name
        """
        return self.dataset_name

    def perform_inference(self, inference_type: str) -> (Tensor, float) | (None, float):
        """Perform inference to retrieve data necessary for metric score computation.

        Args:
            inference_type: type of inference required, one of the following:
                - features
                - distribution
                - predicted labels
                - predicted labels with source dataset
                - None

        Returns:
            _description_
        """
        inference_start = time()
        if inference_type is not None:
            if inference_type == "features":
                model_fn, processor = get_model_and_processor(
                    self.model_name, num_labels=0
                )
                return (
                    get_features(self.dataset, processor, model_fn),
                    time() - inference_start,
                )
            # elif inference_type == "distribution":
            #     model_fn, processor = get_model_and_processor(self.model_name)
        return None, time() - inference_start

    def compute_metric_score(self, metric, model_input, dataset_input) -> float:
        """Compute the metric score for a given metric. Not every metric requires
        either, or both, of the model_input or dataset_input but these are always input
        for a consistent pipeline (even if the input is None) and dealt with within the
        the model classes.

        Args:
            metric: metric object
            model_input: model input, from inference
            dataset_input: dataset input, from dataset (labels)

        Returns:
            metric score, computational time
        """
        metric_start = time()
        return metric.fit_metric(model_input, dataset_input), time() - metric_start

    def run_experiment(self) -> dict:
        """Run the experiment pipeline

        Returns:
            dictionary of results
        """
        print(f"Scoring metrics: {self.metrics}")
        print(f"with inference types requires: {self.inference_types}")
        for inference_type in self.inference_types:
            print(f"Computing metrics with inference type {inference_type}")
            print("Running inference")
            model_input, self.results["time"][inference_type] = self.perform_inference(
                inference_type
            )
            test_metrics = [
                metric
                for metric in self.metrics.keys()
                if self.metrics[metric]["inference_type"] == inference_type
            ]
            for metric in test_metrics:
                print(f"Computing metric score for {metric}")
                (
                    self.results[metric]["score"],
                    self.results[metric]["time"],
                ) = self.compute_metric_score(
                    self.metrics[metric], model_input, self.labels
                )

        return self.results
