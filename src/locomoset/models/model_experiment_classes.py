"""
    Classes for the model experiments. This is the overarching class that takes as an
    input the name of the model, a dataset, a list of metrics to score the model by, and
    any parameters required by said metrics.

    Model inference here is done by pipline.
"""

import json
import os
from datetime import datetime
from time import time
from typing import Tuple

import numpy as np
from datasets import load_dataset
from numpy.typing import ArrayLike
from transformers.modeling_utils import PreTrainedModel

from locomoset.metrics.classes import Metric
from locomoset.metrics.library import METRICS
from locomoset.models.features import get_features
from locomoset.models.load import get_model_and_processor


class ModelExperiment:

    """Model experiment base class. Runs method metric.fit_metric() for each metric
    stated, which takes arguments:

            (model_fn, model_input, dataset_input, **model_kwargs)

    """

    def __init__(self, config: dict) -> None:
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
            (Optional) save_dir: Directory to save results, "results" if not set.
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
        self.labels = self.dataset["label"]
        metric_kwargs_dict = config.get("metric_kwargs", {})
        self.metrics = {
            metric: {"metric_fn": METRICS[metric](**metric_kwargs_dict.get(metric, {}))}
            for metric in config["metrics"]
        }
        for metric in self.metrics.keys():
            self.metrics[metric]["inference_type"] = str(
                self.metrics[metric]["metric_fn"].inference_type
            )
        self.inference_types = np.unique(
            [
                str(self.metrics[metric]["inference_type"])
                for metric in config["metrics"]
            ]
        )
        self.results = config
        self.results["inference_times"] = {}
        self.results["metric_scores"] = {}
        self.save_dir = config.get("save_dir", "results")
        os.makedirs(self.save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.save_path = f"{self.save_dir}/results_{date_str}.json"

    def features_inference(self) -> ArrayLike:
        """Perform inference for features based methods"""
        model_fn, processor = get_model_and_processor(self.model_name, num_labels=0)
        return get_features(self.dataset, processor, model_fn)

    def perform_inference(
        self, inference_type: str
    ) -> Tuple[ArrayLike, float] | Tuple[None, float]:
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
        if inference_type == "features":
            return self.features_inference(), time() - inference_start
        return None, time() - inference_start

    def compute_metric_score(
        self,
        metric: Metric,
        model_fn: PreTrainedModel,
        model_input: ArrayLike,
        dataset_input: ArrayLike,
    ) -> Tuple[float, float] | Tuple[int, float]:
        """Compute the metric score for a given metric. Not every metric requires
        either, or both, of the model_input or dataset_input but these are always input
        for a consistent pipeline (even if the input is None) and dealt with within the
        the model classes.

        Args:
            metric: metric object
            model: model
            model_input: model input, from inference
            dataset_input: dataset input, from dataset (labels)

        Returns:
            metric score, computational time
        """
        metric_start = time()
        if metric.dataset_dependent:
            return (
                metric.fit_metric(model_input, dataset_input),
                time() - metric_start,
            )
        else:
            return metric.fit_metric(model_fn), time() - metric_start

    def run_experiment(self) -> dict:
        """Run the experiment pipeline

        Returns:
            dictionary of results
        """
        print(f"Scoring metrics: {self.metrics}")
        print(f"with inference types requires: {self.inference_types}")
        model_fn, _ = get_model_and_processor(self.model_name, num_labels=0)
        for inference_type in self.inference_types:
            print(f"Computing metrics with inference type {inference_type}")
            print("Running inference")
            (
                model_input,
                self.results["inference_times"][inference_type],
            ) = self.perform_inference(inference_type)
            test_metrics = [
                metric
                for metric in self.metrics.keys()
                if self.metrics[metric]["inference_type"] == inference_type
            ]
            for metric in test_metrics:
                print(f"Computing metric score for {metric}")
                self.results["metric_scores"][metric] = {}
                (
                    self.results["metric_scores"][metric]["score"],
                    self.results["metric_scores"][metric]["time"],
                ) = self.compute_metric_score(
                    self.metrics[metric]["metric_fn"],
                    model_fn,
                    model_input,
                    self.labels,
                )

    def save_results(self):
        """Save the experimental results."""
        with open(self.save_path, "w") as f:
            json.dump(self.results, f, default=float)
        print(f"Results saved to {self.save_path}")
