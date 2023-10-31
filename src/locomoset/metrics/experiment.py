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

from datasets import load_dataset
from numpy.typing import ArrayLike
from transformers.modeling_utils import PreTrainedModel

from locomoset.metrics.classes import Metric
from locomoset.metrics.library import METRICS
from locomoset.models.features import get_features
from locomoset.models.load import get_model_without_head, get_processor


class ModelMetricsExperiment:
    """Model experiment class. Runs method metric.fit_metric() for each metric stated,
    which takes arguments: (model_input, dataset_input).
    """

    def __init__(self, config: dict) -> None:
        """Initialise model experiment class.

        Args:
            config: Dictionary containing the following:
                - model_name: name of model to be computed (str)
                - dataset_name: name of dataset to be scored by (str)
                - dataset_split: dataset split (str)
                - n_samples: number of samples for a metric experiment (int)
                - random_state: random seed for variation of experiments (int)
                - metrics: list of metrics to score (list(str))
                - (Optional) metric_kwargs: dictionary of entries
                    {metric_name: **metric_kwargs} containing parameters for each metric
                - (Optional) save_dir: Directory to save results, "results" if not set.
        """
        # Parse model/seed config
        self.model_name = config["model_name"]
        self.random_state = config["random_state"]

        # Initialise metrics
        metric_kwargs_dict = config.get("metric_kwargs", {})
        self.metrics = {
            metric: METRICS[metric](
                random_state=self.random_state, **metric_kwargs_dict.get(metric, {})
            )
            for metric in config["metrics"]
        }
        self.inference_types = list(
            set(metric.inference_type for metric in self.metrics.values())
        )

        # Load/generate dataset
        print("Generating data sample...")
        self.dataset_name = config["dataset_name"]
        self.dataset = load_dataset(self.dataset_name, split=config["dataset_split"])
        self.n_samples = config["n_samples"]
        if self.n_samples < self.dataset.num_rows:
            self.dataset = self.dataset.train_test_split(
                train_size=self.n_samples, shuffle=True, seed=self.random_state
            )["train"]
        self.labels = self.dataset["label"]

        # Initialise results dict
        self.results = config
        self.results["inference_times"] = {}
        self.results["metric_scores"] = {}
        self.save_dir = config.get("save_dir", "results")
        os.makedirs(self.save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.save_path = f"{self.save_dir}/results_{date_str}.json"

    def features_inference(self) -> ArrayLike:
        """Perform inference for features based methods.

        Returns:
            Features generated by the model with its classification head removed on the
                test dataset.
        """
        model_fn = get_model_without_head(self.model_name)
        processor = get_processor(self.model_name)
        return get_features(self.dataset, processor, model_fn)

    def model_inference(self) -> PreTrainedModel:
        """Perform inference for model based methods (just load the model).

        Returns:
            Model with its classification head removed.
        """
        return get_model_without_head(self.model_name)

    def perform_inference(
        self, inference_type: str | None
    ) -> Tuple[ArrayLike, float] | Tuple[None, float]:
        """Perform inference to retrieve data necessary for metric score computation.

        Args:
            inference_type: type of inference required, one of the following:
                - 'features': The model_input passed to the metric is features generated
                    by the model on the test dataset
                - 'model': The model_input passed to the metric is the model itself
                    (with its classification head removed)
                - None: model_input is set to None.

        Returns:
            Generated inference data (or None if nothing to generate) and computation
            time.
        """
        inference_start = time()
        if inference_type == "features":
            return self.features_inference(), time() - inference_start
        elif inference_type == "model":
            return self.model_inference(), time() - inference_start
        elif inference_type is not None:
            raise NotImplementedError(
                f"Not implemented inference for type '{inference_type}'"
            )
        return None, time() - inference_start

    def compute_metric_score(
        self,
        metric: Metric,
        model_input: ArrayLike | PreTrainedModel | None,
        dataset_input: ArrayLike | None,
    ) -> Tuple[float, float] | Tuple[int, float]:
        """Compute the metric score for a given metric. Not every metric requires
        either, or both, of the model_input or dataset_input but these are always input
        for a consistent pipeline (even if the input is None) and dealt with within the
        the model classes.

        Args:
            metric: metric object
            model_input: model input, type depends on metric inference type
            dataset_input: dataset input, from dataset (labels)

        Returns:
            metric score, computational time
        """
        metric_start = time()
        return (
            metric.fit_metric(model_input=model_input, dataset_input=dataset_input),
            time() - metric_start,
        )

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
            model_input, inference_time = self.perform_inference(inference_type)
            self.results["inference_times"][inference_type] = inference_time

            test_metrics = [
                metric_name
                for metric_name, metric_obj in self.metrics.items()
                if metric_obj.inference_type == inference_type
            ]
            for metric in test_metrics:
                print(f"Computing metric score for {metric}")
                self.results["metric_scores"][metric] = {}
                score, metric_time = self.compute_metric_score(
                    self.metrics[metric],
                    model_input,
                    self.labels,
                )
                self.results["metric_scores"][metric]["score"] = score
                self.results["metric_scores"][metric]["time"] = metric_time

    def save_results(self) -> None:
        """Save the experiment results to self.save_path."""
        with open(self.save_path, "w") as f:
            json.dump(self.results, f, default=float)
        print(f"Results saved to {self.save_path}")
