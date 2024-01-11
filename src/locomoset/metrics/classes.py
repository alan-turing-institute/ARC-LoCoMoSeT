"""
Base metric class for unifying the method.
"""
from abc import ABC, abstractmethod

import numpy as np

from locomoset.config.config_classes import Config, TopLevelConfig


class Metric(ABC):
    """Base class for metrics.

    Attributes:
        metric_name: Name of the metric
        inference_type: Type of inference - either 'features', 'model' or None:
            - 'features': The model_input passed to the metric is features generated
                    by the model on the test dataset
            - 'model': The model_input passed to the metric is the model itself
                (with its classification head removed)
            - None: model_input is set to None.
        dataset_dependent: Whether the metric requires information about the dataset or
            only uses the model.
        random_state: Random seed to set prior to metric computation.
    """

    def __init__(
        self,
        metric_name: str,
        inference_type: str | None,
        dataset_dependent: bool,
        random_state: int | None = None,
    ) -> None:
        self.metric_name = metric_name
        self.inference_type = inference_type
        self.dataset_dependent = dataset_dependent
        self.random_state = random_state

    def set_random_state(self) -> None:
        """Set the random state (should be called in fit_metric before calling
        metric_function)
        """
        np.random.seed(self.random_state)

    @abstractmethod
    def metric_function(self, *args, **kwargs) -> float | int:
        """Base metric function, reimplement for each metric class. Can include
        different number arguments depending on the metric subclass."""

    @abstractmethod
    def fit_metric(self, model_input, dataset_input) -> float | int:
        """Base fit metric function, reimplement for each subclass (metric category).
        Must always take two inputs, and takes care of setting random seeds and calling
        metric_function correctly for each subclass."""


class TaskAgnosticMetric(Metric):
    """Base class for task agnostic metrics, for which the metric function input is
    solely the model function."""

    def __init__(
        self,
        metric_name: str,
        inference_type: str | None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            metric_name,
            inference_type=inference_type,
            dataset_dependent=False,
            random_state=random_state,
        )

    @abstractmethod
    def metric_function(self, model_input) -> float | int:
        """TaskAgnosticMetric classes should implement a metric_function that takes
        only a a model, or model-derived quantity/function, as input."""

    def fit_metric(self, model_input, dataset_input=None) -> float | int:
        """Compute the metric value. The dataset_input is not used for
        TaskAgnosticMetric instances but included for compatibility with the parent
        Metric class."""
        self.set_random_state()
        return self.metric_function(model_input)


class TaskSpecificMetric(Metric):
    """Base class for task specific metric, for which the input is of shape:
    (model_input, dataset_input, **kwargs)
    """

    def __init__(
        self, metric_name: str, inference_type: str, random_state: int | None = None
    ) -> None:
        super().__init__(
            metric_name=metric_name,
            inference_type=inference_type,
            dataset_dependent=True,
            random_state=random_state,
        )

    @abstractmethod
    def metric_function(self, model_input, dataset_input) -> float | int:
        """TaskSpecificMetric classes should implement a metric_function that takes both
        a model-derived input (e.g. features generated on new samples) and a dataset
        input (e.g. ground truth labels for each sample)."""

    def fit_metric(self, model_input, dataset_input) -> float | int:
        self.set_random_state()
        return self.metric_function(model_input, dataset_input)


class MetricConfig(Config):

    """Metric configuration class.

    Attributes:
        model_name: Name of the HuggingFace model to perform metric experiment on.
        dataset_name: Name of the HuggingFace dataset to use for metric experiment.
        metrics: Which metrics to perform the experiments on.
        metric_kwargs: dictionary of entries
            {metric_name: **metric_kwargs} containing parameters for each metric
        save_dir: Where to save a local copy of the results.
        dataset_args: Dict defining the splits and columns of the dataset to use, see
            the docstring of the base Config class for details.
        run_name: Name of the run (used for wandb/local save location), defaults to
            {dataset_name}_{model_name}.
        n_samples: No. samples in the whole training dataset (used to create dataset
            splits that are consistent with training jobs).
        metrics_samples: No. samples to compute metrics with (a subset of the train set)
        random_state: Random state to use for train/test split.
        use_wandb: Whether to use wandb for logging.
        wandb_args: Arguments to pass to wandb.init.
        local_save: Whether to save a local copy of the results or not.
        config_gen_dtime: When the config object was generated.
        caches: Where to cache the huggingface models and datasets.
        device: Which device to run inference on
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        metrics: list[str],
        metric_kwargs: dict,
        save_dir: str | None = None,
        dataset_args: str | None = None,
        run_name: str | None = None,
        n_samples: int | None = None,
        metrics_samples: int | None = None,
        random_state: int | None = None,
        use_wandb: bool = False,
        wandb_args: dict | None = None,
        local_save: bool = False,
        config_gen_dtime: str | None = None,
        caches: dict | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_args=dataset_args,
            n_samples=n_samples,
            run_name=run_name,
            random_state=random_state,
            use_wandb=use_wandb,
            wandb_args=wandb_args,
            config_gen_dtime=config_gen_dtime,
            caches=caches,
        )
        self.metrics = metrics
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else {}
        # `or n_samples` below to default to using whole train set if metrics_samples
        # is None
        self.metrics_samples = metrics_samples or n_samples
        self.save_dir = save_dir
        self.local_save = local_save
        self.wandb_args["job_type"] = "metrics"
        self.device = device

    @classmethod
    def from_dict(cls, config: dict) -> "MetricConfig":
        """Create a MetricConfig from a config dict.

        Args:
            config: Dict that must contain "model_name" and "dataset_name" keys. Can
                also contain "run_name", "random_state", "metrics",
                "save_dir", "dataset_args", "n_samples", "local_save",
                "config_gen_dtime", "caches", "use_wandb" and "wandb_args" keys. If
                "use_wandb" is not specified, it is set to True if "wandb" is in the
                config dict.

        Returns:
            MetricConfig object.
        """
        return cls(
            model_name=config["model_name"],
            dataset_name=config["dataset_name"],
            dataset_args=config.get("dataset_args"),
            metrics=config["metrics"],
            metric_kwargs=config["metric_kwargs"],
            save_dir=config.get("save_dir"),
            n_samples=config.get("n_samples"),
            metrics_samples=config.get("metrics_samples"),
            random_state=config.get("random_state"),
            use_wandb=config.get("use_wandb", "wandb_args" in config),
            wandb_args=config.get("wandb_args"),
            local_save=config.get("local_save"),
            config_gen_dtime=config.get("config_gen_dtime"),
            caches=config.get("caches"),
            device=config.get("device"),
        )

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "dataset_args": self.dataset_args,
            "metrics": self.metrics,
            "metric_kwargs": self.metric_kwargs,
            "save_dir": self.save_dir,
            "n_samples": self.n_samples,
            "metrics_samples": self.metrics_samples,
            "run_name": self.run_name,
            "random_state": self.random_state,
            "use_wandb": self.use_wandb,
            "wandb_args": self.wandb_args,
            "config_gen_dtime": self.config_gen_dtime,
            "local_save": self.local_save,
            "caches": self.caches,
            "device": self.device,
        }


class TopLevelMetricConfig(TopLevelConfig):

    """Takes a YAML file or dictionary with a top level config class containing all
    items to vary over for metric experiments, optionally producing and saving
    individual configs for each variant.

    Possible entries to vary over if multiple given:
        - models
        - dataset_names
        - n_samples
        - random_states

    Args:
        Must contain:
        - config_type: which config type to generate (metrics or train)
        - config_dir: where to save the generated configs to
        - metrics: (list of) metric(s) to run metric experiment on
        - models: (list of) model(s) to generate experiment configs
                                    for
        - dataset_names: (list of) dataset(s) to generate experiment
                                           configs for
        - dataset_args: See the base Config class for more details.
        - n_samples: (list of) training set size(s) to generate experiment
            configs for
        - metrics_samples: (list of) metrics dataset size(s) to generate experiment
            configs for (subset of the train set to use for metrics computation)

        Can also contain:
        - random_states: (list of) random state(s) to generate
            experiment configs for
        - wandb: weights and biases arguments
        - bask: baskerville computational arguments
        - use_bask: flag for using and generating baskerville run
        - caches: caching directories for models and datasets
        - slurm_template_path: path for setting jinja environment to look for jobscript
            template
        - slurm_template_name: path for jobscript template
        - config_gen_dtime: config generation date-time for keeping track of generated
            configs
        - inference_args: arguments required for inference in the metric experiments.
    """

    def __init__(
        self,
        config_type: str,
        config_dir: str,
        models: str | list[str],
        metrics: list[str],
        metric_kwargs: dict,
        n_samples: int | list[int],
        metrics_samples: int | list[int],
        dataset_names: str | list[str],
        dataset_args: str | list[str],
        keep_labels: list[list[str]] | list[list[int]] | None = None,
        save_dir: str | None = None,
        random_states: int | list[int] | None = None,
        wandb_args: dict | None = None,
        bask: dict | None = None,
        use_bask: bool = False,
        caches: dict | None = None,
        slurm_template_path: str | None = None,
        slurm_template_name: str | None = None,
        config_gen_dtime: str | None = None,
        inference_args: dict | None = None,
    ) -> None:
        super().__init__(
            config_type,
            config_dir,
            models,
            dataset_names,
            n_samples,
            dataset_args,
            keep_labels,
            random_states,
            wandb_args,
            bask,
            use_bask,
            caches,
            slurm_template_path,
            slurm_template_name,
            config_gen_dtime,
        )
        self.metrics = metrics
        self.metrics_samples = metrics_samples
        self.metric_kwargs = metric_kwargs
        self.save_dir = save_dir
        self.inference_args = inference_args or {}

    @classmethod
    def from_dict(
        cls, config: dict, config_type: str | None = None
    ) -> "TopLevelMetricConfig":
        """Generate a top level metric config object from a dictionary

        Args:
            config: config dictionary, must contain:
                    - config_type: label for type of experiment config.
                    - config_dir: which directory to save the specific configs to.
                    - models: which model(s) to run the fine tuning on.
                    - dataset_names: which dataset(s) to run the fine tuning on.
                    - slurm_template_path: where the slurm_template is

                    Can also contain "random_states", "n_samples", "caches",
                    "metric_kwargs", "dataset_args", "config_gen_dtime",
                    "use_wandb", "wandb_args", "use_bask", and "bask" keys. If
                    "use_wandb" is not specified, it is set to True if "wandb"
                    is in the config dict.
            config_type (optional): pass the config type to the class constructor
                                    explicitly. Defaults to None.
            inference_args (optional): pass inference specific arguments to the metric
                                       experiments

        Returns:
            TopLevelFineTuningConfig object
        """
        if config_type is not None:
            config_type = config_type
        else:
            config_type = config.get("config_type")
        return cls(
            config_type=config_type,
            config_dir=config["config_dir"],
            models=config["models"],
            dataset_names=config["dataset_names"],
            dataset_args=config["dataset_args"],
            keep_labels=config["keep_labels"],
            metrics=config["metrics"],
            metric_kwargs=config["metric_kwargs"],
            n_samples=config["n_samples"],
            metrics_samples=config["metrics_samples"],
            save_dir=config["save_dir"],
            random_states=config["random_states"],
            wandb_args=config["wandb_args"],
            bask=config["bask"],
            use_bask=config["use_bask"],
            caches=config["caches"],
            slurm_template_path=config["slurm_template_path"],
            slurm_template_name=config["slurm_template_name"],
            config_gen_dtime=config["config_gen_dtime"],
            inference_args=config["inference_args"],
        )

    def parameter_sweep(self) -> list[dict]:
        """Parameter sweep over entries with multiplicity. Returns config dictionaries
        with single variable values for these entries.

        Returns:
            list of config dictionaries for FineTuningConfig objects.
        """
        sweep_args = {
            "models": "model_name",
            "dataset_names": "dataset_name",
            "n_samples": "n_samples",
            "metrics_samples": "metrics_samples",
            "random_states": "random_state",
            "keep_labels": "keep_labels",
        }
        keep_args = [
            "save_dir",
            "wandb_args",
            "metrics",
            "metric_kwargs",
            "config_gen_dtime",
            "caches",
            "dataset_args",
        ]
        param_sweep_dicts = self._gen_sweep_dicts(sweep_args, keep_args)

        # Add remaining arguments not dealt with by _gen_sweep_dicts
        for pdict in param_sweep_dicts:
            pdict["device"] = self.inference_args.get("device")
            pdict["dataset_args"]["keep_labels"] = pdict["keep_labels"]

        return param_sweep_dicts

    def generate_sub_configs(self) -> list[Config]:
        """Generate the sub configs based on parameter_sweeps"""
        self.sub_configs = [
            MetricConfig.from_dict(config)
            for config in self.parameter_sweep()
            # exclude configs where metrics_samples > n_samples
            if config["metrics_samples"] is None
            or config["n_samples"] is None
            or config["metrics_samples"] <= config["n_samples"]
        ]
