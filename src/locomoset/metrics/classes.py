"""
Base metric class for unifying the method.
"""
import warnings
from abc import ABC, abstractmethod
from copy import copy
from itertools import product

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
        n_samples: How many samples to use in the metric experiments.
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
        keep_labels: list[str] | list[int] | None = None,
        keep_size: int | float | None = None,
        run_name: str | None = None,
        n_samples: int | None = None,
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
            keep_labels=keep_labels,
            keep_size=keep_size,
            run_name=run_name,
            random_state=random_state,
            use_wandb=use_wandb,
            wandb_args=wandb_args,
            config_gen_dtime=config_gen_dtime,
            caches=caches,
        )
        self.metrics = metrics
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else {}
        self.save_dir = save_dir
        self.n_samples = n_samples or 50
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
            keep_labels=config["keep_labels"],
            keep_size=config["keep_size"],
            metrics=config["metrics"],
            metric_kwargs=config["metric_kwargs"],
            save_dir=config.get("save_dir"),
            n_samples=config.get("n_samples"),
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
            "save_dir": self.save_dir,
            "n_samples": self.n_samples,
            "run_name": self.run_name,
            "random_state": self.random_state,
            "use_wandb": self.use_wandb,
            "wandb_args": self.wandb_args,
            "config_gen_dtime": self.config_gen_dtime,
            "local_save": self.local_save,
            "caches": self.caches,
            "device": self.device,
        }

    def init_results(self) -> None:
        self.inference_times = {}
        self.metric_scores = {}


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
        - dataset_args: Optionally containing 'metrics_split', the
            dataset split to run metrics experiments on. See the base Config class for
            more details.
        - n_samples: (list of) sample number(s) to generate experiment
            configs for

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
        dataset_names: str | list[str],
        dataset_args: str | list[str],
        keep_labels: list[list[str]] | list[list[int]] | None = None,
        keep_sizes: list[int] | list[float] | None = None,
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
        if dataset_args is not None and "metrics_split" not in dataset_args:
            dataset_args["metrics_split"] = dataset_args.get("train_split", "train")
        if dataset_args is None:
            dataset_args = {"metrics_split": "train"}

        super().__init__(
            config_type,
            config_dir,
            models,
            dataset_names,
            dataset_args,
            keep_labels,
            keep_sizes,
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
        self.metric_kwargs = (metric_kwargs,)
        self.n_samples = n_samples
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
            dataset_names=config["dataset_name"],
            dataset_args=config["dataset_args"],
            keep_labels=config["keep_labels"],
            keep_sizes=config["keep_sizes"],
            metrics=config["metrics"],
            metric_kwargs=config["metric_kwargs"],
            n_samples=config["n_samples"],
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
        sweep_dict = {}

        if isinstance(self.models, list):
            sweep_dict["model_name"] = copy(self.models)
        else:
            sweep_dict["model_name"] = [copy(self.models)]
        if isinstance(self.n_samples, list):
            sweep_dict["n_samples"] = copy(self.n_samples)
        else:
            sweep_dict["n_samples"] = [copy(self.n_samples)]
        if isinstance(self.random_states, list):
            sweep_dict["random_state"] = copy(self.random_states)
        else:
            sweep_dict["random_state"] = [copy(self.random_states)]
        if isinstance(self.keep_labels, list):
            sweep_dict["keep_labels"] = copy(self.keep_labels)
        else:
            sweep_dict["keep_labels"] = [copy(self.keep_labels)]
        if isinstance(self.keep_sizes, list):
            sweep_dict["keep_size"] = copy(self.keep_sizes)
        else:
            sweep_dict["keep_size"] = [copy(self.keep_sizes)]

        sweep_dict_keys, sweep_dict_vals = zip(*sweep_dict.items())
        param_sweep_dicts = [
            dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
        ]

        # input inference args
        device = self.inference_args.get("device")

        for pdict in param_sweep_dicts:
            pdict["dataset_name"] = self.dataset_name
            pdict["save_dir"] = self.save_dir
            pdict["wandb_args"] = self.wandb_args
            pdict["metrics"] = self.metrics
            pdict["metric_kwargs"] = self.metric_kwargs
            pdict["config_gen_dtime"] = self.config_gen_dtime
            pdict["caches"] = self.caches
            pdict["device"] = device
            pdict["dataset_name"] = self.dataset_name
            pdict["dataset_args"] = self.dataset_args

        self.num_configs = len(param_sweep_dicts)
        if self.num_configs > 1001:
            warnings.warn("Slurm array jobs cannot exceed more than 1001!")
        return param_sweep_dicts

    def generate_sub_configs(self) -> list[Config]:
        """Generate the sub configs based on parameter_sweeps"""
        self.sub_configs = [
            MetricConfig.from_dict(config) for config in self.parameter_sweep()
        ]
