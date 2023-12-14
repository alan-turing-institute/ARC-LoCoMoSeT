"""
    Class for train config
"""

import warnings
from copy import copy
from itertools import product

from transformers import TrainingArguments

from locomoset.config.config_classes import Config, TopLevelConfig


class FineTuningConfig(Config):

    """Fine-tuning configuration class.

    Attributes:
        model_name: Name of the HuggingFace model to fine-tune.
        dataset_name: Name of the HuggingFace dataset to use for fine-tuning.
        run_name: Name of the run (used for wandb/local save location), defaults to
            {dataset_name}_{model_name}.
        random_state: Random state to use for train/test split and training.
        dataset_args: Dict defining dataset splits and columns, see the docstring of the
            base Config class.
        training_args: Dict of arguments to pass to TrainingArguments.
        use_wandb: Whether to use wandb for logging.
        wandb_args: Arguments to pass to wandb.init.
        config_gen_dtime: when the config object was created.
        caches: where to cache the huggingface models and datasets.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_args: dict | None = None,
        n_samples: int | None = None,
        random_state: int | None = None,
        config_gen_dtime: str | None = None,
        caches: dict | None = None,
        wandb_args: dict | None = None,
        use_wandb: bool = False,
        run_name: str | None = None,
        training_args: dict | None = None,
    ) -> None:
        super().__init__(
            model_name,
            dataset_name,
            dataset_args,
            n_samples,
            random_state,
            config_gen_dtime,
            caches,
            wandb_args,
            use_wandb,
            run_name,
        )
        self.training_args = training_args or {}
        self.wandb_args["job_type"] = "train"

    @classmethod
    def from_dict(cls, config: dict) -> "FineTuningConfig":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_name" and "dataset_name" keys. Can
                also contain "run_name", "random_state", "dataset_args",
                "training_args", "use_wandb" and "wandb_args" keys. If "use_wandb" is
                not specified, it is set to True if "wandb" is in the config dict.

        Returns:
            FineTuningConfig object.
        """
        return cls(
            model_name=config["model_name"],
            dataset_name=config["dataset_name"],
            dataset_args=config.get("dataset_args"),
            n_samples=config["n_samples"],
            run_name=config.get("run_name"),
            random_state=config.get("random_state"),
            training_args=config.get("training_args"),
            use_wandb=config.get("use_wandb", "wandb_args" in config),
            wandb_args=config.get("wandb_args"),
            caches=config.get("caches"),
            config_gen_dtime=config.get("config_gen_dtime"),
        )

    def get_training_args(self) -> TrainingArguments:
        """Get a TrainingArguments object based on the config. Use the training_args
        attribute of the config as a base, adding in seed, run_name, and output_dir
        using other attributes in the config class where needed.

        Returns:
            TrainingArguments object.
        """
        training_args = copy(self.training_args)
        training_args["seed"] = self.random_state
        training_args["run_name"] = self.run_name
        training_args["report_to"] = "wandb" if self.use_wandb else "none"
        if training_args["report_to"] == "none":
            training_args["output_dir"] += f"/{self.run_name}"
        return TrainingArguments(**training_args)

    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        return {
            "caches": self.caches,
            "config_gen_dtime": self.config_gen_dtime,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "dataset_args": self.dataset_args,
            "n_samples": self.n_samples,
            "run_name": self.run_name,
            "random_state": self.random_state,
            "training_args": self.training_args,
            "use_wandb": self.use_wandb,
            "wandb_args": self.wandb_args,
        }


class TopLevelFineTuningConfig(TopLevelConfig):

    """Takes a YAML file or dictionary with a top level config class containing all
    items to vary over for fine tuning experiments, optionally producing and saving
    individual configs for each variant.

    Possible entries to vary over if multiple given:
        - models
        - dataset_name
        - random_states

    Args:
        Must contain:
        - config_type: which config type to generate (metrics or train)
        - config_dir: where to save the generated configs to
        - models: (list of) model(s) to generate experiment configs for
        - dataset_name: (list of) dataset(s) to generate experiment configs for
        - random_states: (list of) random state(s) to generate experiment configs for
        - wandb: weights and biases arguments
        - bask: baskerville computational arguments
        - use_bask: flag for using and generating baskerville run
        - caches: caching directories for models and datasets
        - slurm_template_path: path for setting jinja environment to look
            for jobscript template
        - slurm_template_name: path for jobscript template
        - config_gen_dtime: config generation date-time for keeping track of generated
            configs
        - dataset_args: dataset arguments for training purposes
        - training_args: arguments for training
    """

    def __init__(
        self,
        config_type: str,
        config_dir: str,
        models: str | list[str],
        dataset_names: str | list[str],
        n_samples: int | list[int],
        training_args: dict,
        dataset_args: dict | None = None,
        keep_labels: list[list[str]] | list[list[int]] | None = None,
        random_states: int | list[int] | None = None,
        wandb_args: dict | None = None,
        bask: dict | None = None,
        use_bask: bool = False,
        caches: dict | None = None,
        slurm_template_path: str | None = None,
        slurm_template_name: str | None = None,
        config_gen_dtime: str | None = None,
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
        self.training_args = training_args

    @classmethod
    def from_dict(
        cls, config: dict, config_type: str | None = None
    ) -> "TopLevelFineTuningConfig":
        """Generate a top level fine tuning config object from a dictionary

        Args:
            config: config dictionary, must contain:
                    - config_type: label for type of experiment config.
                    - config_dir: which directory to save the specific configs to.
                    - models: which model(s) to run the fine tuning on.
                    - dataset_name: which dataset(s) to run the fine tuning on.
                    - slurm_template_path: where the slurm_template is

                    Can also contain "random_states", "dataset_args",
                    "config_gen_dtime", "training_args", "use_wandb", "wandb_args",
                    "use_bask" and "bask" keys. If "use_wandb" is not specified, it is
                    set to True if "wandb" is in the config dict.
            config_type (optional): pass the config type to the class constructor
                                    explicitly. Defaults to None.

        Returns:
            TopLevelFineTuningConfig object
        """
        if config_type is not None:
            config_type = config_type
        else:
            config_type = config.get("config_type")
        return cls(
            config_type=config_type,
            config_dir=config.get("config_dir"),
            models=config.get("models"),
            dataset_names=config["dataset_names"],
            n_samples=config["n_samples"],
            dataset_args=config.get("dataset_args"),
            keep_labels=config["keep_labels"],
            random_states=config.get("random_states"),
            wandb_args=config.get("wandb_args"),
            config_gen_dtime=config.get("config_gen_dtime"),
            caches=config.get("caches"),
            slurm_template_path=config.get("slurm_template_path"),
            slurm_template_name=config.get("slurm_template_name"),
            training_args=config.get("training_args"),
            use_bask=config.get("use_bask"),
            bask=config.get("bask"),
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
        if isinstance(self.dataset_names, list):
            sweep_dict["dataset_name"] = copy(self.dataset_names)
        else:
            sweep_dict["dataset_name"] = [copy(self.dataset_names)]
        if isinstance(self.random_states, list):
            sweep_dict["random_state"] = copy(self.random_states)
        else:
            sweep_dict["random_state"] = [copy(self.random_states)]
        if isinstance(self.keep_labels, list):
            sweep_dict["keep_labels"] = copy(self.keep_labels)
        else:
            sweep_dict["keep_labels"] = [copy(self.keep_labels)]
        if isinstance(self.n_samples, list):
            sweep_dict["n_samples"] = copy(self.n_samples)
        else:
            sweep_dict["n_samples"] = [copy(self.n_samples)]

        sweep_dict_keys, sweep_dict_vals = zip(*sweep_dict.items())
        param_sweep_dicts = [
            dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
        ]
        for pdict in param_sweep_dicts:
            pdict["wandb_args"] = self.wandb_args
            pdict["config_gen_dtime"] = self.config_gen_dtime
            pdict["caches"] = self.caches
            pdict["dataset_args"] = self.dataset_args
            pdict["training_args"] = self.training_args
            pdict["dataset_args"]["keep_labels"] = pdict["keep_labels"]
        self.num_configs = len(param_sweep_dicts)
        if self.num_configs > 1001:
            warnings.warn("Slurm array jobs cannot exceed more than 1001!")
        return param_sweep_dicts

    def generate_sub_configs(self) -> list[Config]:
        """Generate the sub configs objects based on parameter_sweeps."""
        self.sub_configs = [
            FineTuningConfig.from_dict(config) for config in self.parameter_sweep()
        ]
