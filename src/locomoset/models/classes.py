"""
    Class for train config
"""

import warnings
from copy import copy
from itertools import product

from transformers import TrainingArguments

from locomoset.run.config_classes import Config, TopLevelConfig


class FineTuningConfig(Config):

    """Fine-tuning configuration class.

    Attributes:
        model_name: Name of the HuggingFace model to fine-tune.
        dataset_name: Name of the HuggingFace dataset to use for fine-tuning.
        run_name: Name of the run (used for wandb/local save location), defaults to
            {dataset_name}_{model_name}.
        random_state: Random state to use for train/test split and training.
        dataset_args: Dict defining "train_split" and "val_split" (optional), defaults
            to {"train_split": "train"}.
        training_args: Dict of arguments to pass to TrainingArguments.
        use_wandb: Whether to use wandb for logging.
        wandb_args: Arguments to pass to wandb.init.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        random_state: int | None = None,
        config_gen_dtime: str | None = None,
        caches: dict | None = None,
        wandb_args: dict | None = None,
        use_wandb: bool = False,
        run_name: str | None = None,
        dataset_args: dict | None = None,
        training_args: dict | None = None,
    ) -> None:
        super().__init__(
            model_name,
            dataset_name,
            random_state,
            config_gen_dtime,
            caches,
            wandb_args,
            use_wandb,
            run_name,
        )
        self.dataset_args = dataset_args or {"train_split": "train"}
        self.training_args = training_args or {}

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
            run_name=config.get("run_name"),
            random_state=config.get("random_state"),
            dataset_args=config.get("dataset_args"),
            training_args=config.get("training_args"),
            use_wandb=config.get("use_wandb", "wandb_args" in config),
            wandb_args=config.get("wandb_args"),
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
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "run_name": self.run_name,
            "random_state": self.random_state,
            "dataset_args": self.dataset_args,
            "training_args": self.training_args,
            "use_wandb": self.use_wandb,
            "wandb_args": self.wandb_args,
        }


class TopLevelFineTuningConfig(TopLevelConfig):

    """Class for generating Fine Tuning Configs from a top level config file"""

    def __init__(
        self,
        config_type: str,
        config_dir: str,
        models: str | list[str],
        dataset_names: str | list[str],
        random_states: int | list[int] | None = None,
        wandb: dict | None = None,
        bask: dict | None = None,
        use_bask: bool = False,
        caches: dict | None = None,
        slurm_template_path: str | None = None,
        config_gen_dtime: str | None = None,
        dataset_args: dict | None = None,
        training_args: dict | None = None,
    ) -> None:
        super().__init__(
            config_type,
            config_dir,
            models,
            dataset_names,
            random_states,
            wandb,
            bask,
            use_bask,
            caches,
            slurm_template_path,
            config_gen_dtime,
        )
        self.dataset_args = dataset_args
        self.training_args = training_args

    @classmethod
    def from_dict(cls, config: dict) -> "TopLevelFineTuningConfig":
        """Generate a top level fine tuning config object from a dectionary"""
        return cls(
            config_type=config["config_type"],
            config_dir=config["config_dir"],
            models=config["models"],
            dataset_names=config["dataset_names"],
            random_states=config.get("random_state"),
            wandb=config.get("wandb"),
            config_gen_dtime=config.get("config_gen_dtime"),
            caches=config.get("caches"),
            slurm_template_path=config.get("slurm_template_path"),
            dataset_args=config.get("dataset_args"),
            training_args=config.get("training_args"),
            use_bask=config.get("use_bask"),
            bask=config.get("bask"),
        )

    def parameter_sweep(self) -> list[dict]:
        """Parameter sweep over entries with multiplicity."""
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

        sweep_dict_keys, sweep_dict_vals = zip(*sweep_dict.items())
        param_sweep_dicts = [
            dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
        ]
        for pdict in param_sweep_dicts:
            pdict["wandb_args"] = self.wandb
            pdict["config_gen_dtime"] = self.config_gen_dtime
            pdict["caches"] = self.caches
            pdict["dataset_args"] = self.dataset_args
            pdict["training_args"] = self.training_args
        self.num_configs = len(param_sweep_dicts)
        if self.num_configs > 1001:
            warnings.warn("Slurm array jobs cannot exceed more than 1001!")
        return param_sweep_dicts

    def generate_sub_configs(self) -> list[Config]:
        """Generate the sub configs based on parameter_sweeps"""
        self.sub_configs = [
            FineTuningConfig.from_dict(config) for config in self.parameter_sweep()
        ]
