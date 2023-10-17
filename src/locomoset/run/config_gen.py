"""
    Generate configs for runs of the MetricExperimentClass
"""

import warnings
from copy import copy

import yaml

import wandb


class MetricConfig:

    """Base MetricConfig class."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        metrics: list[str],
        save_dir: str | None = None,
        dataset_split: str | None = None,
        run_name: str | None = None,
        n_samples: int | None = None,
        random_state: int | None = None,
        use_wandb: bool = False,
        wandb_args: dict | None = None,
        local_save: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.save_dir = save_dir
        self.dataset_split = dataset_split or {"dataset_split": "train"}
        self.run_name = run_name or f"{dataset_name}_{model_name}".replace("/", "-")
        self.n_samples = n_samples or 50
        self.random_state = random_state
        self.use_wandb = use_wandb
        self.wandb_args = wandb_args or {}
        self.local_save = local_save

    @classmethod
    def from_dict(cls, config: dict) -> "MetricConfig":
        return cls(
            model_name=config["model_name"],
            dataset_name=config["dataset_name"],
            metrics=config["metrics"],
            save_dir=config.get("save_dir"),
            dataset_split=config.get("dataset_split"),
            n_samples=config.get("n_samples"),
            random_state=config.get("random_state"),
            use_wandb=config.get("use_wandb", "wandb" in config),
            wandb_args=config.get("wandb"),
        )

    @classmethod
    def read_yaml(cls, path: str) -> "MetricConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config=config)

    def init_wandb(self) -> None:
        if not self.use_wandb:
            warnings.warn("Ignored wandb initialisation as use_wandb=False")
            return
        if wandb.run is not None:
            raise ValueError("A wandb run has already been initialised")
        wandb.login()
        wandb_config = copy(self.wandb_args)
        if "name" not in wandb_config:
            wandb_config["name"] = self.run_name
        if "group" not in wandb_config:
            wandb_config["group"] = self.dataset_name
        wandb.init(config={"locomoset": self.to_dict()}, **wandb_config)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
            "save_dir": self.save_dir,
            "dataset_split": self.dataset_split,
            "n_samples": self.n_samples,
            "run_name": self.run_name,
            "random_state": self.random_state,
            "use_wandb": self.use_wandb,
            "wandb_args": self.wandb_args,
        }
