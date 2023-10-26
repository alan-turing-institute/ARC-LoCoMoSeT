"""
    Classes for configs for runs of the MetricExperimentClass
"""

import os
import warnings
from copy import copy
from datetime import datetime
from itertools import product

import wandb
import yaml
from jinja2 import Environment, FileSystemLoader


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
        config_gen_dtime: str | None = None,
        caches: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.save_dir = save_dir
        self.dataset_split = dataset_split or "train"
        self.run_name = run_name or f"{dataset_name}_{model_name}".replace("/", "-")
        self.n_samples = n_samples or 50
        self.random_state = random_state
        self.use_wandb = use_wandb
        self.wandb_args = wandb_args or {}
        self.local_save = local_save
        self.config_gen_dtime = config_gen_dtime
        self.caches = caches

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
            use_wandb=config.get("use_wandb", "wandb_args" in config),
            wandb_args=config.get("wandb_args"),
            local_save=config.get("local_save"),
            config_gen_dtime=config.get("config_gen_dtime"),
            caches=config.get("caches"),
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
            if self.config_gen_dtime is not None:
                wandb_config["group"] = f"{self.dataset_name}_{self.config_gen_dtime}"
            else:
                wandb_config["group"] = f"{self.dataset_name}"
        if "job_type" not in wandb_config:
            wandb_config["job_type"] = "metrics"
        print(wandb_config)
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
            "config_gen_dtime": self.config_gen_dtime,
            "local_save": self.local_save,
            "caches": self.caches,
        }


class TopLevelMetricConfig:

    """Takes a YAML file with a top level config class containing all items to vary over
    for metric experiments, optionally producing and saving individual configs for each
    variant.

    Possible entries to vary over if multiple given:
        - models
        - dataset_names
        - n_samples
        - random_states
    """

    def __init__(
        self,
        config_dir: str,
        models: str | list[str],
        metrics: list[str],
        dataset_names: str | list[str],
        dataset_splits: str | list[str],
        n_samples: int | list[int],
        save_dir: str | None = None,
        random_states: int | list[int] | None = None,
        wandb: dict | None = None,
        bask: dict | None = None,
        use_bask: bool = False,
        caches: dict | None = None,
    ) -> None:
        self.config_gen_dtime = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.config_dir = config_dir
        self.models = models
        self.metrics = metrics
        self.dataset_names = dataset_names
        self.dataset_splits = dataset_splits
        self.n_samples = n_samples
        self.save_dir: save_dir
        self.random_states = random_states
        self.wandb = wandb
        self.sub_configs = []
        self.save_dir = save_dir
        self.num_configs = 0
        self.bask = bask
        self.use_bask = use_bask
        self.caches = caches

    @classmethod
    def from_dict(cls, config: dict) -> "TopLevelMetricConfig":
        return cls(
            config_dir=config["config_dir"],
            models=config["models"],
            dataset_names=config["dataset_names"],
            metrics=config["metrics"],
            save_dir=config.get("save_dir"),
            dataset_splits=config.get("dataset_split"),
            n_samples=config.get("n_samples"),
            random_states=config.get("random_states"),
            wandb=config.get("wandb"),
            bask=config.get("bask"),
            use_bask=config.get("use_bask"),
            cahces=config.get("caches"),
        )

    @classmethod
    def read_yaml(cls, path: str) -> "TopLevelMetricConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config=config)

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
        if isinstance(self.n_samples, list):
            sweep_dict["n_samples"] = copy(self.n_samples)
        else:
            sweep_dict["n_samples"] = [copy(self.n_samples)]
        if isinstance(self.random_states, list):
            sweep_dict["random_state"] = copy(self.random_states)
        else:
            sweep_dict["random_state"] = [copy(self.random_states)]

        sweep_dict_keys, sweep_dict_vals = zip(*sweep_dict.items())
        param_sweep_dicts = [
            dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
        ]
        for pdict in param_sweep_dicts:
            pdict["save_dir"] = self.save_dir
            pdict["wandb_args"] = self.wandb
            pdict["metrics"] = self.metrics
            pdict["config_gen_dtime"] = self.config_gen_dtime
            pdict["caches"] = self.caches
        self.num_configs = len(param_sweep_dicts)
        if self.num_configs > 1001:
            warnings.warn("Slurm array jobs cannot exceed more than 1001!")
        return param_sweep_dicts

    def generate_sub_configs(self) -> list[MetricConfig]:
        """Generate the sub configs based on parameter_sweeps"""
        self.sub_configs = [
            MetricConfig.from_dict(config) for config in self.parameter_sweep()
        ]

    def create_bask_job_script(self, array_number) -> None:
        """Generates a baskervill jobscript from template"""
        bask_pars = {}
        bask_pars["job_name"] = self.bask.get("job_name", "locomoset_metric_experiment")
        bask_pars["walltime"] = self.bask.get("walltime", "0-0:30:0")
        bask_pars["node_number"] = self.bask.get("node_number", 1)
        bask_pars["gpu_number"] = self.bask.get("gpu_number", 1)
        bask_pars["cpu_per_gpu"] = self.bask.get("cpu_per_gpu", 36)
        config_path = f"{self.config_dir}/{self.config_gen_dtime}"
        bask_pars["config_path"] = config_path
        bask_pars["array_number"] = array_number

        jenv = Environment(loader=FileSystemLoader("templates/"))
        template = jenv.get_template("metric_exp_slurm_template.sh")
        content = template.render(bask_pars)
        file_name = f"metric_exp_jobscript_{self.config_gen_dtime}.sh"
        with open(f"{config_path}/{file_name}", "w") as f:
            f.write(content)

    def save_sub_configs(self) -> None:
        """Save the generated subconfigs"""
        configs_path = f"{self.config_dir}/{self.config_gen_dtime}"
        os.mkdir(configs_path)
        for idx, config in enumerate(self.sub_configs):
            # save with +1 as slurm array jobs index from 1 not 0!
            with open(f"{configs_path}/config_{idx+1}.yaml", "w") as f:
                yaml.safe_dump(config.to_dict(), f)
