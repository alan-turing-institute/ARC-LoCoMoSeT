"""
    Base classes for config objects and config generating objects for experiments.
"""

import os
import warnings
from abc import ABC, abstractclassmethod, abstractmethod
from copy import copy
from datetime import datetime
from pathlib import Path

import wandb
import yaml
from jinja2 import Environment, FileSystemLoader


class Config(ABC):

    """Base class for config objects

    Attributes:
        model_name: Name of the HuggingFace model to fine-tune.
        dataset_name: Name of the HuggingFace dataset to use for fine-tuning.
        run_name: Name of the run (used for wandb/local save location), defaults to
            {dataset_name}_{model_name}.
        dataset_args: Dict defining the splits and columns of the dataset to use,
            optionally including the keys "train_split" (default: "train"),
            "val_split" (default: None, in which case the validation set will be created
            from the training split), "metrics_split" (default: value of train_split),
            "image_field" (default: "image"), and "label_field" (default: "label").
        random_state: Random state to use for train/test split and training.
        use_wandb: Whether to use wandb for logging.
        wandb_args: Arguments passed to wandb.init, as well as optionally a "log_model"
            value which will be used to set the WANDB_LOG_MODEL environment variable
            which controls the model artifact saving behaviour.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_args: dict | None = None,
        keep_labels: list[str] | list[int] | None = None,
        n_samples: int | None = None,
        random_state: int | None = None,
        config_gen_dtime: str | None = None,
        caches: dict | None = None,
        wandb_args: dict | None = None,
        use_wandb: bool = False,
        run_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.config_gen_dtime = config_gen_dtime
        self.caches = caches
        self.use_wandb = use_wandb
        self.wandb_args = wandb_args or {}
        self.run_name = run_name or f"{dataset_name}_{model_name}".replace("/", "-")
        self.dataset_args = dataset_args or {"train_split": "train"}
        self.n_samples = n_samples
        self.dataset_args["keep_labels"] = keep_labels
        if "image_field" not in self.dataset_args:
            self.dataset_args["image_field"] = "image"
        if "label_field" not in self.dataset_args:
            self.dataset_args["label_field"] = "label"

    def init_wandb(self) -> None:
        """Initialise a wandb run if the config specifies to use wandb and a run has not
        already been initialised.

        If name, group and job_type and not specificied in the input config then they
        are set as:
                name: run_name
                group: data_set_name_config_gen_dtime OR data_set_name
                job_type: misc
        """
        if not self.use_wandb:
            warnings.warn("Ignored wandb initialisation as use_wandb=False")
            return
        if wandb.run is not None:
            raise ValueError("A wandb run has already been initialised")

        if "wandb" in self.caches:
            # where wandb artifacts will be cached
            os.environ["WANDB_CACHE_DIR"] = self.caches["wandb"]
            os.environ["WANDB_DATA_DIR"] = self.caches["wandb"]

        wandb.login()
        wandb_config = copy(self.wandb_args)

        if "log_model" in wandb_config:
            # log_model can only be specified as an env variable, so we set the env
            # variable then remove it from the init args.
            os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]
            wandb_config.pop("log_model")

        # set default names for any that haven't been specified
        if "name" not in wandb_config:
            wandb_config["name"] = self.run_name
        if "group" not in wandb_config:
            if self.config_gen_dtime is not None:
                wandb_config["group"] = f"{self.dataset_name}_{self.config_gen_dtime}"
            else:
                wandb_config["group"] = f"{self.dataset_name}"
        if "job_type" not in wandb_config:
            raise ValueError("No Job type given")

        wandb.init(config={"locomoset": self.to_dict()}, **wandb_config)

    @abstractclassmethod
    def from_dict(cls, dict) -> "Config":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_name" and "dataset_name" keys. Can
                also contain "run_name", "random_state", "dataset_args",
                "training_args", "use_wandb" and "wandb_args" keys. If "use_wandb" is
                not specified, it is set to True if "wandb" is in the config dict.

        Returns:
            FineTuningConfig object.
        """
        raise NotImplementedError

    @classmethod
    def read_yaml(cls, path: str) -> "Config":
        """Create a FineTuningConfig from a yaml file.

        Args:
            path: Path to yaml file.

        Returns:
            FineTuningConfig object.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config=config)

    @abstractclassmethod
    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        raise NotImplementedError


class TopLevelConfig(ABC):
    """Takes a YAML file or dictionary with a top level config class containing all
    items to vary over for experiments, optionally producing and saving individual
    configs for each variant.

    Possible entries to vary over if multiple given:
        - models
        - dataset_name
        - n_samples
        - random_states

    Args:
        Must contain:
        - config_type: which config type to generate (metrics or train)
        - config_dir: where to save the generated configs to
        - models: (list of) model(s) to generate experiment configs
            for
        - dataset_name: (list of) dataset(s) to generate experiment
            configs for

        Can also contain:
        - dataset_args: Dict defining the splits and columns of the dataset to use, see
            the docstring of the Config class for details.
        - random_states: (list of) random state(s) to generate
            experiment configs for
        - wandb: weights and biases arguments
        - bask: baskerville computational arguments
        - use_bask: flag for using and generating baskerville run
        - caches: caching directories for models, datasets, and wandb
        - slurm_template_path: path for setting jinja environment to look for jobscript
            template
        - slurm_template_name: path for jobscript template
        - config_gen_dtime: config generation date-time for keeping track of generated
            configs
    """

    def __init__(
        self,
        config_type: str,
        config_dir: str,
        models: str | list[str],
        dataset_name: str | list[str],
        n_samples: int | list[int],
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
        self.config_type = config_type
        self.config_gen_dtime = config_gen_dtime or datetime.now().strftime(
            "%Y%m%d-%H%M%S-%f"
        )
        self.config_dir = config_dir
        self.models = models
        self.dataset_name = dataset_name
        self.dataset_args = dataset_args
        self.keep_labels = keep_labels
        self.n_samples = n_samples
        self.random_states = random_states
        self.wandb_args = wandb_args
        self.sub_configs = []
        self.num_configs = 0
        self.bask = bask
        self.use_bask = use_bask
        self.caches = caches
        self.slurm_template_path = slurm_template_path or str(
            Path("src", "locomoset", "config/").resolve()
        )
        self.slurm_template_name = slurm_template_name or "jobscript_template.sh"

    @abstractclassmethod
    def from_dict(
        cls, config: dict, config_type: str | None = None
    ) -> "TopLevelConfig":
        """Generate a config generator object from an input dictionary. Parameters are
        specifc to each experiment type and so must be implemented in child class.

        Args:
            config: config dictionary
            config_type (optional): pass the config type to the class constructor
                                    explicitly. Defaults to None.
        """
        raise NotImplementedError

    @classmethod
    def read_yaml(cls, path: str, config_type: str | None = None) -> "TopLevelConfig":
        """Generate a config generator object from an (path to) a yaml file.

        Args:
            path: path to YAML file containing top level config.
            config_type (optional): pass the config type to the class constructor
                                    explicitly. Defaults to None.

        Returns:
            TopLevelConfig object.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config=config, config_type=config_type)

    @abstractmethod
    def parameter_sweep(self) -> list[dict]:
        """Parameter sweep over entries with multiplicity. Specific choice of variable
        over which to vary and by experiment type and so must be implemented in child
        class."""
        raise NotImplementedError

    @abstractmethod
    def generate_sub_configs(self) -> list[Config]:
        """Generate all sub configs from a config generator. Type of config to be
        generated is specific to esperiment type and so must be implemented in child
        class."""
        raise NotImplementedError

    def create_bask_job_script(self, array_number) -> None:
        """Generates a baskervill jobscript from template.

        Args:
            array_number: number of configs to vary over, input from the parameter_sweep
                            method.

        Returns:
            Saves specific baskerville jobscript with correct labels, parameters and
            paths.
        """
        bask_pars = {}
        bask = self.bask[self.config_type]
        bask_pars["job_name"] = bask.get("job_name", "locomoset_experiment")
        bask_pars["walltime"] = bask.get("walltime", "0-0:30:0")
        bask_pars["node_number"] = bask.get("node_number", 1)
        bask_pars["gpu_number"] = bask.get("gpu_number", 1)
        bask_pars["cpu_per_gpu"] = bask.get("cpu_per_gpu", 36)
        config_path = f"{self.config_dir}/{self.config_gen_dtime}"
        bask_pars["config_path"] = config_path
        bask_pars["array_number"] = array_number
        bask_pars["config_type"] = self.config_type

        jenv = Environment(loader=FileSystemLoader(self.slurm_template_path))
        template = jenv.get_template(self.slurm_template_name)
        content = template.render(bask_pars)
        file_name = f"{self.config_type}_jobscript_{self.config_gen_dtime}.sh"
        with open(f"{config_path}/{file_name}", "w") as f:
            f.write(content)

    def save_sub_configs(self) -> None:
        """Save the generated subconfigs to a top level director given by the config
        directory and a specific directory given by the date time that the configs have
        been generated"""
        configs_path = f"{self.config_dir}/{self.config_gen_dtime}"
        os.makedirs(configs_path, exist_ok=True)
        for idx, config in enumerate(self.sub_configs):
            # save with +1 as slurm array jobs index from 1 not 0!
            with open(
                f"{configs_path}/config_{self.config_type}_{idx+1}.yaml", "w"
            ) as f:
                yaml.safe_dump(config.to_dict(), f)
