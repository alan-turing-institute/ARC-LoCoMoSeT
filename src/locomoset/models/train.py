import os
import warnings
from copy import copy
from typing import Callable

import evaluate
import numpy as np
import wandb
import yaml
from datasets import Dataset, load_dataset
from transformers import EvalPrediction, PreTrainedModel, Trainer, TrainingArguments

from locomoset.datasets.preprocess import prepare_training_data
from locomoset.models.load import get_model_with_dataset_labels, get_processor


def get_metrics_fn(metric_name="accuracy") -> Callable:
    """Get a function to use a HuggingFace metrics function for the given metric
    name.

    Args:
        metric_name: Name of a HuggingFace metric

    Returns:
        Function that takes predictions and ground truths (as a EvalPrediction object)
        and returns a dictionary with metric scores.

    """
    metric = evaluate.load(metric_name)

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def train(
    model: PreTrainedModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: TrainingArguments,
) -> Trainer:
    """Train a model on a dataset and evaluate it on a validation set and save the
    results either locally or to wandb.

    Args:
        model: HuggingFace model to train. Should have been loaded with the number of
            labels and the class labels set to be compatible with the training dataset.
        train_dataset: Preprocessed dataset to train on.
        val_dataset: Preprocessed dataset to evaluate on.
        training_args: TrainingArguments to use for training.

    Returns:
        Trainer object.
    """
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=get_metrics_fn(),
    )
    trainer.train()

    if "wandb" not in training_args.report_to or wandb.run is None:
        # save results locally
        trainer.save_model()
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("eval", metrics)

    return trainer


class FineTuningConfig:
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
        run_name: str | None = None,
        random_state: int | None = None,
        dataset_args: dict | None = None,
        training_args: dict | None = None,
        use_wandb: bool = False,
        wandb_args: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.run_name = run_name or f"{dataset_name}_{model_name}".replace("/", "-")
        self.random_state = random_state
        self.dataset_args = dataset_args or {"train_split": "train"}
        self.training_args = training_args or {}
        self.use_wandb = use_wandb
        self.wandb_args = wandb_args or {}

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
            use_wandb=config.get("use_wandb", "wandb" in config),
            wandb_args=config.get("wandb"),
        )

    @classmethod
    def read_yaml(cls, path: str) -> "FineTuningConfig":
        """Create a FineTuningConfig from a yaml file.

        Args:
            path: Path to yaml file.

        Returns:
            FineTuningConfig object.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def init_wandb(self) -> None:
        """Initialise a wandb run if the config specifies to use wandb and a run has not
        already been initialised."""
        if not self.use_wandb:
            warnings.warn("Ignored wandb initialisation as use_wandb=False")
            return
        if wandb.run is not None:
            raise ValueError("A wandb run has already been initialised")

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
            wandb_config["group"] = self.dataset_name
        if "job_type" not in wandb_config:
            wandb_config["job_type"] = "train"

        wandb.init(config={"locomoset": self.to_dict()}, **wandb_config)

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


def run_config(config: FineTuningConfig) -> Trainer:
    """Run the fine-tuning job specified by a FineTuningConfig: Load a dataset and
    model, preprocess it, train the model, and save the results.

    Args:
        config: A FineTuningConfig object.

    Returns:
        Trainer object.
    """
    processor = get_processor(config.model_name)

    train_split = config.dataset_args["train_split"]
    val_split = config.dataset_args.get("val_split", None)
    if val_split is None or val_split == train_split:
        dataset = load_dataset(config.dataset_name, split=train_split)
    else:
        dataset = load_dataset(config.dataset_name)

    train_dataset, val_dataset = prepare_training_data(
        dataset,
        processor,
        train_split,
        val_split,
        config.random_state,
        config.dataset_args.get("test_size"),
    )

    model = get_model_with_dataset_labels(config.model_name, train_dataset)

    if config.use_wandb:
        config.init_wandb()

    return train(model, train_dataset, val_dataset, config.get_training_args())
