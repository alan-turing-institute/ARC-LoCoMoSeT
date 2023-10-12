import argparse
import os
import warnings
from copy import copy
from typing import Callable

import evaluate
import numpy as np
import wandb
import yaml
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments

from locomoset.datasets.preprocess import prepare_training_data
from locomoset.models.load import get_model_with_dataset_labels, get_processor


def get_metrics_fn(metric_name="accuracy") -> Callable:
    accuracy = evaluate.load(metric_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics


def train(
    model: PreTrainedModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: TrainingArguments,
) -> None:
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


class FineTuningConfig:
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
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def init_wandb(self) -> None:
        if not self.use_wandb:
            warnings.warn("Ignored wandb initialisation as use_wandb=False")
            return
        if wandb.run is not None:
            raise ValueError("A wandb run has already been initialised")
        wandb.login()
        wandb_config = copy(self.wandb_args)
        if "log_model" in wandb_config:
            # log_model can only be specified as an env variable
            os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]
            wandb_config.pop("log_model")
        if "name" not in wandb_config:
            wandb_config["name"] = self.run_name
        if "group" not in wandb_config:
            wandb_config["group"] = self.dataset_name
        if "job_type" not in wandb_config:
            wandb_config["job_type"] = "train"
        wandb.init(config={"locomoset": self.to_dict()}, **wandb_config)

    def get_training_args(self) -> TrainingArguments:
        training_args = copy(self.training_args)
        training_args["seed"] = self.random_state
        training_args["run_name"] = self.run_name
        training_args["report_to"] = "wandb" if self.use_wandb else "none"
        if training_args["report_to"] == "none":
            training_args["output_dir"] += f"/{self.run_name}"
        return TrainingArguments(**training_args)

    def to_dict(self) -> dict:
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


def run_config(config: FineTuningConfig) -> None:
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

    train(model, train_dataset, val_dataset, config.get_training_args())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model given a training config."
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    config = FineTuningConfig.read_yaml(args.configfile)
    run_config(config)


if __name__ == "__main__":
    main()
