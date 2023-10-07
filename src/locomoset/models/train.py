import argparse
import os

import evaluate
import numpy as np
import yaml
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments

import wandb
from locomoset.datasets.preprocess import prepare_training_data
from locomoset.models.load import get_model_with_dataset_labels, get_processor


def get_metrics_fn(metric_name="accuracy"):
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
):
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=get_metrics_fn(),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    trainer.log_metrics("eval", metrics)
    # save metrics and model / setup wandb etc.


def run_config(config: dict):
    processor = get_processor(config["model_name"])

    train_split = config["train_split"]
    val_split = config.get("val_split", None)
    if val_split is None or val_split == train_split:
        dataset = load_dataset(config["dataset_name"], split=train_split)
    else:
        dataset = load_dataset(config["dataset_name"])

    random_state = config.get("random_state")
    train_dataset, val_dataset = prepare_training_data(
        dataset,
        processor,
        train_split,
        val_split,
        random_state,
        config.get("test_size"),
    )

    model = get_model_with_dataset_labels(config["model_name"], train_dataset)

    training_args = config.get("training_args", {})
    training_args["seed"] = random_state

    wandb_keys = ["wandb_entity", "wandb_project", "wandb_log_model", "wandb_name"]
    for key in wandb_keys:
        if key in config:
            training_args["report_to"] = "wandb"
            os.environ[key.upper()] = config[key]

    if training_args.get("report_to") == "wandb" and "wandb_name" not in config:
        os.environ["WANDB_NAME"] = f"{config['dataset_name']}_{config['model_name']}"

    wandb.login()

    training_args = TrainingArguments(**training_args)

    train(model, train_dataset, val_dataset, training_args)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model given a training config."
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    run_config(config)


if __name__ == "__main__":
    main()
