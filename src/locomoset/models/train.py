from typing import Callable

import evaluate
import numpy as np
import wandb
from datasets import Dataset, disable_caching
from transformers import EvalPrediction, PreTrainedModel, Trainer, TrainingArguments

from locomoset.datasets.load import load_dataset
from locomoset.datasets.preprocess import (
    drop_images,
    drop_images_by_labels,
    prepare_training_data,
)
from locomoset.models.classes import FineTuningConfig
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


def run_config(config: FineTuningConfig) -> Trainer:
    """Run the fine-tuning job specified by a FineTuningConfig: Load a dataset and
    model, preprocess it, train the model, and save the results.

    Args:
        config: A FineTuningConfig object.

    Returns:
        Trainer object.
    """
    if config.use_wandb:
        config.init_wandb()

    if config.caches.get("preprocess_cache") == "tmp":
        disable_caching()

    processor = get_processor(config.model_name, cache=config.caches["datasets"])

    train_split = config.dataset_args["train_split"]
    val_split = config.dataset_args.get("val_split", None)
    image_field = config.dataset_args.get("image_field", "image")
    label_field = config.dataset_args.get("label_field", "label")
    keep_in_memory = config.caches.get("preprocess_cache") == "ram"
    if val_split is None or val_split == train_split:
        dataset = load_dataset(
            config.dataset_name,
            split=train_split,
            cache_dir=config.caches["datasets"],
            keep_in_memory=keep_in_memory,
            image_field=image_field,
            label_field=label_field,
        )
    else:
        dataset = load_dataset(
            config.dataset_name,
            cache_dir=config.caches["datasets"],
            keep_in_memory=keep_in_memory,
            image_field=image_field,
            label_field=label_field,
        )

    #
    if config["drop_obs"] is not None:
        dataset = drop_images(dataset, config["drop_obs"], config["random_state"])

    if config["label_set"] is not None:
        dataset = drop_images_by_labels(dataset, config["label_set"])

    train_dataset, val_dataset = prepare_training_data(
        dataset,
        processor,
        train_split,
        val_split,
        config.random_state,
        config.dataset_args.get("test_size"),
    )
    del dataset

    model = get_model_with_dataset_labels(
        config.model_name, train_dataset, cache=config.caches["models"]
    )

    return train(model, train_dataset, val_dataset, config.get_training_args())
