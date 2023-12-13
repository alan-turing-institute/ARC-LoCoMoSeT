import os
import tempfile
from typing import Callable

import evaluate
import numpy as np
import wandb
from datasets import Dataset, disable_caching
from transformers import EvalPrediction, PreTrainedModel, Trainer, TrainingArguments

from locomoset.datasets.load import load_dataset
from locomoset.datasets.preprocess import (
    apply_dataset_mutations,
    create_data_splits,
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
    test_dataset: Dataset | None = None,
) -> Trainer:
    """Train a model on a dataset and evaluate it on a validation set and save the
    results either locally or to wandb.

    Args:
        model: HuggingFace model to train. Should have been loaded with the number of
            labels and the class labels set to be compatible with the training dataset.
        train_dataset: Preprocessed dataset to train on.
        val_dataset: Preprocessed dataset to evaluate on.
        test_dataset: Preprocessed dataset to test on.
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
    val_metrics = trainer.evaluate()
    trainer.save_metrics("eval", val_metrics)
    trainer.log_metrics("eval", val_metrics)

    if test_dataset is not None:
        test_metrics = trainer.evaluate(test_dataset)
        trainer.save_metrics("test", test_metrics)
        trainer.log_metrics("test", test_metrics)

    if "wandb" not in training_args.report_to or wandb.run is None:
        # save results locally
        trainer.save_model()

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

    if "tmp_dir" in config.caches and config.caches["tmp_dir"] is not None:
        # This is a workaround for overwriting the default path for tmp dirs,
        # see https://github.com/alan-turing-institute/ARC-LoCoMoSeT/issues/93
        os.environ["TMPDIR"] = config.caches["tmp_dir"]
        tempfile.tempdir = config.caches["tmp_dir"]

    processor = get_processor(config.model_name, cache=config.caches["datasets"])

    keep_in_memory = config.caches.get("preprocess_cache") == "ram"

    # Load Dataset
    dataset = load_dataset(
        config.dataset_name,
        cache_dir=config.caches["datasets"],
        keep_in_memory=keep_in_memory,
        image_field=config.dataset_args["image_field"],
        label_field=config.dataset_args["label_field"],
    )

    # Prepare splits
    dataset = create_data_splits(
        dataset,
        train_split=config.dataset_args["train_split"],
        val_split=config.dataset_args["val_split"],
        test_split=config.dataset_args["test_split"],
        random_state=config.random_state,
        val_size=config.dataset_args["val_size"],
        test_size=config.dataset_args["test_size"],
    )

    # Mutate dataset
    train_and_val = apply_dataset_mutations(
        dataset,
        keep_labels=config.dataset_args["keep_labels"],
        keep_size=config.dataset_args["keep_size"],
        seed=config.random_state,
    )

    # Prepare train and test data
    train_dataset, val_dataset, test_dataset = prepare_training_data(
        dataset,
        processor,
        train_split=config.dataset_args["train_split"],
        val_split=config.dataset_args["val_split"],
        test_split=config.dataset_args["test_split"],
        keep_in_memory=keep_in_memory,
        writer_batch_size=config.caches.get("writer_batch_size", 1000),
    )
    del dataset
    del train_and_val

    model = get_model_with_dataset_labels(
        config.model_name, train_dataset, cache=config.caches["models"]
    )

    return train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=config.get_training_args(),
        test_dataset=test_dataset,
    )
