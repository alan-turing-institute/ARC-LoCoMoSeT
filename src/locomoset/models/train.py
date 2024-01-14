import os
import tempfile
from time import time
from typing import Callable

import evaluate
import numpy as np
import torch
import wandb
from datasets import Dataset, disable_caching
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import EvalPrediction, PreTrainedModel, Trainer, TrainingArguments
from transformers.image_processing_utils import BaseImageProcessor

from locomoset.datasets.load import load_dataset
from locomoset.datasets.preprocess import (
    create_data_splits,
    drop_images,
    preprocess_dataset_splits,
)
from locomoset.models.classes import FineTuningConfig
from locomoset.models.features import get_features
from locomoset.models.load import (
    freeze_model,
    get_model_with_dataset_labels,
    get_model_without_head,
    get_processor,
    unfreeze_classifier,
)


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

    if test_dataset is not None:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    if "wandb" not in training_args.report_to or wandb.run is None:
        # save results locally
        trainer.save_model()
        val_metrics = trainer.evaluate()
        trainer.save_metrics("eval", val_metrics)
        trainer.log_metrics("eval", val_metrics)
        if test_dataset is not None:
            trainer.save_metrics("test", test_metrics)
            trainer.log_metrics("test", test_metrics)

    return trainer


def train_logistic(
    model_head: PreTrainedModel,
    processor: BaseImageProcessor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset | None = None,
    random_state: int | None = None,
    device: int | torch.device = "cuda",
) -> LogisticRegression:
    """Train and evaluate an sklearn LogisticRegression model trained on a frozen model
    with head removed, evaluate it and save the results to wandb.

    Args:
        model: HuggingFace model with its classification head removed.
        train_dataset: Preprocessed dataset to train on.
        val_dataset: Preprocessed dataset to evaluate on.
        test_dataset: Preprocessed dataset to test on.
        random_state: Random state to use for model fitting
        device: Device to run model_head on.

    Returns:
        Trainer object.
    """
    results = {}
    t = time()
    feats_train = get_features(
        dataset=train_dataset, processor=processor, model_head=model_head, device=device
    )
    results["train/feature_extraction_time"] = time() - t

    t = time()
    clf = Pipeline(
        (
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(random_state=random_state)),
        )
    )
    clf.fit(feats_train, train_dataset["label"])
    results["train/runtime"] = time() - t

    t = time()
    pred_train = clf.predict(feats_train)
    train_acc = accuracy_score(train_dataset["label"], pred_train)
    results["train/accuracy"] = train_acc
    results["train/accuracy_time"] = time() - t

    t = time()
    feats_val = get_features(
        dataset=val_dataset, processor=processor, model_head=model_head, device=device
    )
    results["eval/feature_extraction_time"] = time() - t
    t = time()
    pred_val = clf.predict(feats_val)
    val_acc = accuracy_score(val_dataset["label"], pred_val)
    results["eval/accuracy"] = val_acc
    results["eval/accuracy_time"] = time() - t

    t = time()
    feats_test = get_features(
        dataset=test_dataset, processor=processor, model_head=model_head, device=device
    )
    results["test/feature_extraction_time"] = time() - t
    t = time()
    pred_test = clf.predict(feats_test)
    test_acc = accuracy_score(test_dataset["label"], pred_test)
    results["test/accuracy"] = test_acc
    results["test/accuracy_time"] = time() - t

    wandb.log(results)
    wandb.finish()
    return clf, results


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
        keep_labels=config.dataset_args["keep_labels"],
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

    # Subset datasets:
    # train: down to requested n_samples
    dataset[config.dataset_args["train_split"]] = drop_images(
        dataset[config.dataset_args["train_split"]],
        keep_size=config.n_samples,
        seed=config.random_state,
    )
    if config.n_samples is None:
        config.n_samples = dataset[config.dataset_args["train_split"]].num_rows

    # val: down to 0.25 * n_samples or whole val dataset, whichever is smaller
    # also try to make sure that there are at least as many images as there are classes
    # in the val set, as required for stratified sampling
    n_classes = dataset[config.dataset_args["val_split"]].features["label"].num_classes
    dataset[config.dataset_args["val_split"]] = drop_images(
        dataset[config.dataset_args["val_split"]],
        keep_size=min(
            (
                max((round(0.25 * config.n_samples), n_classes)),
                dataset[config.dataset_args["val_split"]].num_rows,
            )
        ),
        seed=config.random_state,
    )

    if config.freeze_model == "logistic":
        model_head = get_model_without_head(config.model_name, config.caches["models"])
        return train_logistic(
            model_head=model_head,
            processor=processor,
            train_dataset=dataset[config.dataset_args["train_split"]],
            val_dataset=dataset[config.dataset_args["val_split"]],
            test_dataset=dataset[config.dataset_args["test_split"]],
            random_state=config.random_state,
        )

    else:
        # Prepare train and test data
        train_dataset, val_dataset, test_dataset = preprocess_dataset_splits(
            dataset,
            processor,
            train_split=config.dataset_args["train_split"],
            val_split=config.dataset_args["val_split"],
            test_split=config.dataset_args["test_split"],
            keep_in_memory=keep_in_memory,
            writer_batch_size=config.caches.get("writer_batch_size", 1000),
        )
        del dataset

        model = get_model_with_dataset_labels(
            config.model_name, train_dataset, cache=config.caches["models"]
        )

        if config.freeze_model:
            freeze_model(model)
            unfreeze_classifier(model)

        return train(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_args=config.get_training_args(),
            test_dataset=test_dataset,
        )
