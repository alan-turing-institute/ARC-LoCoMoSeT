"""
    Test functions for the training module (src/locomoset/models/train.py)
"""

from copy import deepcopy

import numpy as np
import torch
from transformers import EvalPrediction, TrainingArguments

from locomoset.datasets.preprocess import prepare_training_data
from locomoset.models.load import get_model_with_dataset_labels
from locomoset.models.train import FineTuningConfig, get_metrics_fn, run_config, train


def test_get_metrics_fn():
    metric_fn = get_metrics_fn("accuracy")
    eval_pred = EvalPrediction(
        predictions=np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),
        label_ids=np.array([0, 0, 1, 1]),
    )
    result = metric_fn(eval_pred)
    assert result["accuracy"] == 0.5


def test_train(dummy_model_name, dummy_dataset, dummy_processor):
    train_dataset, val_dataset = prepare_training_data(dummy_dataset, dummy_processor)
    model = get_model_with_dataset_labels(dummy_model_name, train_dataset)
    training_args = TrainingArguments(
        output_dir="tmp",
        num_train_epochs=1,
        evaluation_strategy="epoch",
        report_to="none",
    )
    trainer = train(deepcopy(model), train_dataset, val_dataset, training_args)
    assert not torch.equal(
        trainer.model.classifier.weight.cpu(), model.classifier.weight.cpu()
    )


def test_run_config(dummy_fine_tuning_config):
    # remove wandb args for purpose of this test
    dummy_fine_tuning_config.pop("wandb_args")

    dummy_config = FineTuningConfig.from_dict(dummy_fine_tuning_config)
    trainer = run_config(dummy_config)
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] > 0
    assert metrics["epoch"] == 1
