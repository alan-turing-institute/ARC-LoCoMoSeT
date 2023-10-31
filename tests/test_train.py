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


def test_init_fine_tuning_config():
    dummy_config = {
        "model_name": "test_model",
        "dataset_name": "test_dataset",
        "random_state": 42,
        "dataset_args": {"train_split": "test_split"},
        "training_args": {"output_dir": "tmp", "num_train_epochs": 1},
        "wandb": {"entity": "test_entity", "project": "test_project"},
    }
    config = FineTuningConfig.from_dict(dummy_config)
    assert config.run_name == "test_dataset_test_model"
    assert isinstance(config.get_training_args(), TrainingArguments)
    assert config.use_wandb is True


def test_run_config(dummy_model_name, dummy_dataset_name):
    dummy_config = FineTuningConfig.from_dict(
        {
            "model_name": dummy_model_name,
            "dataset_name": dummy_dataset_name,
            "random_state": 42,
            "training_args": {
                "output_dir": "tmp",
                "num_train_epochs": 1,
                "save_strategy": "no",
                "evaluation_strategy": "epoch",
                "report_to": "none",
            },
        }
    )
    trainer = run_config(dummy_config)
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] > 0
    assert metrics["epoch"] == 1
