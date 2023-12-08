from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoImageProcessor, AutoModelForImageClassification


@pytest.fixture
def test_n_features():
    return 5


@pytest.fixture
def test_n_samples():
    return 1000


@pytest.fixture
def test_seed():
    return 42


@pytest.fixture
def rng(test_seed):
    return np.random.default_rng(test_seed)


@pytest.fixture
def dummy_n_classes():
    return 3


@pytest.fixture
def dummy_labels(rng, dummy_n_classes, test_n_samples):
    return rng.integers(0, dummy_n_classes, test_n_samples)


@pytest.fixture
def dummy_features_random(rng, test_n_samples, test_n_features):
    return rng.normal(size=(test_n_samples, test_n_features))


@pytest.fixture
def dummy_features_perfect(dummy_labels, test_n_samples):
    """
    Returns features perfectly correlated with the dummy_labels (one-hot encoded
    labels).
    """
    return OneHotEncoder(sparse_output=False).fit_transform(
        dummy_labels.reshape((test_n_samples, 1))
    )


@pytest.fixture
def dummy_split():
    return "train"


@pytest.fixture
def dummy_dataset_name():
    return str(Path(__file__, "..", "data", "dummy_dataset").resolve())


@pytest.fixture
def dummy_model_name():
    return str(Path(__file__, "..", "data", "dummy_model").resolve())


@pytest.fixture
def dummy_dataset(dummy_dataset_name, dummy_split):
    return load_dataset(dummy_dataset_name, split=dummy_split)


@pytest.fixture
def dummy_processor(dummy_model_name):
    return AutoImageProcessor.from_pretrained(dummy_model_name)


@pytest.fixture
def dummy_model_head(dummy_model_name):
    return AutoModelForImageClassification.from_pretrained(
        dummy_model_name, num_labels=0
    )


@pytest.fixture
def dummy_model_classifier(dummy_model_name):
    return AutoModelForImageClassification.from_pretrained(dummy_model_name)


@pytest.fixture
def dummy_config_gen_dtime():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


@pytest.fixture
def dummy_metric_config(
    test_seed, dummy_split, dummy_dataset_name, dummy_model_name, dummy_config_gen_dtime
):
    return {
        "caches": {
            "datasets": "./.cache/huggingface/datasets",
            "models": "./.cache/huggingface/models",
        },
        "config_gen_dtime": dummy_config_gen_dtime,
        "model_name": dummy_model_name,
        "dataset_name": dummy_dataset_name,
        "dataset_args": {
            "train_split": "train",
            "val_split": "val_split",
            "test_split": "test",
            "val_size": 0.15,
            "test_size": 0.15,
        },
        "keep_labels": None,
        "keep_size": None,
        "local_save": False,
        "save_dir": "results",
        "metrics": ["renggli"],
        "n_samples": 50,
        "random_state": test_seed,
        "wandb_args": {"entity": "test_entity", "project": "test_project"},
    }


@pytest.fixture
def dummy_fine_tuning_config(
    test_seed, dummy_dataset_name, dummy_model_name, dummy_config_gen_dtime
):
    return {
        "caches": {
            "datasets": "./.cache/huggingface/datasets",
            "models": "./.cache/huggingface/models",
        },
        "config_gen_dtime": dummy_config_gen_dtime,
        "model_name": dummy_model_name,
        "dataset_name": dummy_dataset_name,
        "random_state": test_seed,
        "dataset_args": {"train_split": "train"},
        "keep_labels": None,
        "keep_size": None,
        "training_args": {
            "output_dir": "tmp",
            "num_train_epochs": 1,
            "save_strategy": "no",
            "evaluation_strategy": "epoch",
            "report_to": "none",
        },
        "wandb_args": {"entity": "test_entity", "project": "test_project"},
    }


@pytest.fixture
def dummy_validation_scores(rng, test_n_samples):
    # Sort here to induce uncorrelation with dummy_random_metric_scores
    return np.sort(rng.uniform(0, 1, test_n_samples)) * 100


@pytest.fixture
def dummy_perfect_metric_scores(dummy_validation_scores):
    return dummy_validation_scores


@pytest.fixture
def dummy_random_metric_scores(rng, test_n_samples):
    return rng.uniform(0, 1, test_n_samples) * 100


@pytest.fixture
def dummy_jinja_editable_file():
    return str(
        Path(__file__, "..", "data", "dummy_jinja_editable_script.yaml").resolve()
    )


@pytest.fixture
def dummy_jinja_env_location():
    return str(Path("..", "data").resolve())


@pytest.fixture
def dummy_top_level_config(
    dummy_model_name,
    dummy_dataset_name,
    test_seed,
    test_n_samples,
    dummy_config_gen_dtime,
):
    return {
        "config_type": "both",
        "config_dir": "tmp",
        "config_gen_dtime": dummy_config_gen_dtime,
        "save_dir": "tmp",
        "local_save": False,
        "slurm_template_path": "./tests/data/",
        "slurm_template_name": "dummy_jinja_editable_script.yaml",
        "models": dummy_model_name,
        "dataset_name": dummy_dataset_name,
        "random_states": [test_seed, test_seed + 1],
        "wandb_args": {"entity": "test_entity", "project": "test_project"},
        "use_bask": True,
        "bask": {
            "metrics": {
                "job_name": "locomoset_metric_experiment",
                "walltime": "0-0:30:0",
                "node_number": 1,
                "gpu_number": 1,
                "cpu_per_gpu": 36,
            },
            "train": {
                "job_name": "locomoset_train_experiment",
                "walltime": "0-0:30:0",
                "node_number": 1,
                "gpu_number": 1,
                "cpu_per_gpu": 36,
            },
        },
        "caches": {
            "datasets": "./.cache/huggingface/datasets",
            "models": "./.cache/huggingface/models",
        },
        "dataset_args": {
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "val_size": 0.15,
            "test_size": 0.15,
        },
        "keep_labels": None,
        "keep_sizes": None,
        "training_args": {
            "training_args": {
                "output_dir": "tmp",
                "num_train_epochs": 1,
                "save_strategy": "no",
                "evaluation_strategy": "epoch",
                "report_to": "none",
            },
        },
        "metrics": ["renggli", "n_pars", "LogME"],
        "dataset_split": "train",
        "n_samples": test_n_samples,
        "inference_args": {"device": "cuda"},
    }
