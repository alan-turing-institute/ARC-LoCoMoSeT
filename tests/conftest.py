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
def dummy_config(test_seed, dummy_split, dummy_dataset_name, dummy_model_name):
    return {
        "model_name": dummy_model_name,
        "dataset_name": dummy_dataset_name,
        "dataset_split": dummy_split,
        "metric": "renggli",
        "n_samples": 50,
        "random_state": test_seed,
    }


@pytest.fixture
def dummy_validation_scores(rng):
    return np.sort(rng.uniform(0, 1, 10)) * 100
