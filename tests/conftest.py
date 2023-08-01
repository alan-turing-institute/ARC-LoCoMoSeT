from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder


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
def dummy_config(
    test_n_samples, test_seed, dummy_split, dummy_dataset_name, dummy_model_name
):
    return {
        "model_name": dummy_model_name,
        "dataset_name": dummy_dataset_name,
        "dataset_split": dummy_split,
        "metric": "dummy_metric",
        "n_samples": test_n_samples,
        "random_state": test_seed,
    }
