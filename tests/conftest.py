from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image
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
    return "dummy_split"


@pytest.fixture
def dummy_dataset_name():
    return "dummy_dataset"


@pytest.fixture
def dummy_config(test_n_samples, test_seed, dummy_split, dummy_dataset_name):
    return {
        "model_name": "dummy_model",
        "dataset_name": dummy_dataset_name,
        "dataset_split": dummy_split,
        "metric": "dummy_metric",
        "n_samples": test_n_samples,
        "random_state": test_seed,
    }


@pytest.fixture
def dummy_dataset(rng, dummy_labels, dummy_split, test_n_samples):
    img_width = 10
    img_height = 10
    img_channels = 3
    img_bits = 8
    img_array = rng.integers(
        0, 2**img_bits, (test_n_samples, img_width, img_height, img_channels)
    )
    img_PIL = [Image.fromarray(img, mode="RGB") for img in img_array]
    return Dataset.from_dict(
        mapping={
            "image": img_PIL,
            "label": dummy_labels,
        },
        split=dummy_split,
    )


@pytest.fixture(autouse=True)
def patch_load_dataset(monkeypatch, dummy_dataset, dummy_split, dummy_dataset_name):
    def mock_load_dataset(path, split=None, *args, **kwargs):
        if path != dummy_dataset_name or split != dummy_split:
            raise ValueError(
                f"Attempted to run mock_load_dataset with {path=}, {split=}. "
                f"Run tests with path={dummy_dataset_name}, split={dummy_split}"
            )

        return dummy_dataset

    with patch("datasets.load_dataset", new=mock_load_dataset):
        yield None
