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
def dummy_streamed_dataset(dummy_dataset_name, dummy_split):
    return load_dataset(
        dummy_dataset_name, split=dummy_split, token=True, streaming=True
    )


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
def dummy_one_hot_perfect():
    vec = np.zeros(200)
    vec[103] = 1.0
    return vec


@pytest.fixture
def dummy_one_hot_small():
    vec = np.zeros(200)
    vec[103] = 1.0
    return vec


@pytest.fixture
def dummy_one_hot_large():
    vec = np.zeros(260)
    vec[103] = 1.0
    return vec


@pytest.fixture
def dummy_one_hot_extra_large():
    vec = np.zeros(512)
    vec[260] = 1.0
    return vec


@pytest.fixture
def dummy_label():
    return 91.0


@pytest.fixture
def dummy_diag_hom0_weighted():
    num_examples = 3
    num_hom0_feats = 6
    num_hom1_feats = 4
    hom0 = np.zeros((num_examples, num_hom0_feats, 3))
    hom1 = np.zeros((num_examples, num_hom1_feats, 3))
    hom1[:, :, 2] = 1.0
    return np.concatenate((hom0, hom1), axis=1)


@pytest.fixture
def dummy_diag_hom1_weighted():
    num_examples = 3
    num_hom0_feats = 4
    num_hom1_feats = 6
    hom0 = np.zeros((num_examples, num_hom0_feats, 3))
    hom1 = np.zeros((num_examples, num_hom1_feats, 3))
    hom1[:, :, 2] = 1.0
    return np.concatenate((hom0, hom1), axis=1)
