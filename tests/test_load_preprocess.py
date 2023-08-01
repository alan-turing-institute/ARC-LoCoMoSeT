"""
Tests for functions that load or process data/models.
"""
import numpy as np
import pytest
from torch.nn import Identity
from transformers import PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from locomoset.datasets.preprocess import preprocess
from locomoset.models.features import get_features
from locomoset.models.load import get_model_and_processor


@pytest.fixture
def processed_dummy_data(dummy_dataset, dummy_processor):
    return preprocess(dummy_dataset, dummy_processor)


def test_get_model_and_processor(dummy_model_name):
    """
    Test a model and preprocessor are loaded correctly.
    """
    model, processor = get_model_and_processor(dummy_model_name)
    assert isinstance(model, PreTrainedModel)
    assert isinstance(processor, BaseImageProcessor)


def test_get_model_head(dummy_model_name):
    """
    Test setting num_labels=0 replaces the final classifier layer with Identity
    (which just passes through the values from the previous layer)
    """
    model, _ = get_model_and_processor(dummy_model_name, num_labels=0)
    assert isinstance(model.classifier, Identity)


def test_preprocess(processed_dummy_data, dummy_dataset, dummy_processor):
    """
    Test the preprocess function correctly converts images in the dummy dataset
    """
    assert "pixel_values" in processed_dummy_data.features
    assert len(processed_dummy_data) == len(dummy_dataset)

    # check first image matches expected output
    exp_first_image = dummy_processor(dummy_dataset[0]["image"].convert("RGB"))[
        "pixel_values"
    ][0]
    np.testing.assert_equal(processed_dummy_data[0]["pixel_values"], exp_first_image)


def test_get_features(processed_dummy_data, dummy_model_head):
    """
    Test features can be extracted from a model correctly
    """
    features = get_features(processed_dummy_data, dummy_model_head)
    assert features.shape == (
        len(processed_dummy_data),
        dummy_model_head.config.hidden_size,
    )
