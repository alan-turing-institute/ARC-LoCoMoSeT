"""
Tests for functions that load or process data/models.
"""
import string

from torch.nn import Identity
from transformers import PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from locomoset.models.load import (
    get_model_with_dataset_labels,
    get_model_without_head,
    get_processor,
)


def test_get_processor(dummy_model_name):
    """
    Test a model and preprocessor are loaded correctly.
    """
    processor = get_processor(dummy_model_name)
    assert isinstance(processor, BaseImageProcessor)


def test_get_model_without_head(dummy_model_name):
    """
    Test setting num_labels=0 replaces the final classifier layer with Identity
    (which just passes through the values from the previous layer)
    """
    model = get_model_without_head(dummy_model_name)
    assert isinstance(model, PreTrainedModel)
    assert isinstance(model.classifier, Identity)


def test_get_model_with_dataset_labels(
    dummy_model_name, dummy_dataset, dummy_n_classes
):
    """
    Test loading a model and adapting it for use with a new dataset works correctly
    """
    model = get_model_with_dataset_labels(dummy_model_name, dummy_dataset)
    assert model.num_labels == dummy_n_classes
    assert model.config.id2label == list(string.ascii_lowercase)[:dummy_n_classes]
