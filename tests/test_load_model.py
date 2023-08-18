"""
Tests for functions that load or process data/models.
"""
from torch.nn import Identity
from transformers import PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from locomoset.models.load import get_model_and_processor


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
