"""
Helper functions for loading models and preprocessors.
"""
from typing import Optional

from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel


def get_model_and_processor(
    model_name: str, num_labels: Optional[int] = None
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    """Load a model and its corresponding preprocessor.

    Args:
        model_name: Name or path to model and preprocessor to load.
        num_labels: If set, Replace model head with a layer for this many labels. If
            zero return the model without its classification head.

    Returns:
        Model and preprocessor.
    """
    if num_labels is not None:
        model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(model_name)

    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor
