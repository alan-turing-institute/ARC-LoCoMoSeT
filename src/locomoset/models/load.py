from typing import Optional

from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel


def get_model_and_processor(
    model_name: str, num_labels: Optional[int] = None
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    """
    Load a model and its corresponding preprocessor.

    Parameters
    ----------
    model_name : str
        Name or path to model and preprocessor to load
    num_labels : int or None :
        Replace model head with a layer for this many labels. If set to zero return the
        model without its classification head.

    Returns
    -------
    PreTrainedModel, BaseImageProcessor
        Model and preprocessor
    """
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor
