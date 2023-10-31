"""
Helper functions for loading models and preprocessors.
"""
from datasets import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel


def get_processor(model_name: str) -> BaseImageProcessor:
    """Load a model's image preprocessor.

    Args:
        model_name: Name or path of model to load preprocessor for.

    Returns:
        Image processor.
    """
    return AutoImageProcessor.from_pretrained(model_name)


def get_model_without_head(model_name: str) -> PreTrainedModel:
    """Load a model with its classification head removed (or, more precisely, replaced
    with torch.nn.Identity).

    Args:
        model_name: Name or path to model to load.

    Returns:
        Model without its classification head.
    """
    return AutoModelForImageClassification.from_pretrained(model_name, num_labels=0)


def get_model_with_dataset_labels(model_name: str, dataset: Dataset) -> PreTrainedModel:
    """Load a model with its classification head replaced with a new one with the
    correct number of labels.

    Args:
        model_name: Name or path to model to load.
        dataset: Dataset to use to get the number of classes and class index to label
            mappings. Must contain a "label" feature that is an instance of
            datasets.ClassLabel.

    Returns:
        Model with its classification head replaced with one with the correct number of
        classes for the dataset, and the correct class index to label mappings.
    """
    return AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=dataset.features["label"].num_classes,
        id2label=dataset.features["label"]._int2str,
        label2id=dataset.features["label"]._str2int,
        ignore_mismatched_sizes=True,
    )
