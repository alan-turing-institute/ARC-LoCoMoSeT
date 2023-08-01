"""
Helper functions for extracting model features on a dataset.
"""
import torch
from datasets import Dataset
from transformers.modeling_utils import PreTrainedModel


def get_features(
    processed_data: Dataset, model_head: PreTrainedModel, batch_size: int = 4
) -> torch.tensor:
    """Takes preprocessed image data and calls a model with its classification head
    removed to generate features.

    Args:
        processed_data: Preprocessed images compatible with model_head and including
            'pixel_values' as a key.
        model_head: A model with its classification head removed. Takes pixel_values as
            input and outputs logits.
        batch_size: No. of images passed to model_head at once.

    Returns:
        Extracted model features.
    """
    return processed_data.map(
        lambda image: model_head(image["pixel_values"]),
        batched=True,
        batch_size=batch_size,
    )["logits"]
