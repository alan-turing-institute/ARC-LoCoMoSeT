import torch
from datasets import Dataset
from transformers.modeling_utils import PreTrainedModel


def get_features(
    processed_data: Dataset, model_head: PreTrainedModel, **map_kwargs
) -> torch.tensor:
    """Takes preprocessed image data and calls a model with its classification head
    removed to generate features.

    Parameters
    ----------
    processed_data : Dataset
        Preprocessed images compatible with model_head and including 'pixel_values' as
        a key
    model_head : PreTrainedModel
        A model with its classification head removed. Takes pixel_values as input and
        outputs logits.

    Returns
    -------
    torch.tensor
        Extracted model features
    """
    return processed_data.map(
        lambda image: model_head(image["pixel_values"]),
        **map_kwargs,
    )["logits"]
