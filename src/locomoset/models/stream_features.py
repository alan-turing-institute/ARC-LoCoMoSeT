"""
Extract the features and one hot vectorised labels from a dataset streamed from
Huggingface for a given model head (a model without a classifier).
"""
import numpy as np
from datasets import Dataset
from transformers.modeling_utils import PreTrainedModel


def get_features_labels(
    processed_data: Dataset,
    model_head: PreTrainedModel,
) -> tuple(np.ndarray, np.ndarray):
    """Takes preprocessed image data and calls a model with its classification head
    removed to generate features and extract one hot vectors, returning a numpy array
    for the purposes of PARC.

    Args:
        processed_data: Preprocessed streaming dataset iterable, compatible with
            model_head and including 'pixel_values' as a key.
        model_head: A model with its classification head removed. Takes pixel_values as
            input and output logits.

    Returns:
        Extracted model features, and corresponding ground truth one hot vectors.
    """

    def pull_feats(data_iter):
        # Function for pulling the features out of the model
        data_iter["features"] = model_head(
            data_iter["pixel_values"][None, :, :, :]
        ).logits
        return data_iter

    feat_data = processed_data.map(
        pull_feats, batched=True, batch_size=16, remove_columns=["pixel_values"]
    )

    feats = []
    one_hots = []
    for data in feat_data:
        feats.append(data["features"].detach().numpy()[0])
        one_hots.append(data["label"])

    return np.asarray(feats), np.asarray(one_hots)
