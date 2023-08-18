"""
Helper functions for extracting model features on a dataset.
"""
import torch
from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines.pt_utils import KeyDataset

from locomoset.models.load import get_model_and_processor
from locomoset.models.pipeline import ImageFeaturesPipeline


def get_features(
    dataset: Dataset | IterableDataset,
    processor: BaseImageProcessor,
    model_head: PreTrainedModel,
    batch_size: int = 4,
    n_images: int | None = None,
) -> torch.tensor:
    """Takes preprocessed image data and calls a model with its classification head
    removed to generate features.

    Args:
        dataset: Preprocessed images compatible with model_head and including
            'pixel_values' as a key.
        processor: Image preprocessor compatible with model_head.
        model_head: A model with its classification head removed. Takes pixel_values as
            input and outputs logits.
        batch_size: No. of images passed to model_head at once.
        n_images: If set, only extract features for the first n_images images. Otherwise
            get features for all the images.

    Returns:
        Extracted model features.
    """
    # Note the conversion to a KeyDataset is important here:
    #  - KeyDataset inherits from torch.utils.data.Dataset. The HuggingFace
    #    datasets.Dataset class does not.
    #  - Pipelines expect a torch.utils.data.Dataset as input to enable batched
    #    processing.
    #  - Pipelines generally expect only to be given the relevant input (image in this
    #    case), not the whole sample with labels, metadata etc. This is what a
    #    KeyDataset provides when iterated over.
    if isinstance(dataset, Dataset):
        n_images = n_images or len(dataset)  # used for progress bar
    dataset = KeyDataset(dataset, "image")

    pipeline = ImageFeaturesPipeline(model=model_head, image_processor=processor)
    pipe_iter = pipeline(dataset, batch_size=batch_size)

    features = []
    for idx, img_features in enumerate(tqdm(pipe_iter, total=n_images)):
        if n_images is not None and idx == n_images:
            break
        features.append(img_features)

    features = torch.concatenate(features)

    return features


if __name__ == "__main__":
    dataset = load_dataset("pcuenq/oxford-pets", split="train")
    model, processor = get_model_and_processor(
        "facebook/deit-tiny-patch16-224", num_labels=0
    )
    feats = get_features(dataset, processor, model, n_images=100)
    print(feats.shape)
