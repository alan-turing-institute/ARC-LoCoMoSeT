"""
Helper functions for preprocessing datasets.
"""
from datasets import Dataset
from transformers.image_processing_utils import BaseImageProcessor


def preprocess(dataset: Dataset, processor: BaseImageProcessor) -> Dataset:
    """Convert an image dataset to RGB and run it through a pre-processor for
    compatibility with a model.

    Args:
        dataset: HuggingFace image dataset to process. Each samples in the dataset is
            expected to have a key 'image' containing a PIL image that will be converted
            to RGB format and run through the processor. The processed result is saved
            under the key "pixel_values" and the "image" key is removed.
        processor: HuggingFace trained pre-processor to use.

    Returns:
        Processed dataset with feature 'pixel_values' instead of 'image'.
    """

    def proc_sample(sample: dict) -> dict:
        """Process one sample: Convert image to RGB and run through processor

        Args:
            sample: Sample from dataset with key 'image' containing a PIL image.

        Returns:
            Processed sample with feature 'pixel_values' (a tensor of pixel values
            representing the processed image) and the 'image' feature deleted.
        """
        sample["pixel_values"] = processor(sample["image"].convert("RGB"))[
            "pixel_values"
        ][0]
        return sample

    processed_dataset = dataset.map(proc_sample, batched=False, remove_columns="image")
    return processed_dataset.with_format("torch")
