from datasets import Dataset
from transformers.image_processing_utils import BaseImageProcessor


def preprocess(dataset: Dataset, processor: BaseImageProcessor) -> Dataset:
    """Convert an image dataset to RGB and run it through a pre-processor for
    compatibility with a model.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace image dataset to process
    processor : BaseImageProcessor
        HuggingFace trained pre-processor to use

    Returns
    -------
    Dataset
        Processed dataset with feature 'pixel_values' instead of 'image'
    """

    def proc_sample(sample):
        """Process one sample: Convert image to RGB and run through processor"""
        sample["pixel_values"] = processor(sample["image"].convert("RGB"))[
            "pixel_values"
        ][0]
        return sample

    processed_dataset = dataset.map(proc_sample, batched=False, remove_columns="image")
    return processed_dataset.with_format("torch")
