"""
Helper functions for preprocessing datasets.
"""
from datasets import ClassLabel, Dataset, DatasetDict
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import load_image


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
        sample["pixel_values"] = processor(load_image(sample["image"]))["pixel_values"][
            0
        ]
        return sample

    processed_dataset = dataset.map(proc_sample, batched=False, remove_columns="image")
    return processed_dataset.with_format("torch")


def encode_labels_single(
    dataset: Dataset, class_labels: ClassLabel | None = None
) -> Dataset:
    """Check if dataset labels are strings and encode them as ClassLabel if necessary.

    Args:
        dataset: HuggingFace dataset to check. Expected to have the key "label".
        class_labels: Optional ClassLabel object to use for encoding.

    Returns:
        Dataset with labels converted to datasets.ClassLabel if necessary.
    """
    # only attempt to encode string labels
    if dataset.features["label"].dtype != "string":
        return dataset

    if class_labels is not None:
        return dataset.cast_column("label", class_labels)

    return dataset.class_encode_column("label")


def encode_labels_dict(
    dataset_dict: DatasetDict, encoding_split: str | None = None
) -> DatasetDict:
    """Encode labels in a dataset dict.

    Args:
        dataset_dict: HuggingFace dataset dict to check. Each dataset split in the dict
            is Expected to have the key "label".
        encoding_split: Split to use for encoding. If None, the encoding will be based
            on the data in the "train" split, "validation" split, "val" split, or the
            splitwith the most samples, in that order of preference.

    Returns:
        DatasetDict with labels in each split converted to datasets.ClassLabel if
        necessary.
    """
    if encoding_split is None:
        # Decide which split to use as the base encoding. First: use either the train
        # or validation split, if present.
        preferred_splits = ["train", "validation", "val"]
        for split in preferred_splits:
            if split in dataset_dict:
                encoding_split = split
                break
        # If none of the splits in preferred_splits are present, use the split with the
        # most samples.
        if encoding_split is None:
            encoding_split = max(dataset_dict.num_rows, key=dataset_dict.num_rows.get)

    # Create the label encoding based on the encoding split.
    dataset_dict[encoding_split] = encode_labels_single(dataset_dict[encoding_split])

    # Apply the label encoding to all other splits.
    for split in dataset_dict:
        if split != encoding_split:
            dataset_dict[split] = encode_labels_single(
                dataset_dict[split],
                class_labels=dataset_dict[encoding_split].features["label"],
            )

    return dataset_dict


def encode_labels(
    dataset: Dataset | DatasetDict, encoding_split: str | None = None
) -> Dataset | DatasetDict:
    """Calls encode_labels_single or encode_labels_dict depending on the input type."""
    if isinstance(dataset, Dataset):
        return encode_labels_single(dataset)
    return encode_labels_dict(dataset, encoding_split=encoding_split)


def prepare_training_data(
    dataset: Dataset | DatasetDict,
    processor: BaseImageProcessor,
    train_split: str = "train",
    val_split: str = "validation",
    random_state: int | None = None,
    test_size: float | int | None = None,
) -> (Dataset, Dataset):
    """Preprocesses a dataset and splits it into train and validation sets.

    Args:
        dataset: HuggingFace Dataset or DatasetDict to process and split. Each Dataset
            split is expected to have 'image' and 'label' columns.
        processor: HuggingFace pre-trained image pre-processor to use.
        train_split: Name of the split to use for training. Only used if input is a
            DatasetDict with more than one split.
        val_split: Name of the split to use for validation. Only used if input is a
            DatasetDict with more than one split.
        random_state: Random state to use for the train/test split. Only used if the
            input is a single Dataset/single-entry DatasetDict.
        test_size: Size of test set (fraction of features and labels to exclude from
            training for evaluation).

    Returns:
        Tuple of preprocessed train and validation datasets.
    """
    dataset = encode_labels(dataset)

    if isinstance(dataset, DatasetDict) and len(dataset) == 1:
        dataset = dataset[list(dataset.keys())[0]]

    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split(
            stratify_by_column="label", seed=random_state, test_size=test_size
        )
        train_split = "train"
        val_split = "test"

    train_dataset = preprocess(dataset[train_split], processor)
    val_dataset = preprocess(dataset[val_split], processor)
    return train_dataset, val_dataset


def _drop_images(
    dataset: Dataset,
    test_size: float,
    seed: int | None = None,
) -> Dataset:
    """Randomly drops images from the dataset

    Args:
        dataset: HuggingFace Dataset to drop images from
        test_size: Size of test set (fraction of features and labels to exclude from
            training for evaluation).
        seed: Seed for dropping images

    Returns:
        Original Dataset with images dropped
    """
    return dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )["train"]


def _drop_labels(
    dataset: Dataset,
    labels_to_keep: list[str],
) -> Dataset:
    """Drops all images with labels different to those supplied

    Args:
        dataset: HuggingFace Dataset to drop labels from
        labels_to_keep: List of labels to keep in the Dataset or DatasetDict

    Returns:
        Original Dataset retaining all images with matching labels
    """
    return dataset.filter(lambda sample: sample["label"] in labels_to_keep)


def _mutate_dataset(
    dataset: Dataset | DatasetDict,
    fn: callable,
    **kwargs,
) -> Dataset | DatasetDict:
    """Applies a function to a HuggingFace Dataset or Datasets within a DatasetDict

    Args:
        dataset: HuggingFace Dataset or DatasetDict to apply function to
        fn: Function to apply to Dataset or Datasets within DatasetDict
        **kwargs: Arguments passed to fn

    Returns:
        Dataset or DatasetDict with fn applied
    """
    if isinstance(dataset, DatasetDict):
        return DatasetDict({key: fn(dataset[key], **kwargs) for key in dataset})

    if isinstance(dataset, Dataset):
        return fn(dataset, **kwargs)

    raise ValueError(f"dataset must be Dataset or DatasetDict, is {type(dataset)}")


def drop_images(
    dataset: Dataset | DatasetDict,
    test_size: float | int,
    seed: int | None = None,
) -> Dataset | DatasetDict:
    """Randomly drops images

    Args:
        dataset: HuggingFace Dataset or DatasetDict to drop images from
        test_size: Size of test set (fraction of features and labels to exclude from
            training for evaluation).
        seed: Seed for dropping images

    Returns:
        Original Dataset or DatasetDict with images dropped
    """
    return _mutate_dataset(dataset, _drop_images, test_size, seed)


def drop_labels(
    dataset: Dataset | DatasetDict,
    labels_to_keep: list[str],
) -> Dataset | DatasetDict:
    """Drops all images with labels different to those supplied

    Args:
        dataset: HuggingFace Dataset or DatasetDict to drop labels from
        labels_to_keep: List of labels to keep in the Dataset or DatasetDict

    Returns:
        Original Dataset or DatasetDict retaining all images with matching labels
    """
    return _mutate_dataset(dataset, _drop_labels, labels_to_keep)
