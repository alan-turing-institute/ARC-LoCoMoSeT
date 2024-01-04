import datasets

from locomoset.datasets.preprocess import drop_images_by_labels, encode_labels


def load_dataset(
    dataset_name: str,
    split: str | None = None,
    cache_dir: str | None = None,
    image_field: str = "image",
    label_field: str = "label",
    keep_in_memory: bool | None = None,
    keep_labels: list[str] | list[int] | None = None,
) -> datasets.Dataset | datasets.DatasetDict:
    """Loads a dataset (split), renames and selects the appropriate columns that
    contain the images and the labels, and filters the dataset by labels to keep.

    Args:
        dataset_name: Name or path of the dataset to load.
        split: Name of the dataset split to load (or None to load all splits).
        cache_dir: Path to the cache directory.
        image_field: Name of the column that contains the images.
        label_field: Name of the column that contains the labels.
        keep_in_memory: Cache the dataset and any preprocessed files to RAM rather than
            disk if True.
        keep_labels: List of labels to keep (or None to keep all labels).
    Returns:
        HuggingFace Dataset (if a split was defined) or DatasetDict (if no split was
            defined) with fields "image" and "label" and only the labels to keep (if
            keep_labels was defined).
    """
    dataset = datasets.load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
    )

    # remove corrupted file in rvl_cdip
    if dataset_name == "aharley/rvl_cdip":
        if split == "test":
            dataset = dataset.select(
                [i for i in range(len(dataset["test"])) if i != 33669]
            )
        else:
            dataset["test"] = dataset["test"].select(
                [i for i in range(len(dataset["test"])) if i != 33669]
            )

    if image_field != "image":
        dataset = dataset.rename_column(image_field, "image")
    if label_field != "label":
        dataset = dataset.rename_column(label_field, "label")
    dataset = dataset.select_columns(["image", "label"])

    dataset = encode_labels(dataset)

    if keep_labels is not None:
        dataset = drop_images_by_labels(dataset, keep_labels)

    return dataset
