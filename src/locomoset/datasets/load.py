import datasets


def load_dataset(
    dataset_name: str,
    split: str | None = None,
    cache_dir: str | None = None,
    image_field: str = "image",
    label_field: str = "label",
) -> datasets.Dataset | datasets.DatasetDict:
    """Loads a dataset (split) then renames and selects the appropriate columns that
    contain the images and the labels.

    Args:
        dataset_name: Name or path of the dataset to load.
        split: Name of the dataset split to load (or None to load all splits).
        cache_dir: Path to the cache directory.
        image_field: Name of the column that contains the images.
        label_field: Name of the column that contains the labels.
    Returns:
        HuggingFace Dataset (if a split was defined) or DatasetDict (if no split was
            defined).
    """
    dataset = datasets.load_dataset(
        dataset_name, split=split, cache_dir=cache_dir, keep_in_memory=True
    )
    if image_field != "image":
        dataset = dataset.rename_column(image_field, "image")
    if label_field != "label":
        dataset = dataset.rename_column(label_field, "label")
    return dataset.select_columns(["image", "label"])
