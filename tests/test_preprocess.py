from collections import Counter
from math import isclose

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets

from locomoset.datasets.preprocess import (
    _encode_labels_dict,
    _encode_labels_single,
    apply_dataset_mutations,
    create_data_splits,
    drop_images,
    drop_images_by_labels,
    encode_labels,
    preprocess,
    preprocess_dataset_splits,
)


def test_preprocess(dummy_dataset, dummy_processor):
    """
    Test the preprocess function correctly converts images in the dummy dataset
    """
    processed_dummy_data = preprocess(dummy_dataset, dummy_processor)
    assert "pixel_values" in processed_dummy_data.features
    assert len(processed_dummy_data) == len(dummy_dataset)

    # check first image matches expected output
    exp_first_image = dummy_processor(dummy_dataset[0]["image"].convert("RGB"))[
        "pixel_values"
    ][0]
    np.testing.assert_equal(processed_dummy_data[0]["pixel_values"], exp_first_image)


def test_encode_labels_single(dummy_dataset):
    """
    Test the encode_labels_single function correctly encodes labels
    """
    # if dataset labels are already a ClassLabel, return the dataset unchanged
    assert _encode_labels_single(dummy_dataset) == dummy_dataset

    # if labels are strings and no class labels given, automatically create an encoding
    test_data = Dataset.from_dict({"label": ["a", "b", "c", "a"]})
    enc_data = _encode_labels_single(test_data)
    assert enc_data["label"] == [0, 1, 2, 0]
    assert enc_data.features["label"].names == ["a", "b", "c"]

    # if a ClassLabel encoding is provided, use it
    test_labels = ClassLabel(names=["c", "b", "a"])
    enc_data = _encode_labels_single(test_data, test_labels)
    assert enc_data["label"] == [2, 1, 0, 2]
    assert enc_data.features["label"].names == ["c", "b", "a"]


def test_encode_labels_dict():
    # if split to use for encoding is not specified, default to using the "train" split
    # first, if available
    test_data_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"label": ["a", "b", "c", "a"]}),
            "val": Dataset.from_dict({"label": ["b", "c"]}),
        }
    )
    test_data_dict = _encode_labels_dict(test_data_dict)
    assert test_data_dict["train"].features["label"].names == ["a", "b", "c"]
    assert test_data_dict["val"].features["label"].names == ["a", "b", "c"]

    # if split to use for encoding is not specified, and the splits have non-standard
    # names, default to using the split with the most samples for encoding
    test_data_dict = DatasetDict(
        {
            "short": Dataset.from_dict({"label": ["b", "c"]}),
            "long": Dataset.from_dict({"label": ["a", "b", "c", "a"]}),
        }
    )
    test_data_dict = _encode_labels_dict(test_data_dict)
    assert test_data_dict["long"].features["label"].names == ["a", "b", "c"]
    assert test_data_dict["short"].features["label"].names == ["a", "b", "c"]

    # if split is specified, use that split to define the class label encoding
    test_data_dict = DatasetDict(
        {
            "split1": Dataset.from_dict({"label": ["a", "b", "c"]}),
            "split2": Dataset.from_dict({"label": ["a", "b", "b", "b"]}),
        }
    )
    test_data_dict = _encode_labels_dict(test_data_dict, encoding_split="split1")
    assert test_data_dict["split1"].features["label"].names == ["a", "b", "c"]
    assert test_data_dict["split2"].features["label"].names == ["a", "b", "c"]


def test_drop_images(dummy_dataset):
    p = 0.5
    new_dataset = drop_images(dummy_dataset, keep_size=p)
    assert new_dataset.num_rows == (dummy_dataset.num_rows) * p


def test_drop_images_label_stratification(dummy_dataset):
    p = 0.5
    new_dataset = drop_images(dummy_dataset, keep_size=p)
    old_counts = Counter(dummy_dataset["label"])
    new_counts = Counter(new_dataset["label"])
    checks = [
        isclose(new_counts[i], old_counts[i] * p, abs_tol=1) for i in list(range(3))
    ]
    assert all(checks)


def test_drop_images_by_labels(dummy_dataset):
    keeps = [0, 1]
    new_dataset = drop_images_by_labels(dummy_dataset, keep_labels=keeps)
    assert all([x in new_dataset["label"] for x in keeps])
    assert 2 not in new_dataset["label"]


def test_apply_dataset_mutations(dummy_dataset):
    s = 40
    keeps = [0, 2]
    new_dataset = apply_dataset_mutations(dummy_dataset, keep_labels=keeps, keep_size=s)
    assert new_dataset.num_rows == s
    assert all([x in new_dataset["label"] for x in keeps])
    assert 1 not in new_dataset["label"]


def test_encode_labels():
    test_data = Dataset.from_dict({"label": ["a", "b", "c", "a"]})
    encoded_a = encode_labels(test_data)
    encoded_b = _encode_labels_single(test_data)
    assert encoded_a["label"] == encoded_b["label"]
    assert encoded_a.features == encoded_b.features

    test_data_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"label": ["a", "b", "c"]}),
            "val": Dataset.from_dict({"label": ["a", "b", "b", "b"]}),
        }
    )
    assert encode_labels(test_data_dict) == _encode_labels_dict(test_data_dict)


def _test_data_split(data_dict, train_split, val_split, test_split, size, n):
    assert data_dict[train_split].num_rows == (n - size * 2)
    assert data_dict[val_split].num_rows == size
    assert data_dict[test_split].num_rows == size


def test_create_data_splits(dummy_dataset):
    # Constants
    train_split = "train"
    val_split = "val"
    test_split = "test"
    p = 0.15
    size = 15
    n = dummy_dataset.num_rows
    seed = 42

    # Scenario 1: one dataset into 3
    data_dict = create_data_splits(
        dummy_dataset,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        val_size=p,
        test_size=p,
        random_state=42,
    )

    _test_data_split(data_dict, train_split, val_split, test_split, size, n)

    # Scenario 2a: data dict of 2 datasets into 3
    data_dict = DatasetDict(
        {
            train_split: concatenate_datasets(
                [data_dict[train_split], data_dict[val_split]]
            ),
            test_split: data_dict[test_split],
        }
    )

    data_dict = create_data_splits(
        data_dict,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        val_size=p,
        test_size=None,
        random_state=seed,
    )

    _test_data_split(data_dict, train_split, val_split, test_split, size, n)

    # Scenario 2b: data dict of 2 datasets into 3
    data_dict = DatasetDict(
        {
            train_split: concatenate_datasets(
                [data_dict[train_split], data_dict[test_split]]
            ),
            val_split: data_dict[val_split],
        }
    )

    data_dict = create_data_splits(
        data_dict,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        val_size=None,
        test_size=p,
        random_state=seed,
    )

    _test_data_split(data_dict, train_split, val_split, test_split, size, n)


def test_preprocess_dataset_splits(dummy_dataset, dummy_processor):
    dataset_dict = dummy_dataset.train_test_split(test_size=5)
    train_dataset, val_dataset = preprocess_dataset_splits(
        dataset_dict,
        dummy_processor,
        train_split="train",
        val_split="test",
    )
    assert val_dataset.num_rows == 5
    assert train_dataset.num_rows == dummy_dataset.num_rows - 5
    assert "pixel_values" in train_dataset.features
    assert "pixel_values" in val_dataset.features
