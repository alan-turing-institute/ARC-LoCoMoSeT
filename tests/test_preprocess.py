import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict

from locomoset.datasets.preprocess import (
    _encode_labels_dict,
    _encode_labels_single,
    encode_labels,
    prepare_training_data,
    preprocess,
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


def test_prepare_training_data(dummy_dataset, dummy_processor):
    train_dataset, val_dataset = prepare_training_data(dummy_dataset, dummy_processor)
    assert train_dataset.num_rows == 0.75 * dummy_dataset.num_rows
    assert val_dataset.num_rows == 0.25 * dummy_dataset.num_rows
    assert "pixel_values" in train_dataset.features
    assert "pixel_values" in val_dataset.features

    dataset_dict = dummy_dataset.train_test_split(test_size=5)
    train_dataset, val_dataset = prepare_training_data(
        dataset_dict, dummy_processor, val_split="test"
    )
    assert val_dataset.num_rows == 5
    assert train_dataset.num_rows == dummy_dataset.num_rows - 5
    assert "pixel_values" in train_dataset.features
    assert "pixel_values" in val_dataset.features
