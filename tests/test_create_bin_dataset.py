"""
    Test the functions from creating binary datasets,
    (src/locomoset/datasets/create_bin_datasets.py)
"""

import numpy as np

from locomoset.datasets.create_bin_dataset import drop_images_by_id, get_balanced_id

# import pytest


def test_get_balanced_id(dummy_dataset, dummy_label, select_label1):
    """Test the get_balanced_id method"""
    bin_dataset = dummy_dataset.map(
        select_label1, batched=False, fn_kwargs={"label": dummy_label}
    )
    balanced_id = get_balanced_id(bin_dataset, np.sum(bin_dataset["train"]["label"]))
    assert len(balanced_id) == 54


def test_drop_images_by_id(dummy_dataset, dummy_label, select_label1):
    """Test the drop_image_by_id method"""
    bin_dataset = dummy_dataset.map(
        select_label1, batched=False, fn_kwargs={"label": dummy_label}
    )
    dropped_dataset = drop_images_by_id(
        bin_dataset,
        drop_images_by_id(bin_dataset, np.sum(bin_dataset["train"]["label"]) * 2),
    )
    assert len(dropped_dataset["train"]) == 54
    assert np.sum(dropped_dataset["train"]["label"]) == 27
