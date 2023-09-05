"""
Test functions for the tda functions script (src/locomoset/metrics/tda.py)
"""

import numpy as np

# from locomoset.metrics.parc import _feature_reduce
from locomoset.metrics.tda import (  # model_pred_labels,; preprocess,
    ground_truth_label,
    merge_diags,
    one_tda_example,
    pad_and_wrap_one_hot,
    pad_hom_diags,
    tda_preprocess,
)

# import pytest


def test_pad_and_wrap_one_hot(
    dummy_one_hot_perfect,
    dummy_one_hot_large,
    dummy_one_hot_small,
    dummy_one_hot_extra_large,
):
    """Test that the pad and wrap one hot vectors function returns the correct shape for
    the three cases of a perfect one hot vector (224, ), a small one hot vector (200, ),
    a large vector (260, ) and an extra large vector (512, ), for the default pixel
    dimension 224.
    """
    assert pad_and_wrap_one_hot(dummy_one_hot_perfect).shape == (1, 1, 224)
    assert pad_and_wrap_one_hot(dummy_one_hot_small).shape == (1, 1, 224)
    assert pad_and_wrap_one_hot(dummy_one_hot_large).shape == (1, 2, 224)
    assert pad_and_wrap_one_hot(dummy_one_hot_extra_large).shape == (1, 3, 224)
    assert pad_and_wrap_one_hot(dummy_one_hot_extra_large)[0, 1, 36] == 1.0


def test_tda_preprocess_streamed(dummy_streamed_dataset):
    """Test that the tda preprocessing is formatting the images correctly."""
    tda_imgs = dummy_streamed_dataset.map(
        tda_preprocess, batched=True, batch_size=1, remove_columns="image"
    )
    print(next(iter(tda_imgs)))
    assert next(iter(tda_imgs))["pixel_values"].shape == (1, 224, 224)


def test_one_tda_example(
    dummy_streamed_dataset,
    dummy_label,
    dummy_one_hot_small,
    dummy_one_hot_perfect,
    dummy_one_hot_large,
    dummy_one_hot_extra_large,
    rng,
):
    """Test that a single output for the preprocessing of images and labels/features to
    put into the TDA pipeline is giving the correct output.
    """
    tda_imgs = dummy_streamed_dataset.map(
        tda_preprocess, batched=True, batch_size=1, remove_columns="image"
    )
    img = next(iter(tda_imgs))["pixel_values"]
    dummy_features = rng.normal(size=224)
    assert one_tda_example(
        img, dummy_features, one_hot=False, model_features=True
    ).shape == (1, 225, 224)
    assert one_tda_example(
        img, dummy_label, one_hot=False, model_features=False
    ).shape == (1, 225, 224)
    assert one_tda_example(
        img, dummy_one_hot_small, one_hot=True, model_features=False
    ).shape == (1, 225, 224)
    assert one_tda_example(
        img, dummy_one_hot_perfect, one_hot=True, model_features=False
    ).shape == (1, 225, 224)
    assert one_tda_example(
        img, dummy_one_hot_large, one_hot=True, model_features=False
    ).shape == (1, 226, 224)
    assert one_tda_example(
        img, dummy_one_hot_extra_large, one_hot=True, model_features=False
    ).shape == (1, 227, 224)


def test_ground_try_label(test_n_samples, dummy_streamed_dataset):
    """Test that the shape of the ground truth labels output is correct."""
    tda_imgs = dummy_streamed_dataset.map(
        tda_preprocess, batched=True, batch_size=1, remove_columns="image"
    )
    assert ground_truth_label(test_n_samples, tda_imgs, one_hot=False).shape == (
        test_n_samples,
    )
    assert ground_truth_label(test_n_samples, tda_imgs, one_hot=True).shape == (
        test_n_samples,
        3,
    )


def test_pad_hom_diags(dummy_diag_hom0_weighted, dummy_diag_hom1_weighted):
    """Test the padding of homology diagrams for specific homology dimension."""
    hom0_weighted_dims_count = np.unique(
        dummy_diag_hom0_weighted[0, :, 2], return_counts=True
    )[1]
    hom1_weighted_dims_count = np.unique(
        dummy_diag_hom1_weighted[0, :, 2], return_counts=True
    )[1]
    # assert pad_hom_diags(
    #     dummy_diag_hom0_weighted,
    #     hom1_weighted_dims_count[0],
    #     hom0_weighted_dims_count[0],
    #     hom_dim=0,
    # ) == ValueError("Diagram does not match homology dimension features")
    assert pad_hom_diags(
        dummy_diag_hom1_weighted,
        hom1_weighted_dims_count[0],
        hom0_weighted_dims_count[0],
        hom_dim=0,
    ).shape == (
        dummy_diag_hom1_weighted.shape[0],
        hom0_weighted_dims_count[0] + hom1_weighted_dims_count[1],
        3,
    )
    assert (
        np.unique(
            pad_hom_diags(
                dummy_diag_hom1_weighted,
                hom1_weighted_dims_count[0],
                hom0_weighted_dims_count[0],
                hom_dim=0,
            ),
            return_counts=True,
        )[1].all()
        == np.array([hom0_weighted_dims_count[0], hom1_weighted_dims_count[1]]).all()
    )
    assert pad_hom_diags(
        dummy_diag_hom0_weighted,
        hom0_weighted_dims_count[1],
        hom1_weighted_dims_count[1],
        hom_dim=1,
    ).shape == (
        dummy_diag_hom0_weighted.shape[0],
        hom1_weighted_dims_count[1] + hom0_weighted_dims_count[0],
        3,
    )
    assert (
        np.unique(
            pad_hom_diags(
                dummy_diag_hom0_weighted,
                hom0_weighted_dims_count[1],
                hom1_weighted_dims_count[1],
                hom_dim=1,
            ),
            return_counts=True,
        )[1].all()
        == np.array([hom0_weighted_dims_count[0], hom1_weighted_dims_count[1]]).all()
    )


def test_merge_diags(dummy_diag_hom0_weighted, dummy_diag_hom1_weighted):
    """Test that the merging diagrams helper function is working correctly."""
    hom0_weighted_dims_count = np.unique(
        dummy_diag_hom0_weighted[0, :, 2], return_counts=True
    )[1]
    hom1_weighted_dims_count = np.unique(
        dummy_diag_hom1_weighted[0, :, 2], return_counts=True
    )[1]
    assert merge_diags(dummy_diag_hom0_weighted, dummy_diag_hom0_weighted).shape == (
        dummy_diag_hom0_weighted.shape[0] * 2,
        dummy_diag_hom0_weighted.shape[1],
        dummy_diag_hom0_weighted.shape[2],
    )
    assert (
        np.unique(
            merge_diags(dummy_diag_hom0_weighted, dummy_diag_hom0_weighted)[0, :, 2],
            return_counts=True,
        )[1].all()
        == (hom0_weighted_dims_count * 2).all()
    )
    assert merge_diags(dummy_diag_hom0_weighted, dummy_diag_hom1_weighted).shape == (
        dummy_diag_hom0_weighted.shape[0] + dummy_diag_hom1_weighted.shape[0],
        max(hom0_weighted_dims_count[0], hom1_weighted_dims_count[0])
        + max(hom0_weighted_dims_count[1], hom1_weighted_dims_count[1]),
        dummy_diag_hom0_weighted.shape[2],
    )
    assert (
        np.unique(
            merge_diags(dummy_diag_hom0_weighted, dummy_diag_hom1_weighted)[0, :, 2],
            return_counts=True,
        )[1].all()
        == np.array(
            [
                max(hom0_weighted_dims_count[0], hom1_weighted_dims_count[0]),
                max(hom0_weighted_dims_count[1], hom1_weighted_dims_count[1]),
            ]
        ).all()
    )
