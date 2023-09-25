import numpy as np

from locomoset.datasets.preprocess import preprocess


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
