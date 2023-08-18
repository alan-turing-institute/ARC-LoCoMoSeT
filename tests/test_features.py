import numpy as np
import pytest

from locomoset.models.features import get_features
from locomoset.models.pipeline import ImageFeaturesPipeline


@pytest.fixture
def dummy_pipeline(dummy_model_head, dummy_processor):
    return ImageFeaturesPipeline(
        model=dummy_model_head, image_processor=dummy_processor
    )


@pytest.fixture
def dummy_pipeline_iterator(dummy_pipeline, dummy_dataset):
    return dummy_pipeline(dummy_dataset)


@pytest.fixture
def dummy_sample(dummy_dataset):
    return dummy_dataset[0]


@pytest.fixture
def dummy_image(dummy_sample):
    return dummy_sample["image"]


@pytest.fixture
def dummy_processed_image(dummy_image, dummy_processor):
    return dummy_processor(dummy_image.convert("RGB"))


def test_pipe_preprocess(dummy_sample, dummy_processed_image, dummy_pipeline):
    """
    Test the preprocess function correctly converts images in the dummy dataset
    """
    pipe_proc = dummy_pipeline.preprocess(dummy_sample)["pixel_values"][0]
    np.testing.assert_equal(pipe_proc, dummy_processed_image["pixel_values"][0])


def test_pipe_forward(
    dummy_processed_image, dummy_pipeline, dummy_model_head, dummy_sample
):
    pipe_out = dummy_pipeline._forward(dummy_pipeline.preprocess(dummy_sample))
    model_out = dummy_model_head(dummy_processed_image["pixel_values"][0])
    np.testing.assert_equal(pipe_out, model_out)


def test_get_features(dummy_dataset, dummy_processor, dummy_model_head):
    """
    Test features can be extracted from a model correctly
    """
    features = get_features(dummy_dataset, dummy_processor, dummy_model_head)
    assert features.shape == (len(dummy_dataset), dummy_model_head.config.hidden_size)
