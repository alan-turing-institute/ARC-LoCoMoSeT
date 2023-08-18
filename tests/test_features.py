import pytest
import torch
from transformers import AutoModelForImageClassification
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from locomoset.models.features import get_features
from locomoset.models.pipeline import ImageFeaturesPipeline


@pytest.fixture
def dummy_pipeline(dummy_model_head, dummy_processor):
    return ImageFeaturesPipeline(
        model=dummy_model_head, image_processor=dummy_processor
    )


@pytest.fixture
def dummy_image(dummy_dataset):
    return dummy_dataset[0]["image"]


@pytest.fixture
def dummy_processed_image(dummy_image, dummy_processor):
    return dummy_processor(dummy_image.convert("RGB"), return_tensors="pt")


def test_pipe_create(dummy_pipeline):
    """
    Test an ImageFeaturesPipeline can be instantiated.
    """
    assert isinstance(dummy_pipeline, ImageFeaturesPipeline)
    assert isinstance(dummy_pipeline.model, PreTrainedModel)
    assert isinstance(dummy_pipeline.image_processor, BaseImageProcessor)


def test_pipe_fails_with_head(dummy_model_name, dummy_processor):
    """_summary_

    Args:
        dummy_model_name (_type_): _description_
        dummy_processor (_type_): _description_
    """
    model = AutoModelForImageClassification.from_pretrained(dummy_model_name)
    with pytest.raises(ValueError):
        ImageFeaturesPipeline(model=model, image_processor=dummy_processor)


def test_pipe_preprocess(dummy_image, dummy_processed_image, dummy_pipeline):
    """
    Test the preprocess function correctly converts an image in the dummy dataset
    """
    pipe_proc = dummy_pipeline.preprocess(dummy_image)["pixel_values"][0]
    assert torch.equal(pipe_proc, dummy_processed_image["pixel_values"][0])


def test_pipe_forward(dummy_processed_image, dummy_pipeline, dummy_model_head):
    """
    Test the pipeline _forward function correctly passes processed images to the model.
    """
    pipe_out = dummy_pipeline._forward(dummy_processed_image)
    model_out = dummy_model_head(dummy_processed_image["pixel_values"])
    assert torch.equal(pipe_out.logits, model_out.logits)


def test_pipe_postprocess(dummy_processed_image, dummy_pipeline):
    """
    Test the pipeline postprocess function correctly extracts logits from the model
    output.
    """
    pipe_out = dummy_pipeline._forward(dummy_processed_image)
    assert torch.equal(dummy_pipeline.postprocess(pipe_out), pipe_out.logits)


def test_get_features(dummy_dataset, dummy_processor, dummy_model_head):
    """
    Test running the pipeline on a whole dataset to extract features.
    """
    features = get_features(dummy_dataset, dummy_processor, dummy_model_head)
    assert features.shape == (len(dummy_dataset), dummy_model_head.config.hidden_size)