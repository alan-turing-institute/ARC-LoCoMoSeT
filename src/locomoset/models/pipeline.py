import warnings

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import ImageClassificationPipeline


class ImageFeaturesPipeline(ImageClassificationPipeline):
    """
    A custom pipeline for returning the features of an image from a model with its
    classification head removed.
    """

    def __init__(
        self,
        *args,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        **kwargs,
    ):
        # Check the model has its classification head removed
        err_prefix = (
            "model must have its classification head removed to use "
            "ImageFeaturesPipeline"
        )
        if hasattr(model, "classifier"):
            if not isinstance(model.classifier, torch.nn.Identity):
                raise ValueError(
                    f"{err_prefix} (model.classifier should be torch.nn.Identity "
                    f"but it was {type(model.classifier)})"
                )
        elif hasattr(model, "config") and hasattr(model.config, "num_labels"):
            if model.config.num_labels != 0:
                raise ValueError(
                    f"{err_prefix} (model.config.num_labels should be 0 but it was "
                    f"{model.config.num_labels})"
                )
        else:
            warnings.warn(
                "Unable to check model has classification head removed (model does not "
                "have a classifier attribute or a config.num_labels attribute)"
            )

        super().__init__(*args, model=model, image_processor=image_processor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters: dict) -> (dict, dict, dict):
        """Not used here but required for inheriting from the Pipeline class. Returns
        parameters to pass to the `preprocess`, `forward` and `postprocess` methods"""
        return {}, {}, {}

    def postprocess(
        self, model_outputs: ImageClassifierOutput, **postprocess_parameters: dict
    ) -> torch.tensor:
        """Extract features from the model output.

        Args:
            model_outputs: Output of the model including the key `logits` which
                contains the features for one image (or multiple images).
            postprocess_parameters: Unused here but required for compatibility with the
                base Pipeline class.

        Returns:
            Tensor containing features for one image (or multiple images).
        """
        return model_outputs["logits"]
