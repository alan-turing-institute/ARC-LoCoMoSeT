import torch
from PIL.Image import Image
from transformers import Pipeline
from transformers.modeling_outputs import ImageClassifierOutput


class ImageFeaturesPipeline(Pipeline):
    """
    A custom pipeline for returning the features of an image from a model with its
    classification head removed.
    """

    def __init__(self, *args, **kwargs):
        for expected_arg in ["model", "image_processor"]:
            if expected_arg not in kwargs:
                raise ValueError(
                    f"{expected_arg} must be passed to ImageFeaturesPipeline"
                )
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters: dict) -> (dict, dict, dict):
        """Not used here but required for inheriting from the Pipeline class. Returns
        parameters to pass to the `preprocess`, `forward` and `postprocess` methods"""
        return {}, {}, {}

    def preprocess(
        self, input_: dict[str, Image], **preprocess_parameters: dict
    ) -> dict[str, torch.tensor]:
        """Preprocess a batch of images (or a single image in streaming mode).

        Args:
            input_: Batch from an image dataset, a dict with the key 'image'
                containing a single PIL image or a list of PIL images.
            preprocess_parameters: Unused here but required for compatibility with the
                base Pipeline class.

        Returns:
            Processed batch/image, a dict with the key 'pixel_values' containing a
                tensor.
        """
        if isinstance(input_["image"], list):
            return self.image_processor(
                [i.convert("RGB") for i in input_["image"]], return_tensors="pt"
            )

        return self.image_processor(input_["image"].convert("RGB"), return_tensors="pt")

    def _forward(
        self, model_inputs: dict[str, torch.tensor], **forward_parameters: dict
    ) -> ImageClassifierOutput:
        """Pass a batch of processed images to the headless model.

        Args:
            model_inputs: A batch of processed images, a dict with the key
                'pixel_values' containing a tensor with shape
                (batch_size, n_channels, image_size, image_size).
            forward_parameters: Unused here but required for compatibility with the
                base Pipeline class.

        Returns:
            Output of the model, in this including the key `logits` which contains the
                features of each image in the batch.
        """
        return self.model(model_inputs["pixel_values"])

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
