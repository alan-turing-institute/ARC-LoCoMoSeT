"""
Functions for calculating the TDA metric for transferability. This metric measure how
topological invariants of the original data and ground truth labels relate to the same
topological features in a combination of the original data and either predicted labels
or features from the last layer of the network. Functionally this is achieved by the
following pipeline (with preprocessing not included):

1. Combine the pixel array for each image with either the ground truth label (either as
a new row of [lab]*pixels or as a one hot vector that's been wrapped around and padded
with zeros) or the predicted labels (similarly either as an integer value or one hot
vector) or the features of the model (reduced via PCA to a divisor of the number of
pixels on a side and repeated along the array to be of same size.)

1. Compute the birth, death, homology dimension triples (a.k.a. homology diagrams) for
each example image, with a set of diagrams for the ground truth and for each model
considered.

2. Combine all the diagrams for the ground truths and models together into a single
array, padding diagram arrays (of shape [n, f, 3]) with either [0,0,0] or [0,0,1]
(b,d,q) triples such that each f is the same and the balance of q=0 and q=1 is the same.

3. Create persistence image from all diagrams (a mapping of the diagrams to various
copies of R^2 with a binned Gaussian applied approximating the manifold). All diagrams
for each model to be considered are required so the image considers the same area of
R^2 for each.

4. The persistence image will be of dimension (m, 2, 100, 100) for 2 homology dimensions
and binned Gaussian of shape 100x100. m = (num_models + 1) * probe_set_size, i.e. there
are n (probe_set_size) persistence images for the ground truth and for each model.
Extract the array corresponding to the ground truths and each model, computing the
Euclidean distance between the ground truth array and each model as the final metric.

NB: The preprocessing requires these to be square, Grayscale images.

NB: This is currently written with a streamed/iterable datset in mind for ease of
initial experiments.
"""


from typing import Iterable

import numpy as np
from datasets import Dataset
from sklearn.preprocessing import OneHotEncoder
from torch import Tensor
from torchvision.transforms import Grayscale, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from locomoset.metrics.parc import _feature_reduce


def pad_and_wrap_one_hot(one_hot: np.ndarray, pix_val=224):
    """If the one hot vector is less than 224 in dimension then pad it to 224. If
    greater than pad it to a multiple of 224 then wrap it and pad last row.

    Args:
        one_hot: One hot vector
        pix_val: Number of pixels on one side of the image. Defaults to 224.

    Returns:
        Array of shape (m, 224) where m is an integer, containing one hot vector in form
        that can be appended to the pixel values of an image.
    """
    if one_hot.shape[0] % pix_val == 0:
        return one_hot
    else:
        pad_vec = np.concatenate(
            (one_hot, np.zeros(pix_val - (one_hot.shape[0] % pix_val)))
        )
        pad_vec = pad_vec.reshape(1, pad_vec.shape[0] // pix_val, pix_val)
        return pad_vec


def one_tda_example(
    img: Tensor,
    model_output: np.ndarray,
    one_hot: bool = False,
    model_features: bool = False,
    pixels_side: int = 224,
):
    """Append either the features or labels (ground truth of pred) (one hot vectors or
    integers) to the pixel values of the corresponding image for use in homology diagram
    creation.

    Args:
        img: pixel values of image, Tensor
        model_output: labels (one hot or integer) or features from the model, NB this is
                      where the ground truth labels are input as well (the output of
                      the null model).
        one_hot: bool noting whether one hot vectors are in use. Defaults to False.
        model_features: bool noting whether the model features are being used over class
                        labels, supercedes the one hot bool. Defaults to False.
        pixels_side: the number of pixels one one side of the image. Defaults to 224.

    Returns:
        _description_
    """
    if model_features:
        return np.concatenate(
            (img.detach().numpy(), model_output.reshape(1, 1, pixels_side)), axis=1
        )
    if one_hot:
        return np.concatenate(
            (img.detach().numpy(), pad_and_wrap_one_hot(model_output, img.shape[2])),
            axis=1,
        )
    return np.concatenate(
        (img.detach().numpy(), np.full((1, 1, img.shape[1]), model_output)), axis=1
    )


def tda_preprocess(examples: Dataset, pixels: int = 224):
    """Preprocessing function for required resizing and grayscaling for the TDA
    pipeline, used in dataset.map(...)

    Args:
        examples: example images from the dataset.
        pixels: number of pixels on one side of the square. Defaults to 224.

    Returns:
        pixel values of a resized and grayscaled image.
    """
    examples["pixel_values"] = [
        ToTensor()(Resize((pixels, pixels))(Grayscale()(image)))
        for image in examples["image"]
    ]
    return examples


def preprocess(examples: Dataset, processor: AutoImageProcessor):
    """Preprocessing function for inference via loaded models. Each model has it's own
    processor, which is given as an argument here. Used in dataset.map(...)

    Args:
        examples: example images from the dataset.
        processor: processor for particular model.

    Returns:
        pixel values of a processed image.
    """
    examples["pixel_values"] = [
        processor(image.convert("RGB"))["pixel_values"][0]
        for image in examples["image"]
    ]
    return examples


def ground_truth_label(n: int, tda_img_iter: Iterable, one_hot: bool = False):
    """Collect the ground truth labels and convert them to one hot vectors if required.

    Args:
        n: size of probe set
        tda_img_iter: preprocessed images for TDA pipeline
        one_hot: determines whether the labels should be one hot vectors. Defaults to
                 False.
    """
    labels = np.zeros(n)
    for idx, label in enumerate(tda_img_iter):
        labels[idx] = label["label"]
        if idx == n - 1:
            break
    if one_hot:
        return OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))
    else:
        return labels


def model_pred_labels(
    n: int,
    inf_img_iter: Iterable,
    model_fn: AutoModelForImageClassification,
    model_features: bool = False,
    one_hot: bool = False,
    random_state: int = 42,
    feat_red_dim: int = 56,
    pixel_side: int = 224,
):
    """Collect either the prediced labels from inference of a model or the features.

    Args:
        n: size of the probe set.
        inf_img_iter: preprocessed images for inference.
        model_fn: model function for either classification of feature extraction.
        model_features: whether features of labels are being returned. Defaults to False
        one_hot: whether the labels are integers or one hots. Defaults to False.
        random_state: random state for use in PCA. Defaults to 42.
        feat_red_dim: dimensions with which to reduce the features. Defaults to 56.
        pixel_side: dimensions of the image in pixels. Defaults to 224.
    """
    if model_features:
        print("computing model features instead of labels")
        preds = []
        for idx, image in enumerate(inf_img_iter):
            preds.append(
                model_fn(image["pixel_values"][None, :, :, :]).logits.detach().numpy()
            )
            if idx == n - 1:
                break
        preds = np.concatenate(preds, axis=0)
        preds = _feature_reduce(preds, random_state, feat_red_dim)
        return np.concatenate(
            [preds for _ in range(pixel_side // feat_red_dim)], axis=1
        )
    else:
        print("computing model labels instead of features")
        preds = np.zeros(n)
        for idx, image in enumerate(inf_img_iter):
            preds[idx] = (
                model_fn(image["pixel_values"][None, :, :, :]).logits.argmax(-1).item()
            )
            if idx == n - 1:
                break
        if one_hot:
            print("converting labels to one hot vectors")
            return OneHotEncoder(sparse_output=False).fit_transform(
                preds.reshape(-1, 1)
            )


def tda_probe_set(
    n: int,
    tda_img_iter: Iterable,
    inf_img_iter: Iterable = None,
    model_fn: AutoModelForImageClassification.from_pretrained = None,
    one_hot: bool = False,
    model_features: bool = False,
    random_state: int = 42,
    pixel_side: int = 224,
    feat_red_dim: int = 56,
    ground_truth: bool = False,
):
    """Create a probeset of images with labels/features appended for use in the TDA
    pipeline.

    Args:
        n: number of images in probe set
        tda_img_iter: iterable dataset of images preprocessed for the TDA pipeline.
        inf_img_iter: (optional) iterable datset of images preprocessed for inference.
        model_fn: (optional) model for inference.
        one_hot: bool denoting whether one hot vectors are being used. Defaults to
                 False.
        model_features: bool denoting whether model features are being used. Defaults to
                        False.
        random_state: random state variable. Defaults to 42.
        pixel_side: pixel size of the side of a square image. Defaults to 224.
        feat_red_dim: dimension with which to reduce the features to. Defaults to 56.
        ground_truth: bool determining wether this create ground truth labels or not.

    Returns:
        _description_
    """
    # Work around as streaming data
    if ground_truth:
        labels = ground_truth_label(n, tda_img_iter=tda_img_iter, one_hot=one_hot)
    else:
        labels = model_pred_labels(
            n,
            inf_img_iter=inf_img_iter,
            model_fn=model_fn,
            one_hot=one_hot,
            random_state=random_state,
            feat_red_dim=feat_red_dim,
            pixel_side=pixel_side,
        )
    print("finished computing the labes/predicted values")

    examples = []
    for idx, image in enumerate(tda_img_iter):
        examples.append(
            one_tda_example(
                image["pixel_values"],
                labels[idx],
                one_hot,
                model_features,
                random_state,
            )
        )
        if idx == n - 1:
            break
    return np.concatenate(examples, axis=0)


def model_diags(model_name: str, tda_img_iter: Iterable, homology, **pars):
    """Compute the homology diagrams for a given model. This loads the model and
    processor first, before processing the dataset for inference.

    Args:
        model_name: name of the pre trained hugging face model to load
        tda_img_iter: preprocessed dataset for TDA pipeline
        homology: TDA homology for generating the diagrams

    Returns:
        homology diagrams, array of shape (n, f, 3). For n examples in the probe set
        with f topological features, denoted by birth death triples [b, d, q].
    """
    # instantiate model_function and processor
    print("loading model and processor")
    if pars["model_features"]:
        model_fn = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=0
        )
    else:
        model_fn = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    # preprocess dataset
    print("processing data")
    dataset = pars["dataset"]
    fn_kwargs = {"processor": processor}
    imgs = dataset.map(
        preprocess,
        remove_columns="image",
        batched=True,
        batch_size=1,
        fn_kwargs=fn_kwargs,
    )
    imgs = imgs.with_format("torch")

    # create probeset
    print("model probe set computation")
    probe_set = tda_probe_set(
        pars["n_examples"],
        tda_img_iter,
        imgs,
        model_fn,
        pars["one_hot_labels"],
        pars["model_features"],
        pars["random_state"],
    )

    # return b,d,q diagrams
    print("model diags computation")
    return homology.fit_transform(probe_set)


def pad_hom_diags(
    diags1: np.ndarray, diags1_dim0_count: int, diags2_dim0_count: int, hom_dim: int = 0
):
    """Pad diags1 with [0, 0, hom_dim] to match number of equivalent homology dimension
    diagrams in diags2.

    Args:
        diags1: diagram array 1, of shape (n_examples, num_topological features, 3)
        diags1_dim0_count: number of triples in diags 1 of form [x, y, hom_dim]
        diags2_dim0_count: number of triples in daigs 2 of form [x, y, hom_dim]
    """
    padding = np.zeros(
        (diags1.shape[0], diags2_dim0_count - diags1_dim0_count, diags1.shape[0])
    )
    padding[:, :, 2] = float(hom_dim)
    return np.concatenate((diags1, padding), axis=1)


def merge_diags(diags1: np.ndarray, diags2: np.ndarray):
    """Combine two diagram arrays. These are arrays of shape:

        (n_examples, n_topological_features, 3)

    Expect the first and last dimensions to be the same, but a mismatch in the middle,
    but we also need them balanced so there are the same total number of (b,d,q) triples
    for each value of q. This function will pad the diagrams with either [0,0,0] or
    [0,0,1] triples, not affecting the topological characteristics but leading to
    matching arrays for the purpose of image computation.

    Params:
        - diags1: diagram array 1
        - diags2: diagram array 2
    """
    if diags1.shape[1] == diags2.shape[1]:
        return np.concatenate((diags1, diags2), axis=0)
    else:
        diags1_dims_count = np.unique(diags1[0, :, 2], return_counts=True)[1]
        diags2_dims_count = np.unique(diags2[0, :, 2], return_counts=True)[1]

        for hom_dim, hom_dim_counts in enumerate(
            zip(diags1_dims_count, diags2_dims_count)
        ):
            if hom_dim_counts[1] > hom_dim_counts[0]:
                diags1 = pad_hom_diags(
                    diags1, hom_dim_counts[0], hom_dim_counts[1], hom_dim
                )
            else:
                diags2 = pad_hom_diags(
                    diags2, hom_dim_counts[1], hom_dim_counts[0], hom_dim
                )

        return np.concatenate((diags1, diags2), axis=0)
