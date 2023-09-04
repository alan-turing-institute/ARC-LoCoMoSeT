"""
Script for running the full pipeline of the TDA metric to produce experimental results
for varied hyperparameters.

NB: This is currently written with a streamed/iterable datset in mind for ease of 
initial experiments.
"""

from typing import Iterable

import numpy as np
from datasets import Dataset, load_dataset
from gtda import diagrams as dgms
from gtda import homology as hom
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from tdqm import tdqm
from torch import Tensor, cat, flatten, reshape
from torchvision.transforms import Grayscale, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from locomoset.metrics.tda import (
    merge_diags,
    model_diags,
    tda_preprocess,
    tda_probe_set,
)


def tda_metric(**pars):
    """Draft pipeline for TDA metric.

    Params:
        - dataset: dataset
        - n_examples: size of the probe set
        - num_classes: number of classes in dataset
        - models: list of tuples of model functions and processors
        - pixel_square: one side of pixel length to reshape images to
        - one_hot_labels: use one hot labels vs. integer labels in padding
        - model_features: use the features or the classifier
        - random_state: seed for randomness
        - feature_reduction_level: divisor of pixel_square to reduce to
    """
    # TDA objects:
    cub_hom = hom.CubicalPersistence()
    per_img = dgms.PersistenceImage()
    print("loaded tda objects")

    # If using streamed/iterable dataset it needs to be shuffled for variation in images
    pars["dataset"] = pars["dataset"].shuffle(seed=pars["random_state"])

    # Start by creating the diagrams for the ground truths
    print("starting ground truth diagram computation")
    fn_kwargs = None
    if pars["pixel_square"] is not None:
        pixels = pars["pixel_square"]
        fn_kwargs = {"pixels": pixels}
    tda_imgs = pars["dataset"].map(
        tda_preprocess,
        remove_columns="image",
        batched=True,
        batch_size=1,
        fn_kwargs=fn_kwargs,
    )
    tda_imgs = tda_imgs.with_format("torch")
    print("tda image processing done!")

    print("starting computation of ground truth probe set")
    gtruth_probe_set = tda_probe_set(
        pars["n_examples"], tda_imgs, pars["one_hot_labels"], ground_truth=True
    )
    print("ground truth probe set completed")

    print("computing ground truth diagrams from probe set")
    diags = cub_hom.fit_transform(gtruth_probe_set)
    diags_shape = diags.shape
    print(f"ground truth diagrams computed, with shape {diags_shape}")

    # Create diagrams for each model and append to diags
    print("starting computation of model diagrams")
    model_diagrams = []
    for model in tqdm(pars["models"]):
        print(f"computing diagrams for {model}")
        model_diag = model_diags(model, tda_imgs, cub_hom, **pars)
        print(f"model diags shape {model_diag.shape}")
        # assert model_diags.shape == diags.shape, 'Diagram dimensions do not match!'
        # diags = np.concatenate((diags, model_diag), axis=0)
        model_diagrams.append(model_diag)
    mdiags = model_diagrams[0]
    if len(model_diagrams) > 1:
        for i in range(1, len(model_diagrams)):
            mdiags = merge_diags(mdiags, model_diagrams[i])
    print("done with model diag computations")

    print("combine diagrams")
    diags = merge_diags(diags, mdiags)
    print(f"final diagrams shape {diags.shape}")

    # Compute persistance image for entire set
    print("computing persistance image")
    perst_img = per_img.fit_transform(diags)

    # Compute metric from this
    print("computing metrics")
    results = {}
    base_img = perst_img[: pars["n_examples"], :, :, :]
    for idx, model in enumerate(pars["models"]):
        print(f"comp idxs {(idx+1)*pars['n_examples']}, {(idx+2)*pars['n_examples']}")
        results[model] = np.linalg.norm(
            base_img
            - perst_img[
                (idx + 1) * pars["n_examples"] : (idx + 2) * pars["n_examples"], :, :, :
            ]
        )
    return results, perst_img
