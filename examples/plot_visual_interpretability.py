# -*- coding: utf-8 -*-
"""
Inspect trained networks
========================

Credit: A Grigis

Various techniques are available for understanding how neural networks work,
what do they learn and why they work in a given manner. Specifically, we will
try out various visualization techniques to understand them more deeply:

* Visualizing kernels.
* Visualizing intermediate activations: what is the output after each
  convolutional operation.
* Visualizing filters: what features does each filter extract from the
  input image via activation maximization.
* Visualizing saliency maps: highlight most salient regions.
"""

import os
import tempfile
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from brainboard.backprop import Forward, Backprop
from brainboard.activmax import GradientAscent


def load_image(image_path):
    """ Loads image as a PIL RGB image.

    Parameters
    ----------
    image_path: str
        a path to the image.
    Returns
    -------
    out: instance of PIL.Image
        image in RGB.
    """
    return Image.open(image_path).convert("RGB")


def apply_transforms(image, size=None):
    """ Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of `(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.
    The plan is to to expand this to handle custom size/mean/std.

    Parameters
    ----------
    image: PIL.Image.Image or numpy array (C, H, W)
        the input image.
    size: int, default=224
        desired size (width/height) of the output tensor. If None, do not
        apply resizing and cropping.

    Returns
    -------
    out: torch.Tensor (N, C, H, W: torch.float32)
        transformed image tensor.

    Notes
    -----
    Symbols used to describe dimensions:
    * N: number of images in a batch.
    * C: number of channels.
    * H: height of the image.
    * W: width of the image.
    """
    if size is not None and not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    if size is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True

    return tensor


def denormalize(tensor):
    """ Reverses the normalisation on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean

    Parameters
    ----------
    tensor: torch.Tensor (N, C, H, W)
        normalized image tensor.

    Returns
    -------
    torch.Tensor (N, C, H, W)
        demornalised image tensor with pixel values between [0, 1].

    Notes
    -----
    Symbols used to describe dimensions:
    * N: number of images in a batch.
    * C: number of channels.
    * H: height of the image.
    * W: width of the image.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    denormalized = tensor.clone()
    for cnt in range(len(denormalized[0])):
        denormalized[0, cnt] = denormalized[0, cnt] * stds[cnt] + means[cnt]

    return denormalized


def standardize_and_clip(tensor, min_value=0.0, max_value=1.0,
                         saturation=0.1, brightness=0.5):
    """ Standardizes and clips input tensor.

    Standardizes the input tensor (mean=0.0, std=1.0). The color saturation
    and brightness are adjusted, before tensor values are clipped to min/max.

    Parameters
    ----------
    tensor: torch.Tensor (C, H, W)
        the input tensor.
    min_value: float, default 0.0
        the min value range.
    max_value: float, default 1.0
        the max value range.
    saturation: float, default 0.1
        the selected saturation.
    brightness: float, default 0.5
        the selected brightness.

    Returns
    -------
    out: torch.Tensor (C, H, W)
        normalised tensor with values between [min_value, max_value].
    """
    tensor = tensor.detach().cpu()
    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        std += 1e-7
    standardized = tensor.sub(mean).div(std).mul(saturation)
    clipped = standardized.add(brightness).clamp(min_value, max_value)
    return clipped


def format_for_plotting(tensor):
    """ Formats the shape of tensor for plotting.

    Tensors typically have a shape of (N, C, H, W) or (C, H, W)
    which is not suitable for plotting as images. This function formats an
    input tensor (H, W, C) for RGB and (H, W) for mono-channel data.

    Parameters
    ----------
    tensor: torch.Tensor (N, C, H, W) or (C, H, W)
        input image tensor

    Returns
    -------
    out: torch.Tensor (H, W, C) or (H, W)
        formatted image tensor (detached).

    Notes
    -----
    Symbols used to describe dimensions:
    * N: number of images in a batch.
    * C: number of channels.
    * H: height of the image.
    * W: width of the image.
    """
    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()
    if has_batch_dimension:
        formatted = tensor.squeeze(0)
    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def visualize_activations(activations, images_per_row=16):
    """ Visulaize activations.

    Parameters
    ----------
    activations: list of torch.Tensor (1, C, S, S)
        the different layers activations.
    images_per_row: int, default 16
        the number of images to be displayed on each row.
    """
    for layer_idx, layer_activation in enumerate(activations):
        layer_activation = layer_activation.clone().detach().cpu().numpy()
        n_features = layer_activation.shape[1]
        size = layer_activation.shape[2]
        n_cols = n_features // images_per_row

        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        fig = plt.figure(figsize=(scale * display_grid.shape[1],
                                  scale * display_grid.shape[0]))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.set_title("Activation maps at layer {0}".format(layer_idx + 1))
        ax.grid(False)
        ax.imshow(display_grid, aspect="auto", cmap="viridis")


def visualize_kernels(kernels, images_per_row=6):
    """ Visulaize kernels.

    Parameters
    ----------
    kernels: list of torch.Tensor (N, C, S, S)
        the different kernels.
    images_per_row: int, default 6
        the number of images to be displayed on each row.
    """
    n_kernels = kernels.shape[0]
    n_rows =  n_kernels // images_per_row
    if n_kernels % images_per_row > 0:
        n_rows += 1
    fig = plt.figure(figsize=(n_rows, images_per_row))
    for idx, tensor in enumerate(kernels):
        max_val = tensor.max()
        min_val = abs(tensor.min())
        max_val = max(max_val, min_val)
        tensor = tensor / max_val
        tensor = tensor / 2.
        tensor = tensor + 0.5
        ax = fig.add_subplot(n_rows, images_per_row, idx + 1)
        img = format_for_plotting(tensor)
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def visualize_gradients(input_tensor, gradients, max_gradients,
                        figsize=(16, 4), cmap="viridis", alpha=.5):
    """ Visulaize gradients.

    Parameters
    ----------
    input_tensor: torch.Tensor (N, C, H, W)
        the input data.
    gradients: torch.Tensor (C, H, W).
        the computed gradients.
    max_gradients: torch.Tensor (1, H, W).
        the computed channel wise max gradients.
    figsize: tuple, default (16, 4)
        the size of the plot.
    cmap: str, default 'viridis'
        the color map of the gradients plots.
    alpha: float, default 0.5
        the alpha value of the max gradients in order to supperpose the
        gradients on top of the input image.
    """
    # Setup subplots
    # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
    subplots = [
        ("Input image",
         [(format_for_plotting(denormalize(input_tensor)), None, None)]),
        ("Gradients across channels",
         [(format_for_plotting(standardize_and_clip(gradients)),
          None,
          None)]),
        ("Max gradients",
         [(format_for_plotting(standardize_and_clip(max_gradients)),
          cmap,
          None)]),
        ("Overlay",
         [(format_for_plotting(denormalize(input_tensor)), None, None),
          (format_for_plotting(standardize_and_clip(max_gradients)),
           cmap,
           alpha)])
    ]
    fig = plt.figure(figsize=figsize)
    for idx, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, len(subplots), idx + 1)
        ax.set_axis_off()
        ax.set_title(title)
        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)


def visualize_filters(responses, layer, filter_idxs, images_per_row=4):
    """ Visulaize filter responses.

    Parameters
    ----------
    responses: list of list of torch.Tensor (n_filters, n_iter, C, H, W)
        the selected filters responses through the gradient ascent.
    filter_idxs: list of int
        the indicies of the target filters.
    images_per_row: int, default 16
        the number of images to be displayed on each row.
    """
    n_subplots = len(responses)
    n_rows = int(np.ceil(n_subplots / images_per_row))
    fig = plt.figure(figsize=(16, images_per_row * 5))
    plt.title("Filter responses: layer {0}".format(layer))
    plt.axis("off")

    for idx, filter_idx in enumerate(filter_idxs):
        output = responses[idx][-1]
        ax = fig.add_subplot(n_rows, images_per_row, idx + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("filter {0}".format(filter_idx))
        ax.imshow(format_for_plotting(
            standardize_and_clip(output, saturation=0.15, brightness=0.7)))

    plt.subplots_adjust(wspace=0, hspace=0)


def download(datadir):
    """ Download the image.
    """
    base_url = (
        "https://github.com/MisaOgura/flashtorch/blob/master/examples/"
        "images")
    names = ("great_grey_owl.jpg", "peacock.jpg", "toucan.jpg")

    data = {}
    for name in names:
        url = base_url + "/" + name + "?raw=true"
        destfile = os.path.join(datadir, name)
        if not os.path.isfile(destfile):
            print("- downloading: {0}.".format(url))
            subprocess.check_call([
                "curl", "-L", url, "--output", destfile])
        data[name.split(".")[0]] = destfile

    return data


# Load data
with tempfile.TemporaryDirectory() as datadir:
    data = download(datadir)
    for key in data.keys():
        data[key] = load_image(data[key])


# Load a pretrained model
model = models.alexnet(pretrained=True)
print(model)


############################################################################
# Visualizing kernels
# -------------------
# 
# Typical-looking filters on the first conv layer of a trained AlexNet.
# Notice that the first-layer weights are very nice and smooth, indicating
# nicely converged network. The color/grayscale features are clustered
# because the AlexNet contains two separate streams of processing, and an
# apparent consequence of this architecture is that one stream develops
# high-frequency grayscale features and the other low-frequency color features.

conv_layer = model.features[0]
if isinstance(conv_layer, torch.nn.Conv2d):
    weight_tensor = conv_layer.weight.data.clone()
    visualize_kernels(weight_tensor, images_per_row=6)
else:
    print("Can only visualize layers which are convolutional")


############################################################################
# Visualizing intermediate activations
# ------------------------------------
#
# Simply plot what each filter has extracted (output features) after a
# convolution operation in each layer.
# The initial layers retain most of
# the input image features. It looks like the convolution filters are
# activated at every part of the input image. This gives us an intuition that
# these initial filters might be primitive edge detectors since we can
# consider a complex figure to be made up of small edges, with different
# orientations put together.
# As we go deeper the features extracted by the filters become visually less
# interpretable. An intuition for this can be that the convnet is now
# abstracting away visual information of the input image and trying to convert
# it to the required output classification domain.
# We see a lot of blank convolution outputs. This means that the pattern
# encoded by the filters were not found in the input image. Most probably,
# these patterns must be complex shapes that are not present in this input
# image.
# We can compare these insights with how our own visual perception works: when
# we look at an object (say bicycle) we don't sit and observe each and every
# detail of the object. All we see is an object with two wheels being joined
# by a metallic rod. Hence, if we were told to draw a bicycle it would be a
# simple sketch which just conveys the idea of two wheels and a metallic rod.
# This information is enough for us to decide that the given object is a
# bicycle.
# Something similar is happening in deep neural networks as well. They act as
# information distillation pipeline where the input image is being converted
# to a domain which is visually less interpretable (by removing irrelevant
# information) but mathematically useful for convnet to make a choice from
# the output classes in its last layers.

forward = Forward(model.features)
image = data["great_grey_owl"]
input_tensor = apply_transforms(image, size=224)
activations = forward.get_activations(input_tensor, use_gpu=False)
visualize_activations(activations, images_per_row=16)


############################################################################
# Visualizing filters
# -------------------
#
# Start by displaying the mean activation per feature map for a specific
# layer given an input image.
# A convolution operation in its most basic terms is the correlation between
# the filters/kernels and the input image. The filter which matches the most
# with the given input region will produce an output which will be higher in
# magnitude (compared to the output from other filters).
# Ok, there clearly are spikes (but there are more). Let's take a look at
# patterns generated for these three feature maps.

activation_idx = 4
thresh = 2.5
forward = Forward(model.features, activation_klass=nn.Conv2d)
image = data["great_grey_owl"]
input_tensor = apply_transforms(image, size=224)
activations = forward.get_activations(input_tensor, use_gpu=False)
layer_activation = activations[activation_idx]
n_filters_in_layer = layer_activation.size(dim=1)
mean_act = [layer_activation[0, idx].mean().item()
            for idx in range(n_filters_in_layer)]
peaks = [idx for idx in range(n_filters_in_layer) if mean_act[idx] > thresh]
print("- peaks:", peaks)
plt.figure(figsize=(7,5))
act = plt.plot(mean_act, linewidth=2.)
ax = act[0].axes
ax.set_xlim(0, n_filters_in_layer)
for val in peaks:
    plt.axvline(x=val, color="grey", linestyle="--")
ax.set_xlabel("feature map")
ax.set_ylabel("mean activation")
ax.set_xticks([0, n_filters_in_layer] + peaks)


############################################################################
# By visualizing filters we get an idea of what pattern each layer has learned
# to extract from the input.
# The idea is the following: we start with a picture containing random pixels.
# We apply the network in evaluation mode to this random image, calculate the
# average activation of a certain feature map in a certain layer from which
# we then compute the gradients with respect to the input image pixel values.
# Knowing the gradients for the pixel values we then proceed to update the
# pixel values in a way that maximizes the average activation of the chosen
# feature map (response of a specific filter).
# However, we might end up in a very poor local minimum and have to find a
# way to guide our optimizer towards a better minimum/better-looking pattern.
# We can show that frequency of the generated pattern increases with the
# image size because the convolutional filters have a fixed size but their
# relative size compared to the image decreases with increasing image
# resolution. In other words: assume that the pattern that is created always
# has roughly the same size measured in pixels. If we increase the image size,
# the relative size of the generated pattern will reduce and the pattern
# frequency increases.
# So what we want is the low-frequency pattern of a low-resolution example
# but with a high resolution. To do so, we can start with a low resolution to
# get a low-frequency pattern. After upscaling, the upscaled pattern has a
# lower frequency than what the optimizer would have generated if we had
# started at that larger image size with a random image. So when optimizing
# the pixel values in the next iteration we are at a better starting point
# and it appear to avoid poor local minima.

process = GradientAscent(
    model.features, apply_transforms, denormalize, img_size=224,
    upscaling_steps=1)
conv4_idx = 10
conv_4 = model.features[conv4_idx]
responses = process.get_filter_responses(
    conv_4, peaks, input_=None, lr=1, n_iter=30, blur=None)
visualize_filters(responses, conv4_idx, peaks, images_per_row=4)


############################################################################
# Visualizing saliency maps
# -------------------------
#
# Display most salient regions within images. By creating a saliency map for
# neural networks, we can gain some intuition on 'where the network is paying
# the most attention to' in an input image.

backprop = Backprop(model)
targets = {
    "great_grey_owl": 24,
    "peacock": 84,
    "toucan": 96}
for name, image in data.items():
    input_tensor = apply_transforms(image, size=224)
    target_class = targets[name]
    gradients = backprop.calculate_gradients(
        input_tensor, target_class, guided=True, use_gpu=False)
    max_gradients = gradients.max(dim=0, keepdim=True)[0]
    visualize_gradients(input_tensor, gradients, max_gradients)

plt.show()
