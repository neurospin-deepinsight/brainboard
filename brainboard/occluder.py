# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Occlusion patch-based saliency map.
"""

# Imports
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Occluder(object):
    """ Estimate an occlusion patch-based saliency map.

    Suppose that a ConvNet classifies an image as a dog. How can we be certain
    that it's actually picking up on the dog in the image as opposed to some
    contextual cues from the background or some other miscellaneous object?
    One way of investigating which part of the image some classification
    prediction is coming from is by plotting the probability of the class of
    interest (e.g. dog class) as a function of the position of an occluder
    object. That is, we iterate over regions of the image, set a patch of the
    image to be all zero, and look at the probability of the class.
    We can visualize the probability as a 2-dimensional heat map.

    In other word, when the occluder covers the face of the dog, thus
    probability deacrease, giving us some level of confidence that the dog's
    face is primarily responsible for the high classification score.
    Conversely, zeroing out other parts of the image is seen to have
    relatively negligible impact.

    More details on occluder: `Visualizing and Understanding Convolutional
    Networks <https://arxiv.org/pdf/1311.2901.pdf>`_.
    """
    def __init__(self, model, block_size=20):
        """ Init class.

        Parameters
        ----------
        model: nn.Module
            a neural network model.
        """
        self.model = model
        self.block_size = block_size
        self.model.eval()
        self.shape = None
        self.mask = None
        self.grid = None

    def _build_occ_mask(self, input_tensor):
        """ Build the occlusion mask.
        """
        n_channels, size_x, size_y = input_tensor.shape[-3:]
        if self.shape != (size_x, size_y):
            shift_x = int((size_x % self.block_size) / 2)
            shift_y = int((size_y % self.block_size) / 2)
            shift_block = self.block_size // 2
            indices_x = range(shift_x, size_x - shift_x, 1)
            indices_y = range(shift_y, size_y - shift_y, 1)
            self.grid = np.meshgrid(indices_x, indices_y)
            self.mask = np.ones((self.grid[0].size, size_x, size_y), dtype=int)
            for idx_x in range(self.grid[0].shape[0] - 1):
                for idx_y in range(self.grid[0].shape[1] - 1):
                    x, y = (self.grid[0][idx_x, idx_y],
                            self.grid[1][idx_x, idx_y])
                    idx = idx_x * (self.grid[0].shape[0] - 1) + idx_y
                    block = (slice(x - shift_block, x + shift_block, 1),
                             slice(y - shift_block, y + shift_block, 1))
                    self.mask[idx, block[0], block[1]] = 0
            self.mask = torch.from_numpy(self.mask)
            self.mask = self.mask.expand(
                n_channels, -1, -1, -1).transpose(0, 1)
            self.shape = (size_x, size_y)

    def get_saliency(self, input_tensor, target_class, batch_size=1000,
                     use_gpu=False):
        """ Calculates a saliency map using an occlusion patch-based strategy
        w.r.t. an input_tensor.

        Parameters
        ----------
        input_tensor: torch.Tensor (C, H, W)
            the input data.
        target_class: int, default None
            the target class.
        batch_size: int, default 1000
            the batch size.
        use_gpu: bool, default False
            optionally uses GPU if `torch.cuda.is_available()`.

        Returns
        -------
        probs: torch.Tensor (1, K)
            the predicted probabilities of belonging to each class.
        klass_probs_saliency: torch.Tensor (1, H, W).
            the computed probability of the class of interest as a function
            of the position of an occluder object.
        """
        if torch.cuda.is_available() and use_gpu:
            self.model = self.model.to("cuda")
            input_tensor = input_tensor.to("cuda")

        # Estimate true pred
        input_tensor = torch.unsqueeze(input_tensor, dim=0)
        with torch.no_grad():
            output = self.model(input_tensor)
        _, top_class = output.topk(1, dim=1)
        probs = torch.softmax(output, dim=1)
        if top_class != target_class:
            warnings.warn(UserWarning(
                "The predicted class index {0} does not equal the "
                "target class index {1}. Calculating the gradient "
                "w.r.t. the predicted class.".format(
                    top_class.item(), target_class)))

        # Occluder
        self._build_occ_mask(input_tensor)
        data = input_tensor.cpu() * self.mask
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            occluded_probs = []
            for occluded_x in dataloader:
                occluded_x = occluded_x[0]
                if torch.cuda.is_available() and use_gpu:
                    occluded_x = occluded_x.cuda(non_blocking=True)
                occluded_probs.append(
                    torch.softmax(self.model(occluded_x), dim=1))
            occluded_probs = torch.cat(occluded_probs, dim=0)

        # Format saliency map
        klass_probs_saliency = torch.ones(input_tensor.shape[-2:])
        for idx_x in range(self.grid[0].shape[0] - 1):
            for idx_y in range(self.grid[0].shape[1] - 1):
                x, y = (self.grid[0][idx_x, idx_y], self.grid[1][idx_x, idx_y])
                idx = idx_x * (self.grid[0].shape[0] - 1) + idx_y
                klass_probs_saliency[x, y] = occluded_probs[
                    idx, target_class].item()
        klass_probs_saliency = torch.unsqueeze(klass_probs_saliency, dim=0)

        return probs, klass_probs_saliency
