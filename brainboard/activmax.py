# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################


"""
Activation maximization via gradient ascent.
"""

# Imports
import copy
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn


class GradientAscent(object):
    """ Provides an interface for activation maximization via gradient descent.

    This class implements the gradient ascent algorithm in order to perform
    activation maximization with convolutional neural networks (CNN).

    More details on activation maximization: Erhan, Dumitru et al.,
    Visualizing Higher-Layer Features of a Deep Network, 2009.
    """
    def __init__(self, model, normalize, denormalize, img_size=224, lr=1.,
                 upscaling_steps=1, upscaling_factor=1.2, use_gpu=False):
        """ Init class.

        Parameters
        ----------
        model: nn.Module
            a neural network model.
        normalize: callable
            transforms to normalize input image.
        denormalize: callable
            transforms to denormalize generated tensor.
        img_size: int, default 224
            the size of the init random image.
        lr: float, default 0.1
            the gradient ascent.
        upscaling_steps: int, default 1
            optinally the number of upscaling steps: multi-resolution approach.
        upscaling_factor: float, default 1.2
            the zoom of each upscaling step.
        use_gpu: bool, default False
            optionally uses GPU if `torch.cuda.is_available()`.
        """
        self.model = model
        self.normalize = normalize
        self.denormalize = denormalize
        self._img_size = img_size
        self._lr = lr
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self._use_gpu = use_gpu
        self.num_layers = len(list(self.model.named_children()))
        self.activation = None
        self.gradients = None
        self.handlers = []

    def optimize(self, layer, filter_idx, input_=None, n_iter=30):
        """ Generates an image that maximally activates the target filter.

        Parameters
        ----------
        layer: nn.Conv2d
            the target Conv2d layer from which the filter to be chosen,
            based on `filter_idx`.
        filter_idx: int
            the index of the target filter.
        input_: array (H, W, C), default None
            create a random init image or use the specified image as init
            (for DeepDream).
        n_iter: int, default 30
            the number of iteration for the gradient ascent operation.

        Returns
        -------
        output list of torch.Tensor (n_iter, C, H, W)
            the filter response.
        """
        # Checks
        if type(layer) != nn.Conv2d:
            raise TypeError("The layer must be nn.Conv2d.")
        n_total_filters = layer.out_channels
        if (filter_idx < 0) or (filter_idx > n_total_filters):
            raise ValueError("Filter index must be between 0 and "
                             "{0}.".format(n_total_filters - 1))

        # Init input (as noise) if not provided
        if input_ is None:
            input_ = np.uint8(np.random.uniform(
                150, 180, (self._img_size, self._img_size, 3)))
        input_ = self.normalize(input_, size=None)
        if torch.cuda.is_available() and self.use_gpu:
            self.model = self.model.to("cuda")
            input_ = input_.to("cuda")

        # Remove previous hooks if any
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

        # Register hooks to record activation and gradients
        self.handlers.append(self._register_forward_hooks(layer, filter_idx))
        self.handlers.append(self._register_backward_hooks())

        # Init gradients
        self.gradients = torch.zeros(input_.shape)

        # Optimize
        return self._ascent(input_, n_iter)

    def get_filter_responses(self, layer, filter_idxs, input_=None, lr=1.,
                             n_iter=30, blur=None):
        """ Optimizes for the target layer/filter.

        Parameters
        ----------
        layer: nn.Conv2d
            the target Conv2d layer from which the filter to be chosen,
            based on `filter_idxs`.
        filter_idxs: list of int
            the indicies of the target filters.
        input_: array, default None
            create a random init image or use the specified image as init
            (for DeepDream).
        lr: float, default 0.1
            the gradient ascent.
        n_iter: int, default 30
            the number of iteration for the gradient ascent operation.
        blur: float, default None
            optionally blur the generated image at each optimization step.

        Returns
        -------
        responses: list of list of torch.Tensor (n_filters, n_iter, C, H, W)
            the filter responses.
        """
        if input_ is not None and self.upscaling_steps > 1:
            raise ValueError("Multi-resolution approach has only been "
                             "implemented for random init.")
        self._lr = lr
        responses = []
        for filter_idx in filter_idxs:
            filter_input = copy.deepcopy(input_)
            for upscaling_idx in range(self.upscaling_steps):
                _responses = self.optimize(
                    layer, filter_idx, input_=filter_input, n_iter=n_iter)
                filter_input = self.denormalize(_responses[-1])
                filter_input = filter_input.detach().cpu().numpy()[0]
                filter_input = [
                    ndimage.zoom(_img, self.upscaling_factor, order=3)
                    for _img in filter_input]
                if blur is not None:
                    filter_input = [
                        ndimage.gaussian_filter(_img, sigma=blur)
                        for _img in filter_input]
                filter_input = np.asarray(filter_input).transpose(1, 2, 0)
            responses.append(_responses)
        return responses

    def _register_forward_hooks(self, layer, filter_idx):
        """ Record mean activity at specified filter location.
        """
        def _record_activation(module, input_, output):
            self.activation = torch.mean(output[:, filter_idx, :, :])

        return layer.register_forward_hook(_record_activation)

    def _register_backward_hooks(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                return module.register_backward_hook(_record_gradients)

    def _ascent(self, x, n_iter):
        """ Maximize the mean activity.
        """
        output = []
        for idx in range(n_iter):
            self.model(x)
            self.activation.backward()
            self.gradients /= (torch.sqrt(torch.mean(
                torch.mul(self.gradients, self.gradients))) + 1e-5)
            x = x + self.gradients * self._lr
            output.append(x)
        return output
