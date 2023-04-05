# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Interfaces to perform forward pass or backpropagation.
"""

# Imports
import warnings
import numpy as np
import torch
import torch.nn as nn


def set_trainable_attr(module, is_trainable):
    module.trainable = is_trainable
    for param in module.parameters():
        param.requires_grad = is_trainable


def apply_leaf(module, func):
    components = (
        module if isinstance(module, (list, tuple))
        else list(module.children()))
    if isinstance(module, nn.Module):
        func(module)
    if len(components) > 0:
        for layer in components:
            apply_leaf(layer, func)


def set_trainable(layer, is_trainable):
    """ Function to update the parameters that requires grad.

    Parameters
    ----------
    layer: nn.Module
        the layer to change recursively the parameters grad.
    is_trainable: bool
        the grad status.
    """
    apply_leaf(layer, lambda module: set_trainable_attr(module, is_trainable))


class Forward(object):
    """ Provides an interface to perform a forward pass with layer wise
    memory.
    """
    def __init__(self, model, activation_klass=nn.ReLU):
        """ Init class.

        Parameters
        ----------
        model: nn.module
            a neural network model.
        activation_klass: nn.Module, default nn.ReLU
            the activation class.
        """
        self.model = model
        self.activation_klass = activation_klass
        self.model.eval()
        set_trainable(self.model, False)
        self.activations = []
        self._register_activation_hooks()

    def get_activations(self, input_tensor, use_gpu=False):
        """ Get acctivations w.r.t. an input_tensor.

        Parameters
        ----------
        input_tensor: torch.Tensor (N, C, H, W)
            the input data.
        use_gpu: bool, default False
            optionally uses GPU if `torch.cuda.is_available()`.
        """
        self.activations = []
        if torch.cuda.is_available() and use_gpu:
            self.model = self.model.to("cuda")
            input_tensor = input_tensor.to("cuda")
        output = self.model(input_tensor)
        return self.activations

    def _register_activation_hooks(self):
        """ Record all activations.
        """
        def _record_activations(module, input_, output):
            self.activations.append(output)

        for _, module in self.model.named_modules():
            if isinstance(module, self.activation_klass):
                module.register_forward_hook(_record_activations)


class Backprop(object):
    """ Provides an interface to perform backpropagation.

    This class provids a way to calculate the gradients of a target class
    output w.r.t. an input image, by performing a single backprobagation.

    The gradients obtained can be used to visualise an image-specific class
    saliency map, which can gives some intuition on regions within the input
    image that contribute the most (and least) to the corresponding output.

    More details on saliency maps with backpropagation for gradient
    visualization: `Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/pdf/1312.6034.pdf>`_.

    More details on guided backprobagation: `Striving for Simplicity: The
    All Convolutional Net <https://arxiv.org/pdf/1412.6806.pdf>`_.
    """
    def __init__(self, model, conv_klass=nn.Conv2d, activation_klass=nn.ReLU,
                 layer=None):
        """ Init class.

        Parameters
        ----------
        model: nn.Module
            a neural network model.
        conv_class: nn.Module, default nn.Conv2d
            the type of convolution.
        activation_klass: nn.Module, default nn.ReLU
            the activation class.
        layer: nn.Conv, default None
            the target conv layer.
        """
        self.model = model
        self.conv_klass = conv_klass
        self.activation_klass = activation_klass
        self.layer = None
        self.model.eval()
        self.gradients = None
        self._register_conv_hook(layer)

    def calculate_gradients(self, input_tensor, target_class=None,
                            take_max=False, guided=False, use_gpu=False):
        """ Calculates gradients of the target_class output w.r.t. an
        input_tensor.

        The gradients is calculated for each channel. Optionally, the maximum
        gradients across channels can be returned.

        Parameters
        ----------
        input_tensor: torch.Tensor (N, C, H, W)
            the input data.
        target_class: int, default None
            the target class.
        take_max: bool, default False
            optionally takes the maximum gradients across channels.
        guided: bool, default False
            optionally performs guided backpropagation. See
            `Striving for Simplicity: The All Convolutional
            Net <https://arxiv.org/pdf/1412.6806.pdf>`_.
        use_gpu: bool, default False
            optionally uses GPU if `torch.cuda.is_available()`.

        Returns
        --------
        gradients: torch.Tensor (C, H, W).
            the computed gradients.
        """
        if guided:
            self.activation_outputs = []
            self._register_activation_hooks()

        input_tensor = input_tensor.detach().requires_grad_(True)
        if torch.cuda.is_available() and use_gpu:
            self.model = self.model.to("cuda")
            input_tensor = input_tensor.to("cuda")

        self.model.zero_grad()
        self.gradients = torch.zeros(input_tensor.shape)

        # Get a raw prediction value (logit) from the last linear layer
        output = self.model(input_tensor)
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]

        # Don't set the gradient target if the model is a binary classifier
        # i.e. has one class prediction
        if target_class is None or len(output.shape) == 1:
            target = None
        else:
            _, top_class = output.topk(k=1, dim=1)

            # Create a 2D tensor with shape (1, num_classes) and
            # set all element to zero
            target = torch.FloatTensor(1, output.shape[-1]).zero_()

            if torch.cuda.is_available() and use_gpu:
                target = target.to("cuda")

            if (target_class is not None) and (top_class != target_class):
                warnings.warn(UserWarning(
                    "The predicted class index {0} does not equal the "
                    "target class index {1}. Calculating the gradient "
                    "w.r.t. the predicted class.".format(
                        top_class.item(), target_class)))

            # Set the element at top class index to be 1
            target[0][top_class] = 1

        # Calculate gradients of the target class output w.r.t. input_tensor
        output.backward(gradient=target)

        # Detach the gradients from the graph and move to cpu
        gradients = self.gradients.detach().cpu()[0]

        # Take the maximum across channels
        if take_max:
            gradients = gradients.max(dim=0, keepdim=True)[0]

        return gradients

    def _register_conv_hook(self, layer=None):
        """ Record gradients of the first conv block or the specified conv
        layer.

        Parameters
        ----------
        layer: nn.Conv, default None
            the target conv layer.
        """
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        def _record_gradients_nocheck(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        if layer is not None:
            if type(layer) != self.conv_klass:
                raise TypeError("The layer must be {0}.".format(
                    self.conv_klass))
            layer.register_full_backward_hook(_record_gradients_nocheck)
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, self.conv_klass):
                    print(name, module)
                    module.register_full_backward_hook(_record_gradients)
                    break

    def _register_activation_hooks(self):
        """ Record all activations in order to clip gradients in the forward
        pass.
        """
        def _record_output(module, input_, output):
            self.activation_outputs.append(output)

        def _clip_gradients(module, grad_in, grad_out):
            activation_output = self.activation_outputs.pop()
            clippled_grad_out = grad_out[0].clamp(0.0)
            return (clippled_grad_out.mul(activation_output), )

        for name, module in self.model.named_modules():
            if isinstance(module, self.activation_klass):
                module.register_forward_hook(_record_output)
                module.register_full_backward_hook(_clip_gradients)
