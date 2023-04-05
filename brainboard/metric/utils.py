# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module with common functions.
"""

# Imports
import collections
import torch
import numpy as np


def get_named_layers(model):
    """ Function that returned a dictionary with named layers.

    Parameters
    ----------
    model: Net
        the network model.

    Returns
    -------
    layers: dict
        the named layers.
    """
    layers = {}
    for name, mod in model.named_modules():
        name = name.replace("ops.", "")
        layers[name] = mod
    return layers


def layer_at(model, layer_name, x, eval_fct=None, eval_kwargs=None):
    """ Access intermediate layers of pretrained network.

    Parameters
    ----------
    model: torch.nn.Module
        a network model.
    layer_name: str
        the layer name to be inspected.
    x: torch.Tensor
        an input tensor.
    eval_fct: callable, default None
        a specific function to evaluate the model, otherwise use the model
        evaluation function.
    eval_kwargs: dict, default None
        extra arguments passed to the evaluation function.

    Returns
    -------
    layer_data: torch.Tensor or list
        the tensor generated at the requested location.
    weight: torch.Tensor
        the layer associated weight.
    """
    layers = get_named_layers(model)
    layer = layers[layer_name]
    global hook_x

    def hook(module, inp, out):
        """ Define hook.
        """
        if isinstance(inp, collections.Sequence):
            inp_size = [item.data.size() for item in inp]
            inp_dtype = [item.data.type() for item in inp]
        else:
            inp_size = inp.data.size()
            inp_dtype = inp.data.type()
        if isinstance(out, collections.Sequence):
            out_size = [item.data.size() for item in out]
            out_dtype = [item.data.type() for item in out]
            out_data = [item.data for item in out]
        else:
            out_size = out.data.size()
            out_dtype = out.data.type()
            out_data = out.data
        print(
            "layer:", type(module),
            "\ninput:", type(inp),
            "\n   len:", len(inp),
            "\n   data size:", inp_size,
            "\n   data type:", inp_dtype,
            "\noutput:", type(out),
            "\n   data size:", out_size,
            "\n   data type:", out_dtype)
        global hook_x
        hook_x = out_data

    _hook = layer.register_forward_hook(hook)
    if eval_fct is not None:
        eval_fct(model, x, **eval_kwargs)
    else:
       model(x) 
    _hook.remove()

    if isinstance(hook_x, collections.Sequence):
        layer_data = [item.cpu().numpy() for item in hook_x]
    else:
        layer_data = hook_x.cpu().numpy()

    layer_weight = None
    if hasattr(layer, "weight"):
        layer_weight = layer.weight.detach().numpy()

    return layer_data, layer_weight


def reset_weights(model):
    """ Reset all the weights of a model.

    Parameters
    ----------
    model: torch.nn.Module
        a network model.
    """
    def weight_reset(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    model.apply(weight_reset)
    return model
