# -*- coding: utf-8 -*-
"""
Model summary
=============

Credit: A Grigis

A simple example on how to use a model summary tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import brainboard
import brainboard.plotting as plotting 

############################################################################
# Display model summary
# ---------------------
#
# Similar to Tensorflow's model.summary() API to view tha layers of a model.
# This function is helpful while debugging a network.

model = torchvision.models.vgg11()
brainboard.summary(
    model, input_size=(1, 3, 224, 224), col_width=16,
    col_names=["input_size", "kernel_size", "output_size", "num_params",
               "mult_adds"])

#############################################################################
# Display the model
# -----------------
#
# Display the model as a graph.

model = torchvision.models.vgg11()
plotting.plot_net(model, input_size=(1, 3, 224, 224), outfile=None)
