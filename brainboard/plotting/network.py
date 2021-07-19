# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides tools to display a graph.
"""

# Imports
import os
import tempfile
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchviz import make_dot


def plot_net(model, input_size, outfile=None):
    """ Plot the network.

    Parameters
    ----------
    model: nn.Module
        the network to be displayed.
    input_size: list of int
        the shape of a classical input batch dataset.
    outfile: str, default None
        the file containing the graph image.
    """
    x = torch.randn(input_size)
    graph = make_dot(model(x), params=dict(model.named_parameters()))
    graph.format = "png"
    if outfile is None:
        with tempfile.TemporaryDirectory() as dirpath:
            graph.render(directory=dirpath, filename="model", view=False)
            img = mpimg.imread(os.path.join(dirpath, "model.png"))
            fig = plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.show()
    else:
        dirpath = os.path.dirname(outfile)
        basename = os.path.basename(outfile).split(".")[0]
        graph.render(directory=dirpath, filename=basename, view=False)
