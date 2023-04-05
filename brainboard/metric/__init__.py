# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides common metrics.
"""

from .utils import reset_weights, get_named_layers, layer_at
from .cka import linear_cka, kernel_cka
from .interp import eval_interpolation, test_pred



def paired_euclidean_dist(X, Y):
    """ Paired Euclidean l2 distance.

    Parameters
    ----------
    X: array (n_samples, n_features)
        some representations.
    Y: array (n_samples, n_features)
        some representations.

    Returns
    -------
    metric: float
        the computed metric.
    """
    import numpy as np
    dX = X - Y
    norms = np.einsum("ij,ij->i", dX, dX)
    np.sqrt(norms, norms)
    return np.sum(norms)
