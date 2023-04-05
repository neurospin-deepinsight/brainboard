# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides Centered Kernel Alignment (CKA) [Kornblith et al., 2019,
Similarity of neural network representations revisited].
"""

# Imports
import math
import numpy as np


def linear_cka(X, Y):
    """ Centered Kernel Alignment.

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
    hsic = linear_hsic(X, Y)
    var1 = np.sqrt(linear_hsic(X, X))
    var2 = np.sqrt(linear_hsic(Y, Y))
    return hsic / (var1 * var2)


def kernel_cka(X, Y, sigma=None):
    """ Centered Kernel Alignment.

    Parameters
    ----------
    X: array (n_samples, n_features)
        some representations.
    Y: array (n_samples, n_features)
        some representations.
    sigma: float, default None
        the RBF Gaussian kernel bandwidth, which controls the extent to
        which similarity of small distances is emphasized over large distances
        (if None this parameter is set to a fraction of the median distance
        between examples).

    Returns
    -------
    metric: float
        the computed metric.
    """
    hsic = kernel_hsic(X, Y, sigma)
    var1 = np.sqrt(kernel_hsic(X, X, sigma))
    var2 = np.sqrt(kernel_hsic(Y, Y, sigma))
    return hsic / (var1 * var2)


def linear_hsic(X, Y):
    """ Hilbert-Schmidt Independence Criterion: not invariant toisotropic
    scaling.

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
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def kernel_hsic(X, Y, sigma):
    """ Hilbert-Schmidt Independence Criterion: not invariant toisotropic
    scaling.

    Parameters
    ----------
    X: array (n_samples, n_features)
        some representations.
    Y: array (n_samples, n_features)
        some representations.
    sigma: float, default None
        the RBF Gaussian kernel bandwidth, which controls the extent to
        which similarity of small distances is emphasized over large distances
        (if None this parameter is set to a fraction of the median distance
        between examples).

    Returns
    -------
    metric: float
        the computed metric.
    """
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def centering(K):
    """ Centering input data.

    HKH are the same as KH: KH is the first centering, H(KH) do it a second
    time, results are the same with one time centering. H is the centering
    matrix.

    Parameters
    ----------
    K: array (N, N)
        some data.
    """
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)  


def rbf(X, sigma=None):
    """ RBF Gaussian kernel.

    Parameters
    ----------
    X: array (n_samples, n_features)
        some representations.
    sigma: float, default None
        the RBF Gaussian kernel bandwidth, which controls the extent to
        which similarity of small distances is emphasized over large distances
        (if None this parameter is set to a fraction of the median distance
        between examples).
    """
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX
