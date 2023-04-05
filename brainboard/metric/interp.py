# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module to evaluate the linear interpolation of two model checkpoints.
"""

# Import
import numpy as np
from tqdm import tqdm
import torch


def eval_interpolation(model, state_1, state_2, loader, eval_fn, n_coeffs=3,
                       eval_kwargs=None):
    """ Eval an hybrid model composed of a linear interpolation of two
    checkpoints.

    Parameters
    ----------
    model: nn.Module
        a model archirtecture.
    state_1: dict
        first model parameter dictionary as generated with 'state_dict'.
    state_2: dict
        second model parameter dictionary as generated with 'state_dict'.
    loader: DataLoader
        the input data.
    eval_fn: @callable
        the evaluation function that implements a downstream task with
        the desired metric.
    n_coeffs: int, default 3
        the number of linear interpolation coefficents in [0, 1].
    eval_kwargs: dict, default None
        extra arguments passed to the evaluation function.

    Returns
    -------
    coeffs: list (n_coreffs, )
        the linear interpolation coefficents.
    metrics: list (n_coreffs, )
        the generated metrics for each interpolated model.
    """
    eval_kwargs = eval_kwargs or {}
    check_state_dicts(state_1, state_2)
    coeffs = np.linspace(0, 1, n_coeffs)
    metrics = []
    for idx in range(n_coeffs):
        model.load_state_dict(
            interpolate_state_dicts(state_1, state_2, coeffs[idx]))
        metrics.append(eval_fn(model, loader, **eval_kwargs))
    return coeffs, metrics


def check_state_dicts(state_1, state_2):
    """ Check wether states conteins the same nodes.

    Parameters
    ----------
    state_1: dict
        first model parameter dictionary as generated with 'state_dict'.
    state_2: dict
        second model parameter dictionary as generated with 'state_dict'.
    """
    assert sorted(state_1.keys()) == sorted(state_2.keys())


def interpolate_state_dicts(state_1, state_2, coeff):
    """ Interpolated two model weights.

    Parameters
    ----------
    state_1: dict
        first model parameter dictionary as generated with 'state_dict'.
    state_2: dict
        second model parameter dictionary as generated with 'state_dict'.
    coeff: float
        the linear interpolation coefficent in [0, 1].

    Returns
    -------
    state_hybrid: dict
        hybrid model parameter dictionary generated using linear interpolation.
    """
    return {key: (1 - coeff) * state_1[key] + coeff * state_2[key]
            for key in state_1.keys()}


def test_pred(loader, model, criterion):
    """ Define a prediction evaluation function.

    Parameters
    ----------
    loader: DataLoader
        a torch data loader.
    model: nn.Module
        the trained model.
    criterion: @callable
        a metric to evaluate.

    Returns
    -------
    metric: dict
        the top prediction accuracy ('top1') and input loss ('loss').
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx, (X, y) in enumerate(tqdm(loader, disable=None)):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            acc1 = calc_accuracy(y_pred, y, topk=(1, ))[0]
            losses.update(loss.item(), n=X.size(0))
            top1.update(acc1[0], n=X.size(0))
    return dict(top1=top1.avg, loss=losses.avg)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def calc_accuracy(output, target, topk=(1,)):
    """ Computes the accuracy (in %) over the k top predictions for the
    specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res
