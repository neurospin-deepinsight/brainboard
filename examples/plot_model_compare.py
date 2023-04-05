# -*- coding: utf-8 -*-
"""
Model comparison
================

Credit: A Grigis

A simple example on how to comparer two instance of a network [Behnam
Neyshabur et al., What is being transferred in transfer learning?,
Neuripc 2020].
"""

############################################################################
# CKA
# ---
#
# Toy example on how to use the Centered Kernel Alignment(CKA).

import torch
import numpy as np
from brainboard.metric import linear_cka, kernel_cka, layer_at

X = np.random.randn(100, 64)
Y = np.random.randn(100, 64)
print(f"Linear CKA, between X and Y: {linear_cka(X, Y)}")
print(f"Linear CKA, between X and X: {linear_cka(X, X)}")
print(f"RBF Kernel CKA, between X and Y: {kernel_cka(X, Y)}")
print(f"RBF Kernel CKA, between X and X: {kernel_cka(X, X)}")


############################################################################
# Feature Similarity
# ------------------
#
# We use the Centered Kernel Alignment(CKA) as a measure of similarity between
# two output features in a layer of a network architecture given two instances
# of a network.

import copy
import torch
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from brainboard.metric import reset_weights, get_named_layers

# Old weights with accuracy 76.130%
model1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).double()
print(model1)
# New weights with accuracy 80.858%
model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).double()
# Random weights
model3 = reset_weights(copy.deepcopy(model2))
X = torch.randn(1, 3, 224, 224, dtype=torch.double)
scores = {}
for name in ("layer2", "layer3", "layer4"):
    data1, weights1 = layer_at(model1, name, X)
    data2, weights2 = layer_at(model2, name, X)
    data3, weights3 = layer_at(model3, name, X)
    n_samples, n_channels, s1, s2 = data1.shape
    data1 = data1.reshape((n_samples * s1 * s2, n_channels))
    data2 = data2.reshape((n_samples * s1 * s2, n_channels))
    data3 = data3.reshape((n_samples * s1 * s2, n_channels))
    scores[name] = [linear_cka(data1, data2), linear_cka(data1, data3)]
df = pd.DataFrame.from_dict(scores)
df["model"] = ["ResNet&ResNet", "ResNet&RandResNet"]
df.set_index("model", inplace=True)
print(df)


############################################################################
# Distance in parameter space
# ---------------------------
#
# In addition to feature similarity, we look into the distance between two
# networks in the parameter space. More specifically, we measure the l2
# distance between two network per module.

from brainboard.metric import paired_euclidean_dist

name = "avgpool"
scores = {}
data1, weights1 = layer_at(model1, name, X)
data2, weights2 = layer_at(model2, name, X)
data3, weights3 = layer_at(model3, name, X)
data1 = data1[..., 0, 0]
data2 = data2[..., 0, 0]
data3 = data3[..., 0, 0]
scores[name] = [paired_euclidean_dist(data1, data2),
                paired_euclidean_dist(data1, data3)]
df = pd.DataFrame.from_dict(scores)
df["model"] = ["ResNet&ResNet", "ResNet&RandResNet"]
df.set_index("model", inplace=True)
print(df)


############################################################################
# Performance barrier experiments
# -------------------------------
#
# Experiments of comparing the performance barrier interpolating the
# weights of two models and monitoring a common downstream task.

import torch.nn as nn
from brainboard.metric import eval_interpolation, test_pred
import matplotlib.pyplot as plt

state_1 = model1.state_dict()
state_2 = model2.state_dict()
X = torch.randn(2, 3, 224, 224, dtype=torch.double)
y = torch.from_numpy(np.array([1, 1]))
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(
      dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=False)

def eval_fn(model, loader):
    results = test_pred(loader, model, criterion=nn.CrossEntropyLoss())
    return {key: float(val) for key, val in results.items()}

coeffs, metrics = eval_interpolation(model1, state_1, state_2, loader, eval_fn,
                                     n_coeffs=6)
accs = [item["top1"] for item in metrics]
fig, ax = plt.subplots()
plt.plot(coeffs, accs, ".-")
tick_labels = np.array([f"{x:.2f}" for x in coeffs])
ax.set_xticks(coeffs, tick_labels)
ax.set_xlabel("linear interpolation coefficient")
ax.set_ylabel("accuracy %")
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.spines[["right", "top"]].set_visible(False)
plt.show()

