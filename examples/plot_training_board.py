# -*- coding: utf-8 -*-
"""
Training board
==============

Credit: A Grigis

A simple example on how to use a the lightweight online board tools. All
developement are built upon Visdom. If you prefer tensorboard, use the
dedicated `torch.utils.tensorboard.SummaryWriter` class.
"""

import os
import copy
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision
import brainboard

############################################################################
# Load the data
# -------------
#
# We will train a model to classify ants and bees as proposed in the Pytorch
# tutorials. 

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(inp)


data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = "/tmp/hymenoptera_data"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
    subprocess.check_call(
        ["wget", "https://download.pytorch.org/tutorial/hymenoptera_data.zip"],
        cwd="/tmp")
    subprocess.check_call(["unzip", "hymenoptera_data.zip"], cwd="/tmp")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ["train", "val"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=1)
              for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
device = torch.device("cpu")

inputs, classes = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

############################################################################
# Train the model
# ---------------
#
# Use a general function to train a model and display training metrics as well
# as classification intermediate results.

def train_model(model, criterion, optimizer, scheduler, num_epochs=25,
                prep_imgs=None):
    since = time.time()
    board = brainboard.Board(env="bees_experiment")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if hasattr(scheduler, "get_last_lr"):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = scheduler._last_lr[0]
        print("Epoch {0}/{1}: {2}".format(epoch, num_epochs - 1, lr))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            val_inputs = []
            val_classes = []
            for inputs, labels in dataloaders[phase]:
                if phase == "val":
                    val_inputs.append(inputs.numpy())
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    if phase == "val":
                       val_classes.extend([class_names[x] for x in preds])
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            board.update_plot("loss_{0}".format(phase), epoch, epoch_loss)
            board.update_plot("acc_{0}".format(phase), epoch, epoch_acc)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            # Display validation classification results
            if prep_imgs is not None and phase == "val":
                val_inputs = np.concatenate(val_inputs, axis=0)
                val_classes = np.asarray(val_classes)
                indices = np.random.randint(0, len(val_inputs), 4)
                print(val_inputs.shape, val_classes.shape, indices)
                out = prep_imgs(val_inputs[indices])
                board.update_image("prediction", out,
                                   title="-".join(val_classes[indices]))

            # Deep copy the best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def boardshow(inp):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for idx in range(len(inp)):
        _inp = inp[idx].transpose((1, 2, 0))
        _inp = std * _inp + mean
        _inp = _inp.transpose((2, 0, 1))
        _inp = np.clip(_inp, 0, 1) * 255
        inp[idx] = _inp
    return inp


model = models.vgg11(pretrained=True)
print(model)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                    num_epochs=5, prep_imgs=boardshow)
