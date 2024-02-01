# adapted from: https://www.kaggle.com/code/leighplt/densenet121-pytorch
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T


class DensNet(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6):
        super().__init__()
        preloaded = torchvision.models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_classes, bias=True)
        del preloaded
        self.gradients = None
        self.activations = None

        # register hooks
        self.features[-2].register_full_backward_hook(self.backward_hook, prepend=False)
        self.features[-2].register_forward_hook(self.forward_hook, prepend=False)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out

    def forward_hook(self, module, input, output):
        self.activations = output



# TODO: add gradcam https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569
