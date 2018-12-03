"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Convolutional Network model.

@author: Zhenye Na
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ArtNet(nn.Module):
    """CNN model."""

    def __init__(self, args):
        """CNN model initialization."""
        super(ArtNet, self).__init__()

        self.args = args
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """Forward pass."""
        pass
