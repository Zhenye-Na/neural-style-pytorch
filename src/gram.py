"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Gram matrix.

@author: Zhenye Na
"""

import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    """Gram matrix."""

    def __init__(self, arg):
        """Gram matrix initialization."""
        super(GramMatrix, self).__init__()

        self.arg = arg

    def forward(self, x):
        """Forward pass."""
        pass
