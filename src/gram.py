"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Gram matrix.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    """Gram matrix."""

    def __init__(self):
        """Gram matrix initialization."""
        super(GramMatrix, self).__init__()

    def forward(self, inputs):
        """Forward pass."""
        a, b, c, d = inputs.size()
        features = inputs.view(a * b, c * d)
        g = torch.mm(features, features.t())

        return g.div(a * b * c * d)
