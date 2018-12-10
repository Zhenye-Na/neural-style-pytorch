"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Loss functions.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    Content Loss.



    """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, inputs):
        self.loss = F.mse_loss(inputs, self.target)
        return inputs


class StyleLoss(nn.Module):
    """
    Style Loss.
    
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, inputs):
        G = gram_matrix(inputs)
        self.loss = F.mse_loss(G, self.target)
        return inputs


def gram_matrix(inputs):
    """Gram matrix."""
    a, b, c, d = inputs.size()

    # resise F_XL into \hat F_XL
    features = inputs.view(a * b, c * d)

    # compute the gram product
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)
