"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Helper functions.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import os
import torch
import torch.nn as nn

from PIL import Image
import torchvision.transforms as transforms

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu


loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])


# reconvert into PIL image
unloader = transforms.ToPILImage()


def image_loader(image_name):
    """Image loader."""
    image = Image.open(image_name)

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image


def save_image(tensor, path):
    """Save a single image."""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(os.path.join(path, "out.jpg"))


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    """Normalize input image."""

    def __init__(self, mean, std):
        """Initialization."""
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        """Forward pass."""
        return (img - self.mean) / self.std
