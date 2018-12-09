"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Helper functions.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import scipy.misc
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

imsize = 256

loader = transforms.Compose([
    transforms.Scale(imsize),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


def image_loader(image_name):
    """Image loader."""
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image


def save_image(input, path):
    """Save a single image."""
    image = input.data.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    scipy.misc.imsave(path, image)


def save_batch_images(input, paths):
    """Save batch of images."""
    N = input.size()[0]
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view(3, imsize, imsize)
        image = unloader(image)
        scipy.misc.imsave(paths[n], image)
