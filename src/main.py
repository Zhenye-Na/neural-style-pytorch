"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

High level pipeline.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

from __future__ import print_function

import os
import argparse
import torch


from artnet import ArtNet
from utils import *


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--style_path', type=str,
                        default="../styles/", help='path to style images')
    parser.add_argument('--content_path', type=str,
                        default="../contents/", help='path to content images')
    parser.add_argument('--output_path', type=str,
                        default="../outputs/", help='path to output images')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.01,
                        help='default learning rate')
    parser.add_argument('--mode', type=str, default="single",
                        help='mode for training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')

    parser.add_argument('--content_weight', type=int,
                        default=1, help='weight of content images')
    parser.add_argument('--style_weight', type=int,
                        default=1000000, help='weight of style images')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """High level pipeline for Neural Style Transfer."""
    args = parse_args()

    # CUDA Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    style_img = image_loader(os.path.join(
        args.style_path, "picasso.jpg")).to(device, torch.float)
    content_img = image_loader(os.path.join(
        args.content_path, "dancing.jpg")).to(device, torch.float)

    assert style_img.size() == content_img.size(
    ), "Style and Content image should be the same size"

    # Content and style
    # style = image_loader(os.path.join(args.style_path, "starry_night.jpg")).type(dtype)
    # content = image_loader(os.path.join(args.content_path, "friends.jpg")).type(dtype)
    # pastiche = image_loader(os.path.join(args.content_path, "friends.jpg")).type(dtype)
    # pastiche.data = torch.randn(pastiche.data.size()).type(dtype)

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    model = ArtNet(args,
                   normalization_mean,
                   normalization_std,
                   style_img,
                   content_img)

    output_img = model.train(content_img.clone())

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    save_image(output_img, args.output_path)


if __name__ == '__main__':
    main()
