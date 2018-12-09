"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

High level pipeline.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import os
import argparse

import torch
import torch.utils.data
import torchvision.datasets as datasets

from artnet import ArtNet
from utils import *


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--style_path', type=str, default="../styles/", help='path to style images')
    parser.add_argument('--content_path', type=str, default="../contents/", help='path to content images')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.01, help='default learning rate')
    parser.add_argument('--mode', type=str, default="single", help='mode for training')
    parser.add_argument('--epochs', type=int, default=35, help='number of epochs to train')

    parser.add_argument('--content_weight', type=int, default=1, help='weight of content images')
    parser.add_argument('--style_weight', type=int, default=1000, help='weight of style images')

    # parse the arguments
    args = parser.parse_args()

    return args


# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Content and style
style = image_loader(os.path.join(args.style_path, "starry_night.jpg")).type(dtype)
content = image_loader(os.path.join(args.content_path, "friends.jpg")).type(dtype)
pastiche = image_loader(os.path.join(args.content_path, "friends.jpg")).type(dtype)
pastiche.data = torch.randn(input.data.size()).type(dtype)


def main():
    """High level pipeline for Neural Style Transfer."""
    args = parse_args()

    if args.mode == "single":
        # create an ArtNet model
        model = ArtNet(style, content, pastiche, args)
        
        for i in range(args.epochs):
            pastiche = model.train()
        
            if i % 10 == 0:
                print("Iteration: %d" % (i))
                
                path = "outputs/%d.png" % (i)
                pastiche.data.clamp_(0, 1)
                save_image(pastiche, path)

    elif args.mode == "ITN":
        num_epochs = 3
        N = 4

        # create an ArtNet model
        model = ArtNet(style, content, pastiche, args)

        # content images
        coco = datasets.ImageFolder(root=args.content_path, transform=transforms)
        content_loader = torch.utils.data.DataLoader(coco, batch_size=N, shuffle=True, **kwargs)

        for epoch in range(num_epochs):
            for i, content_batch in enumerate(content_loader):
                iteration = epoch * i + i
                content_loss, style_loss, pastiches = ArtNet.batch_train(content_batch, style_batch)

                if i % 10 == 0:
                    print("Iteration: %d" % (iteration))
                    print("Content loss: %f" % (content_loss.data[0]))
                    print("Style loss: %f" % (style_loss.data[0]))

                if i % 500 == 0:
                    path = "outputs/%d_" % (iteration)
                    paths = [path + str(n) + ".png" for n in range(N)]
                    save_batch_images(pastiches, paths)

                    path = "outputs/content_%d_" % (iteration)
                    paths = [path + str(n) + ".png" for n in range(N)]
                    save_batch_images(content_batch, paths)
                    model.save()


if __name__ == '__main__':
    main()
