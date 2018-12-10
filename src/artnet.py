"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Convolutional Network model.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from utils import *
from loss import StyleLoss, ContentLoss


# change from "https" to "http" to avoid SSl Configuration error
# chnage back to "https" if needed
model_urls = {
    'vgg11': 'http://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'http://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'http://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'http://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'http://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'http://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'http://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'http://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArtNet(object):
    """Style Transfer model."""

    def __init__(self, args, normalization_mean, normalization_std,
                 style_img, content_img, content_weight=1, style_weight=1000000):
        """Style Transfer model initialization."""
        super(ArtNet, self).__init__()

        self.args = args

        self.style_img = style_img
        self.content_img = content_img

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # mean and std used for normalization
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

        # weights of content image and style image
        self.content_weight = args.content_weight if args else content_weight
        self.style_weight = args.style_weight if args else style_weight

        # initialize vgg19 pre-trained model
        self.model = vgg19(pretrained=True).features.to(device).eval()

    def train(self, input_img):
        """Training process."""
        print("==> Building the style transfer model ...")
        model, style_losses, content_losses = self._init_model_and_losses()

        # initialize LBFGS optimizer
        self._init_optimizer(input_img)

        print("==> Start training ...")

        for epoch in range(0, self.args.epochs):

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                self.optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                if epoch % 5 == 0:
                    print("Epoch {}: Style Loss : {:4f} Content Loss: {:4f}".format(
                        epoch, style_score.item(), content_score.item()))

                return style_score + content_score

            self.optimizer.step(closure)

        # clamp to correct data antries range from 0 to 1
        input_img.data.clamp_(0, 1)

        return input_img

    def _init_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        self.optimizer = optim.LBFGS([input_img.requires_grad_()])

    def _init_model_and_losses(self):
        cnn = copy.deepcopy(self.model)

        # normalization module
        normalization = Normalization(
            self.normalization_mean, self.normalization_std).to(device)

        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses


class VGG(nn.Module):
    """VGG model."""

    def __init__(self, features, num_classes=1000, init_weights=True):
        """Model initialization."""
        super(VGG, self).__init__()

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    """Create layers of vgg model.

    Args:
        cfg (list): configuration for vgg architecture
        batch_norm (bool): whether use batch normalization for vgg model

    Return:
        nn.Sequential(*layers)
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19(pretrained=False, **kwargs):
    """
    VGG 19-layer model (configuration "E").

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'],
                                                 model_dir='../'))
    return model
