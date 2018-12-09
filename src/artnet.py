"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

Convolutional Network model.

@author: Zhenye Na
@references:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        A Neural Algorithm of Artistic Style. arXiv:1508.06576
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from gram import GramMatrix


class ArtNet(object):
    """CNN model."""

    def __init__(self, style, content, pastiche, args, content_weight=1, style_weight=1000):
        """CNN model initialization."""
        super(ArtNet, self).__init__()

        self.style = style
        self.content = content
        self.pastiche = nn.Parameter(pastiche.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight

        self.loss_network = models.vgg19(pretrained=True)
        self.transform_network = nn.Sequential(
            nn.ReflectionPad2d(40),
            nn.Conv2d(3, 32, 9, stride=1, padding=4),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.Conv2d(32, 3, 9, stride=1, padding=4),
        )

        self.optimizer = optim.Adam(
            self.transform_network.parameters(), lr=args.lr)
        self.grammatrix = GramMatrix()
        self.criterion = nn.MSELoss()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.net.cuda()
            self.grammatrix.cuda()

    def train(self):
        """Training process."""
        def closure():
            self.optimizer.zero_grad()

            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()

            content_loss = 0
            style_loss = 0

            i = 1

            def not_inplace(layer):
                return nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer

            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()

                pastiche, content, style = layer.forward(
                    pastiche), layer.forward(content), layer.forward(style)

                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)

                    # content layers
                    if name in self.content_layers:
                        content_loss += self.criterion(
                            pastiche * self.content_weight, content.detach() * self.content_weight)

                    # style layers
                    if name in self.style_layers:
                        pastiche_g, style_g = self.grammatrix.forward(
                            pastiche), self.grammatrix.forward(style)
                        style_loss += self.criterion(
                            pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

                if isinstance(layer, nn.ReLU):
                    i += 1

            total_loss = content_loss + style_loss
            total_loss.backward()

            return total_loss

        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if state['step'] >= 1024:
                    state['step'] = 1000
        self.optimizer.step(closure)

        return self.pastiche

    def batch_train(self, content):
        """Batch training."""
        self.optimizer.zero_grad()

        content = content.clone()
        style = self.style.clone()
        pastiche = self.loss_network.forward(content)

        content_loss = 0
        style_loss = 0

        i = 1

        def not_inplace(layer):
            return nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer

        for layer in list(self.loss_network.features):
            layer = not_inplace(layer)
            if self.use_cuda:
                layer.cuda()

            pastiche, content, style = layer.forward(
                pastiche), layer.forward(content), layer.forward(style)

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                # content layers
                if name in self.content_layers:
                    content_loss += self.criterion(
                        pastiche * self.content_weight, content.detach() * self.content_weight)

                # style layers
                if name in self.style_layers:
                    pastiche_g, style_g = self.grammatrix.forward(
                        pastiche), self.grammatrix.forward(style)
                    style_loss += self.criterion(
                        pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

            if isinstance(layer, nn.ReLU):
                i += 1

        total_loss = content_loss + style_loss
        total_loss.backward()

        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if state['step'] >= 1024:
                    state['step'] = 1000
        self.optimizer.step()

        return self.pastiche

    def save(self):
        """Save model."""
        torch.save({'state_dict': self.transform_network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'artnet.pth')
