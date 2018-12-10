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
import torch.utils.model_zoo as model_zoo

from gram import GramMatrix

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


class ArtNet(object):
    """Style Transfer model."""

    def __init__(self, style, content, pastiche, args, content_weight=1, style_weight=1000):
        """Style Transfer model initialization."""
        super(ArtNet, self).__init__()

        self.args = args

        # initializeb style, content and pastiche images
        self.style = style
        self.content = content
        self.pastiche = nn.Parameter(pastiche.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight

        self.loss_network = vgg19(pretrained=True).features.cuda().eval()
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

        self.grammatrix = GramMatrix()
        self.criterion = nn.MSELoss()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network = self.loss_network.cuda()
            self.grammatrix.cuda()

    def train(self):
        """Training process."""
        self.optimizer = optim.LBFGS([self.pastiche])

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

                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

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

        self.optimizer.step(closure)
        return self.pastiche

    def batch_train(self, content):
        """Batch training."""
        self.optimizer = optim.Adam(
            self.transform_network.parameters(), lr=self.args.lr)

        self.optimizer.zero_grad()

        content = content.clone()
        style = self.style.clone()
        pastiche = self.transform_network.forward(content)

        content_loss = 0
        style_loss = 0

        i = 1
        not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in list(self.loss_network.features):
            layer = not_inplace(layer)
            if self.use_cuda:
                layer.cuda()

            pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in self.content_layers:
                    content_loss += self.criterion(pastiche * self.content_weight,
                                                   content.detach() * self.content_weight)
                if name in self.style_layers:
                    pastiche_g, style_g = self.grammatrix.forward(pastiche), self.grammatrix.forward(style)
                    style_loss += self.criterion(pastiche_g * self.style_weight,
                                                 style_g.detach() * self.style_weight)

            if isinstance(layer, nn.ReLU):
                i += 1

        total_loss = content_loss + style_loss
        total_loss.backward()

        # for group in self.optimizer.param_groups:
        #     for p in group['params']:
        #         state = self.optimizer.state[p]
        #         if state['step'] >= 1024:
        #             state['step'] = 1000
        self.optimizer.step()

        return content_loss, style_loss, total_loss, self.pastiche

    def save(self):
        """Save model."""
        torch.save({'state_dict': self.transform_network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'artnet.pth')



class VGG(nn.Module):
    
    def __init__(self, features, num_classes=1000, init_weights=True):
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
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
