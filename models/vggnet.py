"""
@File  : vggnet.py
@Author: CodeCat
@Time  : 2021/7/9 10:32
"""
import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weight=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weight:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_features(cfgs: list):
    layers = []
    in_channels = 3
    for v in cfgs:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 156, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(num_classes=1000, init_weight=False):
    model = VGG(make_features(cfgs['vgg11']), num_classes=num_classes, init_weight=init_weight)
    return model


def vgg13(num_classes=1000, init_weight=False):
    model = VGG(make_features(cfgs['vgg13']), num_classes=num_classes, init_weight=init_weight)
    return model


def vgg16(num_classes=1000, init_weight=False):
    model = VGG(make_features(cfgs['vgg16']), num_classes=num_classes, init_weight=init_weight)
    return model


def vgg19(num_classes=1000, init_weight=False):
    model = VGG(make_features(cfgs['vgg19']), num_classes=num_classes, init_weight=init_weight)
    return model


def get_vgg16(flag, num_classes):
    if flag:
        net = models.vgg16(pretrained=True)
        num_input = net.classifier[-1].in_features
        cla_model = list(net.classifier.children())
        cla_model.pop()
        cla_model.append(nn.Linear(num_input, num_classes))
        net.classifier = nn.Sequential(*cla_model)
    else:
        net = vgg16(num_classes=num_classes, init_weight=True)

    return net
