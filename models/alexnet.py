"""
@File  : alexnet.py
@Author: CodeCat
@Time  : 2021/7/9 9:50
"""
import torch.nn as nn
import torch
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weight=False):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        x = self.feature_extraction(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def get_alexnet(flag: bool, num_classes: int):
    if flag:
        net = models.alexnet(pretrained=True)
        num_input = net.classifier[-1].in_features
        cla_model = list(net.classifier.children())
        cla_model.pop()
        cla_model.append(nn.Linear(num_input, num_classes))
        net.classifier = nn.Sequential(*cla_model)
    else:
        net = AlexNet(num_classes=num_classes, init_weight=True)
    return net