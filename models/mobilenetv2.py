"""
@File  : mobilenetv2.py
@Author: CodeCat
@Time  : 2021/5/20 15:39
"""
import torch
import torch.nn as nn
import torchvision


class ConvBNReLu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLu, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointWise conv
            layers.append(ConvBNReLu(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthWise conv
            ConvBNReLu(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointWise conv(linear)
            nn.Conv2d(in_channels=hidden_channel, out_channels=out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 169, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLu(in_channel=3, out_channel=input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channel=input_channel, out_channel=c, stride=stride, expand_ratio=t))
                input_channel = c
        features.append(ConvBNReLu(in_channel=input_channel, out_channel=last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


input = torch.randn(1, 3, 224, 224)
model = MobileNetV2()
out = model(input)
print(out.size())