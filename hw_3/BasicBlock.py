import torch
from torch import nn
from hw_2.ResNet18 import ResNet18

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn=nn.ReLU(inplace=True)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_fn,
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, resnet: ResNet18, use_layer4: bool = True):
        super().__init__()

        # H/2
        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        # H/4
        self.pool = resnet.maxpool

        # H/4
        self.enc2 = resnet.layer1

        # H/8
        self.enc3 = resnet.layer2

        # H/16
        self.enc4 = resnet.layer3

        self.use_layer4 = use_layer4
        if use_layer4:
            # H/32
            self.enc5 = resnet.layer4
            self.out_channels = [64, 64, 128, 256, 512]
        else:
            self.out_channels = [64, 64, 128, 256]

    def forward(self, x):
        x1 = self.enc1(x)             # H/2
        x2 = self.enc2(self.pool(x1)) # H/4
        x3 = self.enc3(x2)            # H/8
        x4 = self.enc4(x3)            # H/16

        if self.use_layer4:
            x5 = self.enc5(x4)        # H/32
            return [x1, x2, x3, x4, x5]

        return [x1, x2, x3, x4]