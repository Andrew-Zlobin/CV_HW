import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self
                 , in_channels
                 , out_channels
                 , stride=1
                 , activation_fn=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels
            , out_channels
            , kernel_size=3
            , stride=stride
            , padding=1
            , bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels
            , out_channels
            , kernel_size=3
            , stride=1
            , padding=1
            , bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels
                    , out_channels
                    , kernel_size=1
                    , stride=stride
                    , bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.relu = activation_fn

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
