import torch
from torch import nn
from hw_2.BasicBlock import BasicBlock


class ResNet18(nn.Module):
    def __init__(
        self
        , in_channel:int=64
        , num_classes:int=10
        , use_layer4:bool=True
        , tiny:bool=False
        , num_blocks:list[int]=[2,2,2,2]
        , channels: list[int]=[64, 128, 256, 512]
        , activation_fn=nn.ReLU(inplace=True)
    ):
        super(ResNet18, self).__init__()

        if tiny:
            self.conv1 = nn.Conv2d(
                3, in_channel, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                3, in_channel, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
            
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = activation_fn

        self.layer1 = self._add_basic_blocks(in_channel
                                             , channels[0]
                                             , blocks=num_blocks[0]
                                             , stride=1
                                             , activation_fn=activation_fn)
        self.layer2 = self._add_basic_blocks(channels[0]
                                             , channels[1]
                                             , blocks=num_blocks[1]
                                             , stride=2
                                             , activation_fn=activation_fn)
        self.layer3 = self._add_basic_blocks(channels[1]
                                             , channels[2]
                                             , blocks=num_blocks[2]
                                             , stride=2
                                             , activation_fn=activation_fn)

        if use_layer4:
            self.layer4 = self._add_basic_blocks(channels[2]
                                                 , channels[3]
                                                 , blocks=num_blocks[3]
                                                 , stride=2
                                                 , activation_fn=activation_fn)
            out_channels = channels[3]
        else:
            self.layer4 = nn.Identity()
            out_channels = channels[2]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)


    def _add_basic_blocks(self, in_channels, out_channels, blocks, stride, activation_fn):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(
            self.relu(
                self.bn1(
                    self.conv1(x))))

        x = self.layer4(
            self.layer3(
                self.layer2(
                    self.layer1(x))))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x