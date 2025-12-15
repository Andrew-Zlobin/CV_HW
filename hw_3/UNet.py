from hw_3.BasicBlock import DoubleConv, UpBlock
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        encoder: nn.Module | None = None,
        encoder_channels: list[int] | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.use_external_encoder = encoder is not None

        if not self.use_external_encoder:
            self.enc1 = DoubleConv(in_channels, base_channels)
            self.enc2 = DoubleConv(base_channels, base_channels * 2)
            self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
            self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

            self.pool = nn.MaxPool2d(2)

            self.encoder_channels = [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
            ]

            bottleneck_in = base_channels * 8

        else:
            self.encoder = encoder
            self.encoder_channels = encoder_channels

            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False

            bottleneck_in = encoder_channels[-1]

        self.bottleneck = DoubleConv(
            bottleneck_in,
            bottleneck_in * 2
        )

        dec_ch = bottleneck_in * 2
        self.up_blocks = nn.ModuleList()

        for skip_ch in reversed(self.encoder_channels):
            self.up_blocks.append(
                UpBlock(dec_ch, skip_ch, skip_ch)
            )
            dec_ch = skip_ch

        self.out_conv = nn.Conv2d(dec_ch, out_channels, kernel_size=1)

    def forward(self, x):
        if not self.use_external_encoder:
            skips = []

            x1 = self.enc1(x)
            skips.append(x1)
            x = self.pool(x1)

            x2 = self.enc2(x)
            skips.append(x2)
            x = self.pool(x2)

            x3 = self.enc3(x)
            skips.append(x3)
            x = self.pool(x3)

            x4 = self.enc4(x)
            skips.append(x4)
            x = self.pool(x4)

        else:
            feats = self.encoder(x)
            x = feats[-1]
            skips = feats[:-1]

        x = self.bottleneck(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)

