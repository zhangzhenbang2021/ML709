from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .blocks import ConvBNReLU, DoubleConv, UpBlock


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden_dim, kernel_size=1))
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Encoder(nn.Module):
    """Minimal MobileNetV2 encoder returning U-Net skip features."""

    def __init__(self, in_channels: int = 1, width_mult: float = 1.0):
        super().__init__()
        setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self.stem = ConvBNReLU(in_channels, input_channel, stride=2)

        in_ch = input_channel
        groups = []
        for t, c, n, s in setting:
            out_ch = int(c * width_mult)
            block_list = []
            for i in range(n):
                stride = s if i == 0 else 1
                block_list.append(InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=t))
                in_ch = out_ch
            groups.append(nn.Sequential(*block_list))
        self.groups = nn.ModuleList(groups)
        self.last_conv = ConvBNReLU(in_ch, last_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)          # 32 @ 1/2
        x = self.groups[0](x)     # 16 @ 1/2
        s1 = x
        x = self.groups[1](x)     # 24 @ 1/4
        s2 = x
        x = self.groups[2](x)     # 32 @ 1/8
        s3 = x
        x = self.groups[3](x)     # 64 @ 1/16
        x = self.groups[4](x)     # 96 @ 1/16
        s4 = x
        x = self.groups[5](x)     # 160 @ 1/32
        x = self.groups[6](x)     # 320 @ 1/32
        x = self.last_conv(x)     # 1280 @ 1/32
        return s1, s2, s3, s4, x


class UNetMobileNetV2(nn.Module):
    """U-Net with a MobileNetV2 encoder (binary segmentation by default)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, encoder_pretrained: bool = False):
        super().__init__()
        if encoder_pretrained:
            # TODO
            pass

        self.encoder = MobileNetV2Encoder(in_channels=in_channels)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up4 = UpBlock(in_ch=256, skip_ch=96, out_ch=128)
        self.up3 = UpBlock(in_ch=128, skip_ch=32, out_ch=64)
        self.up2 = UpBlock(in_ch=64, skip_ch=24, out_ch=32)
        self.up1 = UpBlock(in_ch=32, skip_ch=16, out_ch=16)

        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(16, 16),
        )
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(256, 1)  # 用 bottleneck(256通道)做分类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, s2, s3, s4, x = self.encoder(x)
        x = self.bottleneck(x)
        # classification branch
        cls_feat = self.cls_pool(x).flatten(1)     # (B,256)
        cls_logits = self.cls_head(cls_feat)       # (B,1)
        
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x)
        seg_logits = self.head(x)
        
        return seg_logits, cls_logits
