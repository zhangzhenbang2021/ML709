from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class DoubleConv(nn.Module):
    """(Conv->BN->ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def center_crop(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Center-crop src spatially to match tgt (N,C,H,W)."""
    _, _, h, w = src.shape
    _, _, th, tw = tgt.shape
    if h == th and w == tw:
        return src
    y1 = max(0, (h - th) // 2)
    x1 = max(0, (w - tw) // 2)
    return src[:, :, y1 : y1 + th, x1 : x1 + tw]


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = center_crop(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
