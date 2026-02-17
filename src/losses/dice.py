from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Expects:
      logits:  (B, 1, H, W)
      targets: (B, 1, H, W) in {0,1}
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()
