from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch

@torch.no_grad()
def per_sample_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    logits:  (B,1,H,W)
    targets: (B,1,H,W) in {0,1}
    returns:
      dice: (B,)
      iou:  (B,)
      gt_has_fg: (B,) bool
      pred_has_fg:(B,) bool
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=targets.dtype)

    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)
    union = pred_sum + target_sum - intersection

    gt_has_fg = target_sum > 0
    pred_has_fg = pred_sum > 0
    eps = 1e-7

    # Dice：空&空 => 1
    dice = torch.where(
        (pred_sum + target_sum) == 0,
        torch.ones_like(intersection),
        (2.0 * intersection) / (pred_sum + target_sum + eps),
    )

    # IoU：空&空 => 1
    iou = torch.where(
        union == 0,
        torch.ones_like(intersection),
        intersection / (union + eps),
    )

    return dice, iou, gt_has_fg, pred_has_fg


@dataclass
class SegmentationMeter:
    threshold: float = 0.5

    sum_dice_all: float = 0.0
    sum_iou_all: float = 0.0
    n_all: int = 0

    sum_dice_def: float = 0.0
    sum_iou_def: float = 0.0
    n_def: int = 0

    sum_dice_norm: float = 0.0
    sum_iou_norm: float = 0.0
    n_norm: int = 0

    norm_fp: int = 0  # 正常图中预测出前景的张数

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        dice, iou, gt_has_fg, pred_has_fg = per_sample_metrics_from_logits(
            logits, targets, threshold=self.threshold
        )

        # all
        self.sum_dice_all += float(dice.mean().item())
        self.sum_iou_all += float(iou.mean().item())
        self.n_all += 1

        # defect-only & normal-only：按样本拆开累加（用 sum / count 更精确）
        # 先 flatten 成 batch 维度
        dice_b = dice.detach().cpu()
        iou_b = iou.detach().cpu()
        gt_has_fg_b = gt_has_fg.detach().cpu()
        pred_has_fg_b = pred_has_fg.detach().cpu()

        if gt_has_fg_b.any():
            d = dice_b[gt_has_fg_b]
            j = iou_b[gt_has_fg_b]
            self.sum_dice_def += float(d.sum().item())
            self.sum_iou_def += float(j.sum().item())
            self.n_def += int(d.numel())

        if (~gt_has_fg_b).any():
            d = dice_b[~gt_has_fg_b]
            j = iou_b[~gt_has_fg_b]
            self.sum_dice_norm += float(d.sum().item())
            self.sum_iou_norm += float(j.sum().item())
            self.n_norm += int(d.numel())

            # normal false positive：正常图预测出了前景
            self.norm_fp += int((pred_has_fg_b[~gt_has_fg_b]).sum().item())

    def compute(self) -> Dict[str, float]:
        out = {}
        out["dice_all"] = self.sum_dice_all / max(self.n_all, 1)
        out["iou_all"] = self.sum_iou_all / max(self.n_all, 1)

        out["dice_defect_only"] = self.sum_dice_def / max(self.n_def, 1)
        out["iou_defect_only"] = self.sum_iou_def / max(self.n_def, 1)

        out["dice_normal_only"] = self.sum_dice_norm / max(self.n_norm, 1)
        out["iou_normal_only"] = self.sum_iou_norm / max(self.n_norm, 1)

        out["normal_fp_rate"] = self.norm_fp / max(self.n_norm, 1)  # 正常图误报率（按张）
        out["n_defect_imgs"] = float(self.n_def)
        out["n_normal_imgs"] = float(self.n_norm)
        return out

    def reset(self) -> None:
        self.sum_dice_all = 0.0
        self.sum_iou_all = 0.0
        self.n_all = 0

        self.sum_dice_def = 0.0
        self.sum_iou_def = 0.0
        self.n_def = 0

        self.sum_dice_norm = 0.0
        self.sum_iou_norm = 0.0
        self.n_norm = 0

        self.norm_fp = 0

