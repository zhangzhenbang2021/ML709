from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.kolektor_sdd2 import KolektorSDD2SegDataset, load_splits, build_kolektor_splits
from src.losses.dice import DiceLoss
from src.metrics.seg_metrics import SegmentationMeter
from src.models.unet_mobilenetv2 import UNetMobileNetV2
from src.src_utils import get_logger, load_checkpoint, load_yaml, seed_everything


def _save_pred_png(out_path: Path, prob: np.ndarray, thr: float = 0.5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (prob >= thr).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on KolektorSDD2")
    parser.add_argument("--run_dir", type=str, required=True, help="runs/<run_name>")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--ckpt", type=str, default="best", help="best|last|/path/to/ckpt.pt")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--thr", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    splits_path = run_dir / "splits.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")
    cfg = load_yaml(cfg_path)
    seed_everything(int(cfg["experiment"].get("seed", 42)))

    logger = get_logger()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Load splits (if missing, rebuild splits from config paths)
    if splits_path.exists():
        splits = load_splits(splits_path)
    else:
        splits = build_kolektor_splits(
            data_root=cfg["data"]["data_root"],
            val_ratio=float(cfg["data"]["val_ratio"]),
            seed=int(cfg["experiment"].get("seed", 42)),
            save_path=None,
        )

    samples = splits[args.split]
    if len(samples) == 0:
        raise RuntimeError(f"Split '{args.split}' is empty. Check your dataset path and naming convention.")

    in_channels = int(cfg["data"].get("in_channels", 1))
    ds = KolektorSDD2SegDataset(
        samples,
        in_channels=in_channels,
        augment=False,
        aug_cfg=cfg.get("aug", {}),
    )

    bs = args.batch_size if args.batch_size is not None else int(cfg["train"]["batch_size"])
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )

    model = UNetMobileNetV2(
        in_channels=in_channels,
        num_classes=int(cfg["model"]["num_classes"]),
        encoder_pretrained=bool(cfg["model"].get("encoder_pretrained", False)),
    ).to(device)

    # Load ckpt
    if args.ckpt in {"best", "last"}:
        ckpt_path = run_dir / "checkpoints" / f"{args.ckpt}.pt"
    else:
        ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, map_location=device)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss(smooth=float(cfg["loss"].get("dice_smooth", 1.0)))
    bce_w = float(cfg["loss"].get("bce_weight", 0.5))
    dice_w = float(cfg["loss"].get("dice_weight", 0.5))

    def loss_fn(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return bce_w * bce(logits, masks) + dice_w * dice(logits, masks)

    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    meter = SegmentationMeter(threshold=args.thr)

    model.eval()
    total_loss = 0.0
    n = 0

    out_pred_dir = run_dir / "preds" / f"{args.split}_{args.ckpt}"
    if args.save_preds:
        out_pred_dir.mkdir(parents=True, exist_ok=True)

    for imgs, masks, labels, meta in tqdm(loader, desc=f"Evaluating {args.split}", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)


        with torch.cuda.amp.autocast(enabled=amp):
            seg_logits, cls_logits = model(imgs)
            loss = loss_fn(seg_logits, masks)  
            meter.update(seg_logits, masks)
     
                        
        total_loss += float(loss.item())
        n += 1
        meter.update(seg_logits, masks)

        if args.save_preds:
            probs = torch.sigmoid(seg_logits).detach().cpu().numpy()  # (B,1,H,W)
            for i in range(probs.shape[0]):
                img_path = Path(meta["img_path"][i])
                out_path = out_pred_dir / img_path.name
                _save_pred_png(out_path, probs[i, 0], thr=args.thr)

    metrics = meter.compute()
    avg_loss = total_loss / max(n, 1)

    results = {
        "split": args.split,
        "ckpt": str(ckpt_path),
        "avg_loss": avg_loss,

        # --- overall (includes normal images; empty&empty treated as 1) ---
        "dice_all": metrics.get("dice_all", metrics.get("dice")),
        "iou_all": metrics.get("iou_all", metrics.get("iou")),

        # --- defect-only (only images with GT foreground) ---
        "dice_defect_only": metrics.get("dice_defect_only", None),
        "iou_defect_only": metrics.get("iou_defect_only", None),

        # --- normal-only (only images with GT empty) ---
        "dice_normal_only": metrics.get("dice_normal_only", None),
        "iou_normal_only": metrics.get("iou_normal_only", None),

        # --- normal false positive rate (normal images predicted with any foreground) ---
        "normal_fp_rate": metrics.get("normal_fp_rate", None),

        "num_samples": len(ds),
        "batch_size": bs,
        "in_channels": in_channels,
        "threshold": args.thr,
    }

    out_json = run_dir / f"eval_{args.split}_{args.ckpt}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved results: {out_json}")

    # 日志打印：同时看 overall / defect-only / 误报率
    logger.info(
        f"avg_loss={avg_loss:.4f} "
        f"dice_all={results['dice_all']:.4f} iou_all={results['iou_all']:.4f} "
        f"dice_def={results['dice_defect_only'] if results['dice_defect_only'] is not None else 'NA'} "
        f"iou_def={results['iou_defect_only'] if results['iou_defect_only'] is not None else 'NA'} "
        f"fp_norm={results['normal_fp_rate'] if results['normal_fp_rate'] is not None else 'NA'}"
    )

    if args.save_preds:
        logger.info(f"Saved preds to: {out_pred_dir}")

if __name__ == "__main__":
    main()
