from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.unet_mobilenetv2 import UNetMobileNetV2
from src.src_utils import get_logger, load_checkpoint, load_yaml


def _preprocess(img_path: Path,  in_channels: int) -> tuple[torch.Tensor, np.ndarray]:
    if in_channels == 1:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(str(img_path))
        img_r = cv2.resize(img, (256, 640), interpolation=cv2.INTER_LINEAR)
        x = img_r.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x_t = torch.from_numpy(x)[None, None, :, :]  # (1,1,H,W)
        return x_t, img_r
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img, (256, 640), interpolation=cv2.INTER_LINEAR)
        x = img_r.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x_t = torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :]  # (1,3,H,W)
        # for overlay we also keep a grayscale version
        gray = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
        return x_t, gray


def _save_mask(out_path: Path, prob: np.ndarray, thr: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (prob >= thr).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask)


def _save_overlay(out_path: Path, img_gray: np.ndarray, prob: np.ndarray, thr: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (prob >= thr).astype(np.uint8) * 255
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], mask)  # red channel
    cv2.imwrite(str(out_path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for KolektorSDD2 defect segmentation")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best", help="best|last|/path/to/ckpt.pt")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--cls_thr", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_overlay", action="store_true")
    args = parser.parse_args()

    logger = get_logger()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    in_channels = int(cfg["data"].get("in_channels", 1))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    model = UNetMobileNetV2(
        in_channels=in_channels,
        num_classes=int(cfg["model"]["num_classes"]),
        encoder_pretrained=bool(cfg["model"].get("encoder_pretrained", False)),
    ).to(device)
    model.eval()

    if args.ckpt in {"best", "last"}:
        ckpt_path = run_dir / "checkpoints" / f"{args.ckpt}.pt"
    else:
        ckpt_path = Path(args.ckpt)
    load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, map_location=device)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    inp = Path(args.input)
    if inp.is_dir():
        imgs = sorted(list(inp.rglob("*.png")))
    else:
        imgs = [inp]

    out_dir = Path(args.out_dir)
    out_mask_dir = out_dir / "masks"
    out_overlay_dir = out_dir / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_rows = []

    with torch.no_grad():
        for p in tqdm(imgs, desc="Infer", leave=False):
            if p.name.endswith("_GT.png") or p.name.endswith("_gt.png"):
                continue
            x, gray = _preprocess(p, in_channels)
            x = x.to(device)

            seg_logits, cls_logits = model(x)

            # segmentation
            prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
            _save_mask(out_mask_dir / p.name, prob, args.thr)
            if args.save_overlay:
                _save_overlay(out_overlay_dir / p.name, gray, prob, args.thr)

            # classification (defect score)
            cls_prob = torch.sigmoid(cls_logits)[0, 0].item()
            cls_pred = 1 if cls_prob >= args.cls_thr else 0
            cls_rows.append((p.name, float(cls_prob), int(cls_pred)))

    logger.info(f"Saved masks to: {out_mask_dir}")
    if args.save_overlay:
        logger.info(f"Saved overlays to: {out_overlay_dir}")

    # save classification CSV
    import csv
    csv_path = out_dir / "cls_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "defect_prob", "defect_pred"])
        w.writerows(cls_rows)
    logger.info(f"Saved classification results to: {csv_path}")

if __name__ == "__main__":
    main()
