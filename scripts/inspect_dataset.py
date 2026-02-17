from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.datasets.kolektor_sdd2 import build_kolektor_splits
from src.src_utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KolektorSDD2 dataset pairs and save a few overlays")
    parser.add_argument("--data_root", type=str, default="data/KolektorSDD2")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/inspect_kolektor")
    args = parser.parse_args()

    logger = get_logger()
    splits = build_kolektor_splits(data_root=args.data_root, val_ratio=0.1, seed=42, save_path=None)
    logger.info(f"Train pairs: {len(splits['train'])} | Val pairs: {len(splits['val'])} | Test pairs: {len(splits['test'])}")

    if not args.save_viz:
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save_overlay(img_path: str, mask_path: str, out_path: Path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay[:, :, 2] = cv2.max(overlay[:, :, 2], mask)
        cv2.imwrite(str(out_path), overlay)
        return True

    saved = 0
    for s in splits["train"][:50]:
        ok = save_overlay(s.img_path, s.mask_path, out / f"train_overlay_{Path(s.img_path).name}")
        if ok:
            saved += 1
        if saved >= 20:
            break

    saved_test = 0
    for s in splits["test"][:50]:
        ok = save_overlay(s.img_path, s.mask_path, out / f"test_overlay_{Path(s.img_path).name}")
        if ok:
            saved_test += 1
        if saved_test >= 10:
            break

    logger.info(f"Saved overlays: train={saved}, test={saved_test} to {out}")


if __name__ == "__main__":
    main()
