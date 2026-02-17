from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def parse_ellipse_file(txt_path: Path) -> Dict[str, Tuple[float, float, float, float, float]]:
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    def looks_like_filename(s: str) -> bool:
        s = s.lower()
        return any(ext in s for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])

    mapping: Dict[str, Tuple[float, float, float, float, float]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        parts = re.split(r"[\s,\t]+", ln)

        # Variant A: filename on its own line
        if looks_like_filename(ln) and len(parts) == 1 and i + 1 < len(lines):
            fn = parts[0]
            nums = re.split(r"[\s\t]+", lines[i + 1].strip())
            if len(nums) >= 5:
                try:
                    a, b, ang, xc, yc = map(float, nums[:5])
                    mapping[fn] = (a, b, ang, xc, yc)
                    i += 2
                    continue
                except ValueError:
                    pass

        # Variant B: everything on one line
        if len(parts) >= 6 and looks_like_filename(parts[0]):
            fn = parts[0]
            try:
                a, b, ang, xc, yc = map(float, parts[1:6])
                mapping[fn] = (a, b, ang, xc, yc)
                i += 1
                continue
            except ValueError:
                pass

        i += 1

    if len(mapping) < 10:
        raise ValueError(f"Failed to parse ellipse file (too few entries): {txt_path}")
    return mapping


def ellipse_to_mask(h: int, w: int, ellipse: Tuple[float, float, float, float, float]) -> np.ndarray:
    a, b, angle_deg, x_c, y_c = ellipse
    # MATLAB image coordinates are 1-based; OpenCV uses 0-based.
    x_c0 = float(x_c) - 1.0
    y_c0 = float(y_c) - 1.0

    mask = np.zeros((h, w), dtype=np.uint8)
    axes = (int(round(a)), int(round(b)))
    center = (int(round(x_c0)), int(round(y_c0)))
    cv2.ellipse(mask, center, axes, float(angle_deg), 0.0, 360.0, 255, thickness=-1)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo masks from DAGM ellipse txt labels.")
    parser.add_argument("--images_dir", type=str, required=True, help="Folder containing images (e.g., Class1/Train)")
    parser.add_argument("--ellipse_txt", type=str, required=True, help="Ellipse parameter .txt file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder to save masks (e.g., Class1/Train/Label)")
    parser.add_argument("--ext", type=str, default=".PNG", help="Image extension to look for")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    ellipse_txt = Path(args.ellipse_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = parse_ellipse_file(ellipse_txt)

    saved = 0
    skipped = 0
    for img_path in sorted(images_dir.glob(f"*{args.ext}")):
        if img_path.name not in mapping:
            continue
        out_path = out_dir / img_path.name
        if out_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = ellipse_to_mask(h, w, mapping[img_path.name])
        cv2.imwrite(str(out_path), mask)
        saved += 1

    print(f"Done. Saved {saved} masks to {out_dir} (skipped existing: {skipped}).")


if __name__ == "__main__":
    main()
