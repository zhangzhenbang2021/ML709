from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    img_path: str
    mask_path: Optional[str] = None
    # If mask_path is None but ellipse parameters exist, we can create a pseudo mask on-the-fly.
    # (semi-major, semi-minor, rotation_deg, x_center_matlab, y_center_matlab)
    ellipse: Optional[Tuple[float, float, float, float, float]] = None
    is_defective: Optional[int] = None  # 0/1 if available


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_class_dir(data_root: str | Path, class_id: int) -> Path:
    data_root = Path(data_root)
    # Typical: data_root/Class1 or data_root/Class01
    candidates = [
        data_root / f"Class{class_id}",
        data_root / f"Class{class_id:02d}",
        data_root / f"class{class_id}",
        data_root / f"class{class_id:02d}",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # Maybe user passed the class dir directly
    if data_root.name.lower() in {f"class{class_id}".lower(), f"class{class_id:02d}".lower()}:
        return data_root
    # Fallback: search one level down
    for p in data_root.glob("Class*"):
        if p.is_dir() and re.fullmatch(r"Class0*%d" % class_id, p.name):
            return p
    raise FileNotFoundError(
        f"Could not find Class{class_id} under {data_root}. "
        "Expected something like <data_root>/Class1/Train/*.PNG"
    )


def _collect_images(split_dir: Path) -> List[Path]:
    """Collect image files under a split dir (e.g. Train/ or Test/).

    Heuristics:
      - excludes anything inside a folder that has 'Label' in its path
      - prefers images directly in split_dir if that seems plausible
    """
    imgs: List[Path] = []
    for p in split_dir.rglob("*"):
        if not p.is_file():
            continue
        if "label" in [x.lower() for x in p.parts]:
            continue
        if _is_image_file(p):
            imgs.append(p)

    # Prefer direct children if present (common structure)
    top_level = [p for p in imgs if p.parent == split_dir]
    if len(top_level) >= max(50, len(imgs) // 2):
        imgs = top_level

    return sorted(imgs, key=lambda x: x.name)


def _read_grayscale(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return arr


def _resize_image_and_mask(img: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def _apply_aug(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    hflip_p: float,
    vflip_p: float,
    rot90: bool,
    brightness_jitter: float,
    contrast_jitter: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # img, mask are HxW uint8
    if hflip_p > 0 and random.random() < hflip_p:
        img = np.ascontiguousarray(np.fliplr(img))
        mask = np.ascontiguousarray(np.fliplr(mask))
    if vflip_p > 0 and random.random() < vflip_p:
        img = np.ascontiguousarray(np.flipud(img))
        mask = np.ascontiguousarray(np.flipud(mask))
    if rot90:
        k = random.randint(0, 3)
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))

    # Light photometric jitter (only on image)
    if brightness_jitter and brightness_jitter > 0:
        b = brightness_jitter
        factor = 1.0 + random.uniform(-b, b)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if contrast_jitter and contrast_jitter > 0:
        c = contrast_jitter
        factor = 1.0 + random.uniform(-c, c)
        mean = float(img.mean())
        img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    return img, mask


def _ellipse_to_mask(h: int, w: int, ellipse: Tuple[float, float, float, float, float]) -> np.ndarray:
    # ellipse = (a, b, angle_deg, x_center_matlab, y_center_matlab)
    a, b, angle_deg, x_c, y_c = ellipse

    # MATLAB image coordinates are 1-based; OpenCV uses 0-based.
    x_c0 = float(x_c) - 1.0
    y_c0 = float(y_c) - 1.0

    mask = np.zeros((h, w), dtype=np.uint8)

    # OpenCV expects axes lengths as half-sizes in pixels.
    axes = (int(round(a)), int(round(b)))
    center = (int(round(x_c0)), int(round(y_c0)))

    cv2.ellipse(
        img=mask,
        center=center,
        axes=axes,
        angle=float(angle_deg),
        startAngle=0.0,
        endAngle=360.0,
        color=255,
        thickness=-1,  # filled
        lineType=cv2.LINE_8,
    )
    return mask


def _parse_labels_txt(labels_txt: Path) -> Dict[str, Tuple[int, Optional[str]]]:
    """Parse DAGM 'Labels.txt' mapping file.

    The dataset documentation describes a file in the Label folder:

        1
        [id] [0/1] [raw_filename] 0 [label_filename or 0]
        ...

    Returns:
        raw_filename -> (is_defective, label_filename_or_none)
    """
    lines = labels_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    mapping: Dict[str, Tuple[int, Optional[str]]] = {}

    if not lines:
        return mapping

    start = 0
    if lines[0].strip().isdigit():
        start = 1

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue
        is_def = int(parts[1])
        raw_fn = parts[2]
        lbl_fn = parts[4]
        if lbl_fn == "0":
            lbl_fn = ""
        mapping[raw_fn] = (is_def, lbl_fn or None)

    return mapping


def _parse_csv_list(csv_path: Path) -> List[Tuple[str, Optional[str], int]]:
    """Parse a CSV with columns: input_image_name, image_mask_name, label."""
    rows: List[Tuple[str, Optional[str], int]] = []
    with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = r.get("input_image_name") or r.get("image") or r.get("img") or ""
            mask = r.get("image_mask_name") or r.get("mask") or ""
            lbl = r.get("label") or r.get("is_defective") or "0"
            if not img:
                continue
            mask = mask.strip()
            rows.append((img.strip(), (mask if mask and mask != "0" else None), int(lbl)))
    return rows


def _try_parse_ellipse_file(txt_path: Path) -> Dict[str, Tuple[float, float, float, float, float]]:
    """Try to parse an ellipse-parameter .txt file (weak labels).

    Supported variants:

    A) Two-line record:
        <filename>
        <a> <b> <angle> <x_center> <y_center>

    B) One-line record:
        <filename> <a> <b> <angle> <x_center> <y_center>

    Returns:
        filename -> (a, b, angle_deg, x_center, y_center)
    """
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return {}

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

    # Heuristic: if too few entries, probably not the right file
    if len(mapping) < 10:
        return {}
    return mapping


def _discover_split_samples(split_dir: Path) -> List[Sample]:
    """Build samples for a Train/ or Test/ directory."""

    # 1) Preferred: CSV lists (some preprocessing pipelines)
    csv_list = None
    for name in ["train_list.csv", "test_list.csv", "list.csv"]:
        p = split_dir / name
        if p.exists():
            csv_list = p
            break

    if csv_list is not None:
        rows = _parse_csv_list(csv_list)
        label_dir = split_dir / "Label"
        samples: List[Sample] = []
        for img_name, mask_name, lbl in rows:
            img_path = split_dir / img_name
            if not img_path.exists():
                img_path = split_dir / Path(img_name).name
            mask_path = str(label_dir / mask_name) if mask_name else None
            samples.append(Sample(img_path=str(img_path), mask_path=mask_path, ellipse=None, is_defective=lbl))
        return samples

    # 2) Labels.txt mapping (Zenodo / HCI distribution)
    labels_txt = None
    for cand in [split_dir / "Label" / "Labels.txt", split_dir / "Label" / "labels.txt"]:
        if cand.exists():
            labels_txt = cand
            break

    label_dir = split_dir / "Label"
    labels_map: Dict[str, Tuple[int, Optional[str]]] = {}
    if labels_txt is not None:
        labels_map = _parse_labels_txt(labels_txt)

    # 3) Ellipse params file (MPI distribution)
    ellipse_map: Dict[str, Tuple[float, float, float, float, float]] = {}
    if not labels_map:
        # Try any .txt in split_dir or its parent
        txt_candidates = list(split_dir.glob("*.txt")) + list((split_dir.parent).glob("*.txt"))
        for t in txt_candidates:
            if t.name.lower() in {"readme.txt"}:
                continue
            ellipse_map = _try_parse_ellipse_file(t)
            if ellipse_map:
                break

    # 4) Collect images
    images = _collect_images(split_dir)
    samples: List[Sample] = []

    for img_path in images:
        img_name = img_path.name

        # If Labels.txt exists, use it
        if labels_map:
            if img_name in labels_map:
                is_def, lbl_fn = labels_map[img_name]
                if lbl_fn and (label_dir / lbl_fn).exists():
                    samples.append(
                        Sample(img_path=str(img_path), mask_path=str(label_dir / lbl_fn), ellipse=None, is_defective=is_def)
                    )
                else:
                    samples.append(Sample(img_path=str(img_path), mask_path=None, ellipse=None, is_defective=is_def))
            else:
                samples.append(Sample(img_path=str(img_path), mask_path=None, ellipse=None, is_defective=0))
            continue

        # Else: if label images exist with same basename, use them
        same_name_mask = label_dir / img_name
        if same_name_mask.exists():
            samples.append(Sample(img_path=str(img_path), mask_path=str(same_name_mask), ellipse=None, is_defective=1))
            continue

        # Else: if ellipse params exist, use them
        if img_name in ellipse_map:
            samples.append(Sample(img_path=str(img_path), mask_path=None, ellipse=ellipse_map[img_name], is_defective=1))
        else:
            samples.append(Sample(img_path=str(img_path), mask_path=None, ellipse=None, is_defective=0))

    return samples


def build_dagm_splits(
    *,
    data_root: str | Path,
    class_id: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
) -> Dict[str, List[Sample]]:
    """Create reproducible train/val/test splits from a raw DAGM folder.

    If Train/ and Test/ exist, it uses them; validation is a split of Train/.
    """
    class_dir = _find_class_dir(data_root, class_id=class_id)

    train_dir = class_dir / "Train"
    test_dir = class_dir / "Test"

    if train_dir.exists() and test_dir.exists():
        train_samples = _discover_split_samples(train_dir)
        test_samples = _discover_split_samples(test_dir)
    else:
        # Fallback: treat class_dir as a single pool, no separate test split
        train_samples = _discover_split_samples(class_dir)
        test_samples = []

    # Split train into train/val
    rng = random.Random(seed)
    idx = list(range(len(train_samples)))
    rng.shuffle(idx)
    n_val = int(round(len(train_samples) * float(val_ratio)))
    val_idx = set(idx[:n_val])

    train_list: List[Sample] = []
    val_list: List[Sample] = []
    for i, s in enumerate(train_samples):
        if i in val_idx:
            val_list.append(s)
        else:
            train_list.append(s)

    splits = {"train": train_list, "val": val_list, "test": test_samples}

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: [
                {
                    "img_path": s.img_path,
                    "mask_path": s.mask_path,
                    "ellipse": list(s.ellipse) if s.ellipse is not None else None,
                    "is_defective": s.is_defective,
                }
                for s in v
            ]
            for k, v in splits.items()
        }
        save_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    return splits


def load_splits(path: str | Path) -> Dict[str, List[Sample]]:
    """Load splits saved by build_dagm_splits(save_path=...)."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Sample]] = {}
    for split, arr in obj.items():
        samples: List[Sample] = []
        for d in arr:
            ellipse = d.get("ellipse", None)
            samples.append(
                Sample(
                    img_path=d["img_path"],
                    mask_path=d.get("mask_path", None),
                    ellipse=tuple(ellipse) if ellipse is not None else None,
                    is_defective=d.get("is_defective", None),
                )
            )
        out[split] = samples
    return out


class DAGMSegDataset(Dataset):
    """DAGM weak-label segmentation dataset (binary mask).

    Returns:
      image: float32 tensor (1, H, W), normalized to [-1, 1]
      mask:  float32 tensor (1, H, W), in {0,1}
      meta:  dict with paths and is_defective flag
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        img_size: int = 512,
        augment: bool = False,
        aug_cfg: Optional[Dict] = None,
    ):
        self.samples = list(samples)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.aug_cfg = aug_cfg or {}

        self.hflip_p = float(self.aug_cfg.get("hflip_p", 0.5))
        self.vflip_p = float(self.aug_cfg.get("vflip_p", 0.0))
        self.rot90 = bool(self.aug_cfg.get("rot90", True))
        self.brightness_jitter = float(self.aug_cfg.get("brightness_jitter", 0.0))
        self.contrast_jitter = float(self.aug_cfg.get("contrast_jitter", 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = _read_grayscale(s.img_path)  # uint8 HxW
        h, w = img.shape[:2]

        if s.mask_path is not None and Path(s.mask_path).exists():
            mask = _read_grayscale(s.mask_path)
        elif s.ellipse is not None:
            mask = _ellipse_to_mask(h, w, s.ellipse)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Resize
        img, mask = _resize_image_and_mask(img, mask, self.img_size)

        # Augment
        if self.augment:
            img, mask = _apply_aug(
                img,
                mask,
                hflip_p=self.hflip_p,
                vflip_p=self.vflip_p,
                rot90=self.rot90,
                brightness_jitter=self.brightness_jitter,
                contrast_jitter=self.contrast_jitter,
            )

        # Normalize image to [-1, 1]
        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - 0.5) / 0.5

        mask_f = (mask.astype(np.float32) / 255.0)
        mask_f = (mask_f > 0.5).astype(np.float32)

        img_t = torch.from_numpy(img_f)[None, :, :]
        mask_t = torch.from_numpy(mask_f)[None, :, :]

        meta = {
            "img_path": s.img_path,
            "mask_path": s.mask_path,
            "is_defective": s.is_defective,
        }
        return img_t, mask_t, meta
