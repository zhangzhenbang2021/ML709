from __future__ import annotations

import json
import random
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
    mask_path: str


def _is_png(p: Path) -> bool:
    return p.suffix.lower() == ".png"


def _collect_pairs(split_dir: Path) -> List[Sample]:
    """Collect (image, mask) pairs in a folder using naming convention:
       xx.png and xx_GT.png

    Supports nested subfolders by rglob.
    """
    split_dir = Path(split_dir)
    imgs = []
    for p in split_dir.rglob("*.png"):
        if not p.is_file():
            continue
        if p.name.endswith("_GT.png"):
            continue
        imgs.append(p)

    samples: List[Sample] = []
    for img_path in sorted(imgs, key=lambda x: str(x)):
        mask_path = img_path.with_name(img_path.stem + "_GT.png")
        if mask_path.exists():
            samples.append(Sample(img_path=str(img_path), mask_path=str(mask_path)))
        else:
            # Some datasets may use lowercase gt, try alternatives
            alt = img_path.with_name(img_path.stem + "_gt.png")
            if alt.exists():
                samples.append(Sample(img_path=str(img_path), mask_path=str(alt)))
            else:
                # Skip silently; you can log in inspect script
                continue
    return samples


def build_kolektor_splits(
    *,
    data_root: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
) -> Dict[str, List[Sample]]:
    """Create train/val/test splits for KolektorSDD2.

    Expects:
      <data_root>/train/*.png + *_GT.png
      <data_root>/test/*.png + *_GT.png

    Validation split is a subset of train/ (shuffled with seed).
    """
    root = Path(data_root)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    train_samples = _collect_pairs(train_dir)
    test_samples = _collect_pairs(test_dir)

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
        serializable = {k: [{"img_path": s.img_path, "mask_path": s.mask_path} for s in v] for k, v in splits.items()}
        save_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    return splits


def load_splits(path: str | Path) -> Dict[str, List[Sample]]:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Sample]] = {}
    for split, arr in obj.items():
        out[split] = [Sample(img_path=d["img_path"], mask_path=d["mask_path"]) for d in arr]
    return out


def _read_image(path: str | Path, in_channels: int) -> np.ndarray:
    """Read image as uint8 HxW (gray) or HxWx3 (RGB)."""
    if in_channels == 1:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img
    elif in_channels == 3:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError(f"in_channels must be 1 or 3, got {in_channels}")


def _read_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return m


def _resize(img: np.ndarray, mask: np.ndarray, H: int, W:int) -> Tuple[np.ndarray, np.ndarray]:
    if img.ndim == 2:
        img_r = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
    else:
        img_r = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (H, W), interpolation=cv2.INTER_NEAREST)
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
    # flips/rotations for both
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

    # photometric jitter only for image
    if brightness_jitter and brightness_jitter > 0:
        b = float(brightness_jitter)
        factor = 1.0 + random.uniform(-b, b)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if contrast_jitter and contrast_jitter > 0:
        c = float(contrast_jitter)
        factor = 1.0 + random.uniform(-c, c)
        mean = float(img.mean())
        img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    return img, mask


def _normalize(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5  # [-1, 1]
    return x


class KolektorSDD2SegDataset(Dataset):
    """KolektorSDD2 segmentation dataset (binary mask).

    Returns:
      image: float32 tensor (C, H, W) normalized to [-1,1]
      mask:  float32 tensor (1, H, W) in {0,1}
      meta: dict with img_path, mask_path
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        in_channels: int = 1,
        augment: bool = False,
        aug_cfg: Optional[Dict] = None,
    ):
        self.samples = list(samples)
        self.in_channels = int(in_channels)
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
        img = _read_image(s.img_path, self.in_channels)
        mask = _read_mask(s.mask_path)

        img, mask = _resize(img, mask, H=256, W=640) # avg size in dataset

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

        img = _normalize(img)

        # mask to {0,1}
        mask_f = (mask.astype(np.float32) / 255.0)
        mask_f = (mask_f > 0.5).astype(np.float32)

        if self.in_channels == 1:
            # (H,W)->(1,H,W)
            img_t = torch.from_numpy(img)[None, :, :]
        else:
            # (H,W,C)->(C,H,W)
            img_t = torch.from_numpy(img).permute(2, 0, 1)

        mask_t = torch.from_numpy(mask_f)[None, :, :]

        meta = {"img_path": s.img_path, "mask_path": s.mask_path}
        
        label = 1.0 if mask_t.sum().item() > 0 else 0.0
        label_t = torch.tensor([label], dtype=torch.float32)  # shape (1,)
        
        return img_t, mask_t, label_t, meta
