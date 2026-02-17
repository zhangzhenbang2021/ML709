from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    best_metric: float,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    torch.save(ckpt, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[int, float, Optional[Dict[str, Any]]]:
    """Load checkpoint. Returns (epoch, best_metric, config)."""
    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if optimizer is not None and ckpt.get("optim_state") is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    epoch = int(ckpt.get("epoch", 0))
    best_metric = float(ckpt.get("best_metric", 0.0))
    config = ckpt.get("config", None)
    return epoch, best_metric, config
