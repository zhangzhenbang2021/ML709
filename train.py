from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import WeightedRandomSampler
import cv2
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.kolektor_sdd2 import KolektorSDD2SegDataset, build_kolektor_splits, load_splits
from src.losses.dice import DiceLoss
from src.metrics.seg_metrics import SegmentationMeter
from src.models.unet_mobilenetv2 import UNetMobileNetV2
from src.src_utils import get_logger, load_yaml, save_checkpoint, save_yaml, seed_everything


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def build_run_dir(out_dir: str, run_name: str) -> Path:
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "preds").mkdir(exist_ok=True)
    return run_dir


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seg_loss_fn: nn.Module,         
    meter: SegmentationMeter,
    amp: bool,
    cls_loss_fn: nn.Module | None = None,  
    cls_w: float = 0.0,                   
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Returns:
      avg_total_loss, avg_seg_loss, avg_cls_loss, metrics
    """
    model.eval()
    meter.reset()

    total_total = 0.0
    total_seg = 0.0
    total_cls = 0.0
    n = 0  # count samples

    for imgs, masks, labels, _meta in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=amp):
            seg_logits, cls_logits = model(imgs)

            seg_loss = seg_loss_fn(seg_logits, masks)

            if cls_loss_fn is not None:
                cls_loss = cls_loss_fn(cls_logits, labels)
            else:
                cls_loss = torch.zeros((), device=device)

            loss = seg_loss + cls_w * cls_loss

        bs = imgs.size(0)
        total_total += float(loss.detach().item()) * bs
        total_seg += float(seg_loss.detach().item()) * bs
        total_cls += float(cls_loss.detach().item()) * bs
        n += bs

        meter.update(seg_logits.detach(), masks)

    avg_total = total_total / max(n, 1)
    avg_seg = total_seg / max(n, 1)
    avg_cls = total_cls / max(n, 1)

    return avg_total, avg_seg, avg_cls, meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net + MobileNetV2 on KolektorSDD2 segmentation")
    parser.add_argument("--config", type=str, default="configs/kolektorsdd2_unet_mnv2.yaml")
    parser.add_argument("--data_root", type=str, default=None, help="Override data.data_root")
    parser.add_argument("--run_name", type=str, default=None, help="Override experiment.run_name")
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.run_name is not None:
        cfg["experiment"]["run_name"] = args.run_name

    seed = int(cfg["experiment"].get("seed", 42))
    seed_everything(seed)

    logger = get_logger()
    run_dir = build_run_dir(cfg["experiment"]["out_dir"], cfg["experiment"]["run_name"])
    save_yaml(cfg, run_dir / "config.yaml")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    splits_path = run_dir / "splits.json"
    if splits_path.exists():
        splits = load_splits(splits_path)
        logger.info(f"Loaded existing splits: {splits_path}")
    else:
        splits = build_kolektor_splits(
            data_root=cfg["data"]["data_root"],
            val_ratio=float(cfg["data"]["val_ratio"]),
            seed=seed,
            save_path=splits_path,
        )
        logger.info(f"Saved splits to: {splits_path}")

    aug_cfg = cfg.get("aug", {})
    in_channels = int(cfg["data"].get("in_channels", 1))

    train_ds = KolektorSDD2SegDataset(
        splits["train"],
        in_channels=in_channels,
        augment=bool(aug_cfg.get("enable", True)),
        aug_cfg=aug_cfg,
    )
    val_ds = KolektorSDD2SegDataset(
        splits["val"],
        in_channels=in_channels,
        augment=False,
        aug_cfg=aug_cfg,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    
    # ------------------------------
    # Sampling + losses (seg + cls)
    # ------------------------------
    train_samples = splits["train"]

    # ---- 1) image-level stats: how many defect images? ----
    flags: list[int] = []
    n_pos = 0  # defect images
    n_neg = 0  # normal images

    for s in train_samples:
        m = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
        is_def = int(m is not None and m.max() > 0)
        flags.append(is_def)
        n_pos += is_def
        n_neg += (1 - is_def)

    defect_ratio = n_pos / max(len(flags), 1)
    logger.info(f"[DATA] train n_pos={n_pos}, n_neg={n_neg}, defect_ratio={defect_ratio:.4f}")

    # ---- 2) sampler: upsample defect images to ~1:1 ----
    w_def = n_neg / max(n_pos, 1)  # expected ~ balance
    w_def_max = float(cfg["data"].get("sampler_max_weight", 10.0))
    w_def = min(w_def, w_def_max)

    weights = [w_def if f == 1 else 1.0 for f in flags]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    logger.info(f"[SAMPLER] w_def={w_def:.4f} (cap={w_def_max})")

    # ---- 3) classification loss: DO NOT use pos_weight if sampler is used ----
    # sampler already makes batches roughly balanced -> pos_weight should be ~1
    cls_loss_fn = nn.BCEWithLogitsLoss()
    cls_w = float(cfg["loss"].get("cls_loss_weight", 0.1))
    logger.info(f"[CLS] cls_loss_weight={cls_w}")

    # ---- 4) segmentation loss: pixel-level pos_weight from fg_ratio (auto) ----
    fg = 0
    tot = 0
    for s in train_samples:
        m = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        fg += int((m > 0).sum())
        tot += int(m.size)

    fg_ratio = fg / max(tot, 1)
    seg_pos_weight_value = (1.0 - fg_ratio) / max(fg_ratio, 1e-12)

    seg_pw_max = float(cfg["loss"].get("seg_pos_weight_max", 300.0))
    seg_pos_weight_value = min(seg_pos_weight_value, seg_pw_max)

    logger.info(
        f"[SEG] fg_ratio={fg_ratio:.6f}, seg_pos_weight={seg_pos_weight_value:.2f} (cap={seg_pw_max})"
    )

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([seg_pos_weight_value], device=device, dtype=torch.float32)
    )
    dice = DiceLoss(smooth=float(cfg["loss"].get("dice_smooth", 1.0)))
    bce_w = float(cfg["loss"].get("bce_weight", 0.5))
    dice_w = float(cfg["loss"].get("dice_weight", 0.5))

    def loss_fn(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return bce_w * bce(logits, masks) + dice_w * dice(logits, masks)

    # ---- 5) dataloaders ----
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        shuffle=False,  # must be False when sampler is provided
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=pin,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=pin,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    model = UNetMobileNetV2(
        in_channels=in_channels,
        num_classes=int(cfg["model"]["num_classes"]),
        encoder_pretrained=bool(cfg["model"].get("encoder_pretrained", False)),
    ).to(device)
    

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["train"]["epochs"]))

    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    print('amp:', amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    meter = SegmentationMeter(threshold=0.5)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    best_metric = -1.0
    best_epoch = -1
    patience = int(cfg["train"].get("early_stop_patience", 0))
    bad_epochs = 0

    log_every = int(cfg["train"].get("log_every", 25))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    save_best_metric = cfg["train"].get("save_best_metric", "dice")

    logger.info(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(splits['test'])}")
    
    
    # print cfg
    logger.info("Config:")
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")
        

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}", leave=False)


        running_total = 0.0
        running_seg = 0.0
        running_cls = 0.0
        
        for step, (imgs, masks, labels, _meta) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                seg_logits, cls_logits = model(imgs)
                seg_loss = loss_fn(seg_logits, masks)

                labels = labels.to(device, non_blocking=True).float()
                cls_loss = cls_loss_fn(cls_logits, labels)

                loss = seg_loss + cls_w * cls_loss

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            running_total += float(loss.detach().item())
            running_seg += float(seg_loss.detach().item())
            running_cls += float(cls_loss.detach().item())
            
            if step % log_every == 0:
                avg_total = running_total / log_every
                avg_seg = running_seg / log_every
                avg_cls = running_cls / log_every

                running_total = 0.0
                running_seg = 0.0
                running_cls = 0.0

                lr = float(opt.param_groups[0]["lr"])
                pbar.set_postfix(
                    loss=f"{avg_total:.4f}",
                    seg=f"{avg_seg:.4f}",
                    cls=f"{avg_cls:.4f}",
                    lr=f"{lr:.2e}",
                )

                global_step = (epoch - 1) * len(train_loader) + step

                writer.add_scalar("train/loss_total", avg_total, global_step)
                writer.add_scalar("train/loss_seg", avg_seg, global_step)
                writer.add_scalar("train/loss_cls", avg_cls, global_step)
                writer.add_scalar("train/lr", lr, global_step)

        scheduler.step()

        if epoch % int(cfg["train"].get("eval_every", 1)) == 0:
            val_loss, val_seg_loss, val_cls_loss, val_metrics = evaluate(
                model, val_loader, device, loss_fn, meter, amp,
                cls_loss_fn=cls_loss_fn, cls_w=cls_w
            )

                        
            writer.add_scalar("val/loss_total", val_loss, epoch)
            writer.add_scalar("val/loss_seg", val_seg_loss, epoch)
            writer.add_scalar("val/loss_cls", val_cls_loss, epoch)

            for k in [
                "dice_all", "iou_all",
                "dice_defect_only", "iou_defect_only",
                "dice_normal_only", "iou_normal_only",
                "normal_fp_rate",
                "n_defect_imgs", "n_normal_imgs",
            ]:
                v = val_metrics.get(k, None)
                if v is not None:
                    writer.add_scalar(f"val/{k}", float(v), epoch)

            # ---- choose metric for "best" checkpoint ----
            save_best_metric = cfg["train"].get("save_best_metric", "dice_defect_only")

            def _metric_for_compare(metrics: dict, name: str) -> float:
                """Return a scalar where larger is better."""
                v = metrics.get(name, None)
                if v is None:
                    return float("-inf")

                # normal_fp_rate: smaller is better -> maximize negative
                if name == "normal_fp_rate":
                    return -float(v)

                return float(v)

            metric_value = _metric_for_compare(val_metrics, save_best_metric)

            # ---- logging ----
            dice_all = val_metrics.get("dice_all", None)
            dice_def = val_metrics.get("dice_defect_only", None)
            dice_norm = val_metrics.get("dice_normal_only", None)
            fp_norm = val_metrics.get("normal_fp_rate", None)
            n_def = val_metrics.get("n_defect_imgs", None)
            n_norm = val_metrics.get("n_normal_imgs", None)

            logger.info(
                f"[Epoch {epoch}] val_total={val_loss:.4f} val_seg={val_seg_loss:.4f} val_cls={val_cls_loss:.4f} "
                f"dice_all={dice_all:.4f} "
                f"dice_def={dice_def if dice_def is not None else 'NA'} "
                f"dice_norm={dice_norm if dice_norm is not None else 'NA'} "
                f"fp_norm={fp_norm if fp_norm is not None else 'NA'} "
                f"n_def={n_def if n_def is not None else 'NA'} "
                f"n_norm={n_norm if n_norm is not None else 'NA'}"
            )

            # ---- save last ----
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                epoch=epoch,
                best_metric=best_metric,
                model=model,
                optimizer=opt,
                scaler=scaler,
                config=cfg,
            )

            # ---- save best ----
            if metric_value > best_metric:
                best_metric = metric_value
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    epoch=epoch,
                    best_metric=best_metric,
                    model=model,
                    optimizer=opt,
                    scaler=scaler,
                    config=cfg,
                )

                raw_best = val_metrics.get(save_best_metric, None)
                logger.info(f"New best {save_best_metric}: {raw_best if raw_best is not None else best_metric:.4f} at epoch {epoch}")
            else:
                bad_epochs += 1

            if patience and patience > 0 and bad_epochs >= patience:
                logger.info(f"Early stopping: no improvement in {patience} evals. Best epoch={best_epoch}.")
                break

    writer.close()
    logger.info(f"Done. Best {save_best_metric}={best_metric:.4f} at epoch {best_epoch}.")
    logger.info(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
