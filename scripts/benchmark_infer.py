from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import psutil
import torch

from src.models.unet_mobilenetv2 import UNetMobileNetV2
from src.src_utils import get_logger, load_checkpoint, load_yaml


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference speed for a trained model")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best", help="best|last|/path/to/ckpt.pt")
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    logger = get_logger()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")

    img_size = int(cfg["data"]["img_size"])
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

    x = torch.randn(args.batch_size, in_channels, img_size, img_size, device=device)

    process = psutil.Process()
    rss_before = process.memory_info().rss

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        _sync(device)

    times = []
    with torch.no_grad():
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _ = model(x)
            _sync(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    rss_after = process.memory_info().rss
    rss_mb = (rss_after - rss_before) / (1024 ** 2)

    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times)
    mean = statistics.mean(times)
    fps_mean = args.batch_size / (mean / 1000.0)

    logger.info(f"Latency (ms) mean={mean:.2f} p50={p50:.2f} p90={p90:.2f} | Throughput (FPS) ~ {fps_mean:.2f}")

    out = {
        "device": str(device),
        "batch_size": args.batch_size,
        "img_size": img_size,
        "in_channels": in_channels,
        "iters": args.iters,
        "warmup": args.warmup,
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "fps_mean": fps_mean,
        "rss_delta_mb": rss_mb,
    }
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        out["cuda_peak_allocated_mb"] = peak_mb
        logger.info(f"Peak CUDA memory allocated: {peak_mb:.2f} MB")

    out_path = run_dir / f"benchmark_{args.ckpt}_bs{args.batch_size}_{device.type}.json"
    out_path.write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark json: {out_path}")


if __name__ == "__main__":
    main()
