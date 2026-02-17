# KolektorSDD2 Defect Segmentation 
Baseline **U-Net + MobileNetV2 encoder** for defect segmentation / localization.

https://www.vicos.si/resources/kolektorsdd2/
## Dataset layout 
Put the dataset here:

```
data/KolektorSDD2/
  train/
    xx.png
    xx_GT.png
    ...
  test/
    yy.png
    yy_GT.png
    ...
```

The code matches pairs by filename:
- image: `xx.png`
- mask : `xx_GT.png` (also supports `xx_gt.png`)

## Install
```bash
pip install -r requirements.txt
```

## 1) Inspect dataset 
This checks how many pairs were found and optionally saves some overlay previews.

```bash
python scripts/inspect_dataset.py --data_root data/KolektorSDD2 --save_viz
```

Overlays saved to: `outputs/inspect_kolektor/`

## 2) Train
```bash
python train.py --config configs/kolektorsdd2_unet_mnv2.yaml
```

Or override the dataset path:
```bash
python train.py --config configs/kolektorsdd2_unet_mnv2.yaml --data_root data/KolektorSDD2 --run_name unet_mnv2_kolektor
```

Outputs:
- `runs/<run_name>/checkpoints/best.pt`
- `runs/<run_name>/checkpoints/last.pt`
- `runs/<run_name>/splits.json` (train/val/test file lists for reproducibility)
- TensorBoard logs: `runs/<run_name>/tb/`

TensorBoard:
```bash
tensorboard --logdir runs/<run_name>/tb
```

## 3) Evaluate (val/test)
```bash
python eval.py --run_dir runs/unet_mnv2_kolektorsdd2 --split val  --ckpt best
python eval.py --run_dir runs/unet_mnv2_kolektorsdd2 --split test --ckpt best --save_preds
```

If `--save_preds` is enabled, predicted masks are saved under:
`runs/<run_name>/preds/<split>_<ckpt>/`.

## 4) Inference on any folder of images
```bash
python infer.py --run_dir runs/unet_mnv2_kolektorsdd2 --ckpt best --input data/KolektorSDD2/test --out_dir outputs --save_overlay
```

## 5) Benchmark inference latency/FPS + memory
```bash
python scripts/benchmark_infer.py --run_dir runs/unet_mnv2_kolektorsdd2 --ckpt best --device cuda --batch_size 1
python scripts/benchmark_infer.py --run_dir runs/unet_mnv2_kolektorsdd2 --ckpt best --device cpu  --batch_size 1
```

## Notes
- Default config uses `in_channels: 1` (grayscale). If your images are RGB, change `data.in_channels` to `3`.
- Metrics: Dice + IoU are computed with a default threshold (0.5).
