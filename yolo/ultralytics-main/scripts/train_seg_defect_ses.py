"""Train YOLO11-seg with defect preprocess + SES-Net (ACFEE/CSSG) modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.defect_preprocess import apply_defect_preprocess
from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imread

from train_seg_defect import _save_preprocess_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train YOLO11-seg with defect preprocess + SES-Net modules")
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "ultralytics/cfg/models/11/yolo11-seg-ses.yaml"),
        help="Model yaml (yolo11n-seg-ses.yaml) or weights",
    )
    parser.add_argument("--data", type=str, default="G:/压印 - 副本/dataSet-mixed/data.yaml", help="Dataset yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/seg_ses")
    parser.add_argument("--name", type=str, default="defect_seg_ses")
    parser.add_argument("--defect-preprocess", type=str, default="mixed", choices=["none", "point", "line", "mixed"])
    parser.add_argument("--ses-gain", type=float, default=0.1, help="CSSG auxiliary loss weight")
    parser.add_argument("--ses-patch", type=int, default=32, help="CSSG patch size (must divide imgsz)")
    parser.add_argument("--cache", type=str, default="disk", choices=["False", "ram", "disk"])
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--save-preprocess-every", type=int, default=5)
    parser.add_argument("--save-preprocess-count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = str(Path(args.data).expanduser().resolve())
    model = YOLO(args.model)

    model.add_callback(
        "on_train_epoch_end",
        lambda trainer: _save_preprocess_snapshot(
            trainer=trainer,
            defect_mode=args.defect_preprocess,
            every_n=args.save_preprocess_every,
            sample_count=args.save_preprocess_count,
        ),
    )

    cache_value = False if args.cache == "False" else args.cache

    model.train(
        task="segment",
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        defect_preprocess=args.defect_preprocess,
        ses_gain=args.ses_gain,
        cache=cache_value,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        seed=args.seed,
        amp=args.amp,
        cos_lr=args.cos_lr,
        patience=args.patience,
        resume=args.resume,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
