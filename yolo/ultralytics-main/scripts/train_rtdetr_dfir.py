"""Train DFIR-DETR (2512.07078v4) via Ultralytics RT-DETR API.

Paper defaults (Sec. 4.1):
  optimizer=AdamW, lr0=1e-4, weight_decay=5e-4, epochs=300, batch=4, imgsz=640

Example:
  python scripts/train_rtdetr_dfir.py --data path/to/visdrone.yaml --epochs 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Train DFIR-DETR (RT-DETR + DCFA/DFPN/FIRC3)")
    p.add_argument(
        "--model",
        type=str,
        default=str(root / "ultralytics/cfg/models/rt-detr/rtdetr-dfir-r18.yaml"),
        help="DFIR-DETR yaml",
    )
    p.add_argument("--data", type=str, required=True, help="Dataset yaml (VisDrone / NEU-DET)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="dfir_detr_r18")
    p.add_argument("--lr0", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--pretrained", action="store_true", help="Load RT-DETR-R18 weights where shapes match")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = RTDETR(args.model)
    if args.pretrained:
        model.load("rtdetr-r18.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        amp=True,
    )


if __name__ == "__main__":
    main()
