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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train YOLO segmentation with defect preprocess")
    parser.add_argument("--model", type=str, default=r"H:\Python_cls\YOLO1111111\yolo\ultralytics-main\yolo11n-seg.pt", help="Model path or preset")
    parser.add_argument("--data", type=str, default="L:/1-原始数据合并 -0616/train_val/data.yaml", help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/fanshe")
    parser.add_argument("--name", type=str, default="seg")
    parser.add_argument("--defect-preprocess", type=str, default="none", choices=["none", "point", "line", "mixed"])
    parser.add_argument("--cache", type=str, default="disk", choices=["False", "ram", "disk"])
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--hsv-h", type=float, default=0.00)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="从 last.pt 续训（需与 --project/--name 一致）")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"",
        help="续训权重，默认 project/name/weights/last.pt；也可填 best.pt",
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--save-period", type=int, default=-1, help="每 N epoch 额外保存 epochN.pt，-1 不保存")
    parser.add_argument("--save-preprocess-every", type=int, default=5, help="Save preprocess analysis every N epochs")
    parser.add_argument("--save-preprocess-count", type=int, default=6, help="How many train images to save each snapshot")
    return parser.parse_args()


def _safe_stem(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


def _save_preprocess_snapshot(trainer, defect_mode: str, every_n: int, sample_count: int) -> None:
    # trainer.epoch is zero-based; use human-readable epoch number in outputs.
    epoch = int(trainer.epoch) + 1
    if every_n <= 0 or epoch % every_n != 0:
        return

    dataset = getattr(trainer.train_loader, "dataset", None)
    im_files = list(getattr(dataset, "im_files", [])) if dataset is not None else []
    if not im_files:
        LOGGER.warning("No train images found for preprocess snapshot.")
        return

    sample_count = max(1, min(sample_count, len(im_files)))
    step = max(1, len(im_files) // sample_count)
    chosen = im_files[::step][:sample_count]

    out_dir = Path(trainer.save_dir) / "preprocess_snapshots" / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in chosen:
        im = imread(src, flags=cv2.IMREAD_COLOR)
        if im is None:
            continue
        proc = apply_defect_preprocess(im, defect_mode)
        stem = _safe_stem(src)
        cv2.imencode(".png", im)[1].tofile(str(out_dir / f"{stem}_orig.png"))
        cv2.imencode(".png", proc)[1].tofile(str(out_dir / f"{stem}_proc_{defect_mode}.png"))

        if proc.ndim == 3 and proc.shape[2] == 3:
            b, g, r = cv2.split(proc)
            cv2.imencode(".png", b)[1].tofile(str(out_dir / f"{stem}_ch_b.png"))
            cv2.imencode(".png", g)[1].tofile(str(out_dir / f"{stem}_ch_g.png"))
            cv2.imencode(".png", r)[1].tofile(str(out_dir / f"{stem}_ch_r.png"))

    LOGGER.info(f"Saved preprocess snapshot: {out_dir}")


def _resolve_weights(args: argparse.Namespace) -> str:
    if args.weights:
        p = Path(args.weights).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"权重不存在: {p}")
        return str(p.resolve())
    if args.resume:
        proj_tail = Path(args.project).name  # runs/seg -> seg
        candidates = [
            Path("runs") / "segment" / "runs" / proj_tail / args.name / "weights" / "last.pt",
            Path(args.project) / args.name / "weights" / "last.pt",
        ]
        for p in candidates:
            if p.is_file():
                return str(p.resolve())
        raise FileNotFoundError(
            "未找到 last.pt，请在本仓库 yolo 目录下运行，或使用 --weights 指定:\n"
            f"  runs/segment/runs/{proj_tail}/{args.name}/weights/last.pt"
        )
    return args.model


def main() -> None:
    args = parse_args()
    data_path = str(Path(args.data).expanduser().resolve())
    model_path = _resolve_weights(args)
    if args.resume:
        LOGGER.info(f"续训权重: {model_path}")
    model = YOLO(model_path)

    model.add_callback(
        "on_train_epoch_end",
        lambda trainer: _save_preprocess_snapshot(
            trainer=trainer,
            defect_mode=args.defect_preprocess,
            every_n=args.save_preprocess_every,
            sample_count=args.save_preprocess_count,
        ),
    )

    cache_value: bool | str
    if args.cache == "False":
        cache_value = False
    else:
        cache_value = args.cache

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
        save_period=args.save_period,
        resume=args.resume,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
