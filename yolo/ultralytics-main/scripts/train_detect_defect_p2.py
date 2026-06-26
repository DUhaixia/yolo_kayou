"""Train YOLO11-P2 detect with SES/DFIR neck; input defect_preprocess defaults to none (no offline/online preprocess)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.dfir_net import FIRCLite, _stabilize_bn_buffers
from ultralytics.nn.modules.ses_net import SmallDefectEnhance
from ultralytics.utils.freq_viz import run_freq_training_viz
from ultralytics.utils.torch_utils import TORCH_2_4, unwrap_model
from train_seg_defect import _save_preprocess_snapshot


def _apply_small_defect_hparams(trainer, edge_w: float) -> None:
    for m in trainer.model.model.modules():
        if isinstance(m, SmallDefectEnhance):
            m.edge_w.fill_(edge_w)


def _stabilize_firc_bn(trainer) -> None:
    """Repair FIRCLite spatial-branch BN running stats (fresh train only)."""
    for m in unwrap_model(trainer.model).modules():
        if isinstance(m, FIRCLite):
            _stabilize_bn_buffers(m)


def _on_train_start(trainer, edge_w: float, is_resume: bool) -> None:
    _apply_small_defect_hparams(trainer, edge_w)
    if not is_resume:
        _stabilize_firc_bn(trainer)


def _rebuild_optimizer_after_resume(trainer) -> None:
    """YOLO last.pt stores EMA weights + optimizer state for *training* weights — mismatch causes P/R collapse."""
    if not trainer.resume:
        return
    batch_size = trainer.batch_size // max(trainer.world_size, 1)
    weight_decay = trainer.args.weight_decay * trainer.batch_size * trainer.accumulate / trainer.args.nbs
    iterations = math.ceil(len(trainer.train_loader.dataset) / max(trainer.batch_size, trainer.args.nbs)) * trainer.epochs
    trainer.optimizer = trainer.build_optimizer(
        model=trainer.model,
        name=trainer.args.optimizer,
        lr=trainer.args.lr0,
        momentum=trainer.args.momentum,
        decay=weight_decay,
        iterations=iterations,
    )
    import torch

    trainer.scaler = (
        torch.amp.GradScaler("cuda", enabled=trainer.amp)
        if TORCH_2_4
        else torch.cuda.amp.GradScaler(enabled=trainer.amp)
    )
    trainer.scheduler.last_epoch = trainer.start_epoch - 1
    if trainer.ema:
        trainer.ema.updates = trainer.start_epoch * len(trainer.train_loader)
    print(
        f"Resume: fresh optimizer @ epoch {trainer.start_epoch + 1} "
        f"(EMA weights kept; fixes mAP drop after --resume)"
    )


def _ses_gain_warmup(trainer, target_gain: float, start_epoch: int) -> None:
    """Enable CSSG aux loss after backbone is stable (reduces early noise)."""
    if target_gain <= 0 or trainer.epoch < start_epoch:
        return
    trainer.args.ses_gain = target_gain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train YOLO11-P2 detect for sparse point/line defects")
    p.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "ultralytics/cfg/models/11/yolo11-p2.yaml"),
        help=(
            "Ablation models: yolo11.yaml (Exp1 baseline P3-P5), "
            "yolo11-p2.yaml (Exp2 P2), yolo11-p2-eca.yaml (Exp3), "
            "yolo11-p2-freqgate.yaml (Exp4), yolo11-p2-ses-dfir-stable.yaml (legacy FFT)"
        ),
    )
    p.add_argument(
        "--exp",
        type=str,
        default="",
        choices=["", "exp1", "exp2", "exp3", "exp4"],
        help="Shortcut: exp1=baseline, exp2=p2, exp3=p2+eca, exp4=p2+freqgate (overrides --model)",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default="yolo11n.pt",
        help="yolo11n.pt (partial backbone) or your yolo11n-p2 best.pt; False/none to train from scratch",
    )
    p.add_argument("--data", type=str, default=r"M:\压印 - 副本\dataSet-原始222\data.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8, help="Lower if OOM; 8 is safer for ses-dfir @1024")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="G:/卡游/runs/detect_p2")
    p.add_argument("--name", type=str, default="G:/卡游/defect_p2_ses_dfir")
    p.add_argument(
        "--defect-preprocess",
        type=str,
        default="none",
        choices=["none", "point", "line", "mixed"],
        help="none=raw image (recommended with ses-dfir); mixed/point/line=legacy OpenCV preprocess",
    )
    p.add_argument("--ses-gain", type=float, default=0.0, help="CSSG aux loss; use 0 until stable, then 0.02~0.05")
    p.add_argument(
        "--ses-gain-start",
        type=int,
        default=10,
        help="Epoch to apply --ses-gain (0=from start); only if ses-gain>0",
    )
    p.add_argument("--edge-w", type=float, default=0.05, help="Sobel edge blend in SmallDefectEnhance (lower=less FP)")
    p.add_argument("--lr0", type=float, default=0.005, help="Lower lr helps new FFT/CSSG neck layers")
    p.add_argument(
        "--cache",
        type=str,
        default="False",
        choices=["False", "ram", "disk"],
        help="False=no cache (~3MB/bmp read each epoch); disk needs ~26GB+ on dataset drive; ram needs enough memory",
    )
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--hsv-h", type=float, default=0.0)
    p.add_argument("--hsv-s", type=float, default=0.0)
    p.add_argument("--hsv-v", type=float, default=0.0)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=0.15)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--copy-paste", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from project/name/weights/<last|best>.pt (same --name/--project)",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default="last",
        choices=["last", "best"],
        help="After crash use 'best' if last.pt resume shows wrong P/R/mAP50",
    )
    p.add_argument(
        "--resume-keep-optimizer",
        action="store_true",
        help="Keep optimizer in checkpoint (default: rebuild — required for correct metrics)",
    )
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--cos-lr", action="store_true", default=True)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--save-preprocess-every", type=int, default=10)
    p.add_argument("--save-preprocess-count", type=int, default=4)
    p.add_argument(
        "--freq-viz-every",
        type=int,
        default=5,
        help="Every N epochs save FIRC/FGU/SES frequency panels under save_dir/freq_viz/ (0=off)",
    )
    p.add_argument("--freq-viz-samples", type=int, default=2, help="Train images per freq viz snapshot")
    return p.parse_args()


_EXP_MODELS = {
    "exp1": ROOT / "ultralytics/cfg/models/11/yolo11.yaml",
    "exp2": ROOT / "ultralytics/cfg/models/11/yolo11-p2.yaml",
    "exp3": ROOT / "ultralytics/cfg/models/11/yolo11-p2-eca.yaml",
    "exp4": ROOT / "ultralytics/cfg/models/11/yolo11-p2-freqgate.yaml",
}


def _resume_ckpt(project: str, name: str, which: str = "last") -> Path | None:
    ckpt = Path(project).expanduser() / name / "weights" / f"{which}.pt"
    return ckpt if ckpt.is_file() else None


def main() -> None:
    args = parse_args()
    model_path = _EXP_MODELS[args.exp] if args.exp else Path(args.model)
    is_resume = bool(args.resume)
    if is_resume:
        ckpt = _resume_ckpt(args.project, args.name, args.resume_from)
        if ckpt is None:
            raise FileNotFoundError(
                f"Resume checkpoint not found: {Path(args.project) / args.name / 'weights' / args.resume_from}.pt\n"
                "Check --project and --name match the interrupted run."
            )
        print(f"Resuming from {ckpt}")
        model = YOLO(str(ckpt.resolve()))
    else:
        model = YOLO(str(model_path.expanduser().resolve()))
    model.add_callback(
        "on_train_start",
        lambda trainer: _on_train_start(trainer, args.edge_w, is_resume),
    )
    if is_resume and not args.resume_keep_optimizer:
        model.add_callback("on_pretrain_routine_end", _rebuild_optimizer_after_resume)
    model.add_callback(
        "on_train_epoch_end",
        lambda trainer: _save_preprocess_snapshot(
            trainer=trainer,
            defect_mode=args.defect_preprocess,
            every_n=args.save_preprocess_every,
            sample_count=args.save_preprocess_count,
        ),
    )
    if args.freq_viz_every > 0:
        model.add_callback(
            "on_train_epoch_end",
            lambda trainer: run_freq_training_viz(
                trainer,
                every_n=args.freq_viz_every,
                sample_count=args.freq_viz_samples,
            ),
        )
    if args.ses_gain > 0 and args.ses_gain_start > 0:
        model.add_callback(
            "on_train_epoch_start",
            lambda trainer: _ses_gain_warmup(trainer, args.ses_gain, args.ses_gain_start),
        )
    cache_value = False if args.cache == "False" else args.cache
    pretrained = False if str(args.pretrained).lower() in {"false", "0", "none"} else args.pretrained
    if args.resume:
        pretrained = False
    model.train(
        pretrained=pretrained,
        task="detect",
        data=str(Path(args.data).expanduser().resolve()),
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
        optimizer="AdamW",
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
