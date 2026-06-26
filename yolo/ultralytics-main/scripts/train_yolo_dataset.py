"""通用 YOLO 数据集训练脚本（自动识别检测 / 分割）。

在本仓库根目录运行（所有模型配置均在当前路径下）::

    cd H:\\Python_cls\\YOLO1111111\\yolo\\ultralytics-main
    python scripts/train_yolo_dataset.py --data D:/data.yaml

数据集目录::

    dataset/data.yaml
    dataset/images/train  dataset/labels/train
    dataset/images/val    dataset/labels/val

推荐用 --preset 选择结构（路径相对本仓库 ROOT，无需手写 ultralytics/cfg/...）::

    --preset detect          # yolo11n.pt 检测
    --preset segment         # yolo11n-seg.pt 分割
    --preset yolo11          # cfg/models/11/yolo11.yaml
    --preset yolo11-seg      # cfg/models/11/yolo11-seg.yaml
    --preset rtdetr-l        # cfg/models/rt-detr/rtdetr-l.yaml
    --preset rtdetr-resnet50 # cfg/models/rt-detr/rtdetr-resnet50.yaml
    --preset yolo11-p2       # cfg/models/11/yolo11-p2.yaml（扩展仓库）
    --preset ses-dfir        # stable 检测（P2 SES+DFIR，P3 无 FIRC）
    --preset ses-dfir-p3     # 检测 + P3 FIRC3Lite（推荐在 stable 基础上升级）
    --preset ses-dfir-opt    # P3 FIRC + patch=8 更小点
    --preset ses-dfir-seg      # 实例分割 stable（P3 无 FIRC）
    --preset ses-dfir-seg-p3   # 实例分割 + P3 FIRC3Lite（点/线缺陷推荐）
    --preset dfir-detr       # cfg/models/rt-detr/rtdetr-dfir-r18.yaml

示例::

    python scripts/train_yolo_dataset.py --data D:/data.yaml
    python scripts/train_yolo_dataset.py --data D:/data.yaml --preset segment --imgsz 1024
    python scripts/train_yolo_dataset.py --data D:/data.yaml --preset yolo11-p2 --pretrained yolo11n.pt
    python scripts/train_yolo_dataset.py --data D:/data.yaml --preset ses-dfir-p3 --size m --batch 4
    python scripts/train_yolo_dataset.py --data D:/data.yaml --preset ses-dfir-seg-p3 --size s --fast --cache disk
    python scripts/train_yolo_dataset.py --data D:/data.yaml --preset dfir-detr --framework rtdetr --lr0 1e-4
    python scripts/train_yolo_dataset.py --list-presets
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR, YOLO
from ultralytics.utils import LOGGER

# 相对本仓库 ROOT 的模型配置
_MODEL_REL = {
    "yolo11": "ultralytics/cfg/models/11/yolo11.yaml",
    "yolo11-seg": "ultralytics/cfg/models/11/yolo11-seg.yaml",
    "yolo11-p2": "ultralytics/cfg/models/11/yolo11-p2.yaml",
    "ses-dfir": "ultralytics/cfg/models/11/yolo11-p2-ses-dfir-stable.yaml",
    "ses-dfir-p3": "ultralytics/cfg/models/11/yolo11-p2-ses-dfir-p3.yaml",
    "ses-dfir-opt": "ultralytics/cfg/models/11/yolo11-p2-ses-dfir-opt.yaml",
    "ses-dfir-seg": "ultralytics/cfg/models/11/yolo11-p2-ses-dfir-stable-seg.yaml",
    "ses-dfir-seg-full": "ultralytics/cfg/models/11/yolo11-p2-ses-dfir-seg.yaml",
    "ses-dfir-seg-p3": "ultralytics/cfg/models/11/yolo11-p2-ses-dfir-p3-seg.yaml",
    "rtdetr-l": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
    "rtdetr-resnet50": "ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml",
    "rtdetr-resnet101": "ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml",
    "dfir-detr": "ultralytics/cfg/models/rt-detr/rtdetr-dfir-r18.yaml",
}

# preset -> (framework, 权重或 yaml 键, 默认 pretrained 提示)
PRESETS: dict[str, dict] = {
    "detect": {"framework": "yolo", "model": "yolo11n.pt", "task": "detect"},
    "segment": {"framework": "yolo", "model": "yolo11n-seg.pt", "task": "segment"},
    "yolo11": {"framework": "yolo", "model": "yolo11", "task": "detect"},
    "yolo11-seg": {"framework": "yolo", "model": "yolo11-seg", "task": "segment"},
    "yolo11-p2": {"framework": "yolo", "model": "yolo11-p2", "task": "detect", "pretrained": "yolo11n.pt"},
    "ses-dfir": {"framework": "yolo", "model": "ses-dfir", "task": "detect", "pretrained": "yolo11n.pt"},
    "ses-dfir-p3": {
        "framework": "yolo",
        "model": "ses-dfir-p3",
        "task": "detect",
        "pretrained": "",
        "finetune_from": "runs/detect/runs/train/detect_ses_dfir/weights/best.pt",
        "finetune_scale": "n",
        "lr0": 0.001,
    },
    "ses-dfir-opt": {
        "framework": "yolo",
        "model": "ses-dfir-opt",
        "task": "detect",
        "pretrained": "yolo11n.pt",
    },
    "ses-dfir-seg": {
        "framework": "yolo",
        "model": "ses-dfir-seg",
        "task": "segment",
        "pretrained": "yolo11n-seg.pt",
    },
    "ses-dfir-seg-full": {
        "framework": "yolo",
        "model": "ses-dfir-seg-full",
        "task": "segment",
        "pretrained": "yolo11n-seg.pt",
    },
    "ses-dfir-seg-p3": {
        "framework": "yolo",
        "model": "ses-dfir-seg-p3",
        "task": "segment",
        "pretrained": "yolo11n-seg.pt",
        "copy_paste": 0.15,
        "overlap_mask": False,
        "mask_ratio": 2,
        "defect_preprocess": "mixed",
    },
    "rtdetr-l": {"framework": "rtdetr", "model": "rtdetr-l", "task": "detect"},
    "rtdetr-resnet50": {"framework": "rtdetr", "model": "rtdetr-resnet50", "task": "detect"},
    "dfir-detr": {"framework": "rtdetr", "model": "dfir-detr", "task": "detect", "lr0": 1e-4},
}


def _insert_yaml_scale(yaml_path: Path, size: str) -> Path:
    """yolo11-p2.yaml + m -> yolo11m-p2.yaml（Ultralytics 按文件名解析 scale）。"""
    if size == "n":
        return yaml_path
    stem = yaml_path.stem
    if re.search(r"yolo\d+[nslmx]", stem):
        return yaml_path
    new_stem = re.sub(r"(yolo\d+)", rf"\1{size}", stem, count=1)
    return yaml_path.with_name(f"{new_stem}{yaml_path.suffix}")


def _scaled_pt_name(pt: str, size: str) -> str:
    if not pt or not re.search(r"yolo\d+[nslmx]", pt):
        return pt
    return re.sub(r"(yolo\d+)[nslmx]", rf"\g<1>{size}", pt)


def _default_pretrained(size: str, task: str) -> str:
    return f"yolo11{size}-seg.pt" if task == "segment" else f"yolo11{size}.pt"


def _apply_fast_overrides(args: argparse.Namespace, preset_info: dict) -> dict:
    """--fast: 优先吞吐，关闭确定性并减轻数据/增强开销。"""
    if not args.fast:
        return preset_info
    LOGGER.info(
        "启用 --fast：deterministic=False, cache=disk, workers↑, 关闭绘图，"
        "defect_preprocess=none, copy_paste=0, mask_ratio=4（可用参数单独覆盖）"
    )
    info = dict(preset_info)
    if args.cache == "False":
        args.cache = "disk"
    if args.workers <= 4:
        args.workers = 8
    if args.deterministic is None:
        args.deterministic = False
    if args.plots is None:
        args.plots = False
    if args.defect_preprocess == "none":
        info.pop("defect_preprocess", None)
    if args.copy_paste < 0:
        info["copy_paste"] = 0.0
    if args.mask_ratio <= 0:
        info["mask_ratio"] = 4
    return info


def _resolve_model_file(key: str) -> Path:
    rel = _MODEL_REL.get(key)
    if not rel:
        raise KeyError(key)
    cand = ROOT / rel
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        f"未找到模型配置 [{key}]: {cand}\n请在本仓库 ultralytics-main 下运行，或使用 --model 指定绝对路径。"
    )


def resolve_model_path(
    preset: str,
    explicit_model: str,
    task: str,
    framework: str,
    size: str,
) -> tuple[str, str, str]:
    """返回 (model_path, framework, task)。"""
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"未知 preset: {preset}，可用: {', '.join(PRESETS)}")
        info = PRESETS[preset]
        framework = info["framework"]
        task = info["task"]
        m = info["model"]
        if m in _MODEL_REL:
            p = _insert_yaml_scale(_resolve_model_file(m), size)
            return str(p), framework, task
        return m, framework, task

    if explicit_model:
        p = Path(explicit_model).expanduser()
        if p.is_file():
            p = p.resolve()
        else:
            cand = ROOT / explicit_model
            if cand.is_file():
                p = cand.resolve()
            else:
                raise FileNotFoundError(f"模型不存在: {explicit_model}")
        if p.suffix in (".yaml", ".yml"):
            p = _insert_yaml_scale(p, size)
        return str(p), framework, task

    if framework == "rtdetr":
        return str(_resolve_model_file("rtdetr-l")), framework, "detect"

    if task == "segment":
        return f"yolo11{size}-seg.pt", framework, task
    return f"yolo11{size}.pt", framework, task


def list_presets() -> None:
    print(f"仓库 ROOT: {ROOT}\n")
    print(f"{'preset':<18} {'framework':<8} {'task':<8} {'model'}")
    print("-" * 72)
    for name, info in PRESETS.items():
        m = info["model"]
        if m in _MODEL_REL:
            try:
                path = _resolve_model_file(m)
                m = f"OK  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
            except FileNotFoundError:
                m = f"缺失  {_MODEL_REL[m]}"
        print(f"{name:<18} {info['framework']:<8} {info['task']:<8} {m}")


def _load_data_yaml(data_yaml: Path) -> dict:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid data.yaml: {data_yaml}")
    return cfg


def _resolve_dataset_root(data_yaml: Path, cfg: dict) -> Path:
    root = cfg.get("path") or data_yaml.parent
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def _label_dirs(root: Path, cfg: dict) -> list[Path]:
    dirs: list[Path] = []
    for key in ("train", "val", "test"):
        split = cfg.get(key)
        if not split:
            continue
        parts = Path(split).parts
        if parts and parts[0] == "images":
            lbl = Path("labels", *parts[1:])
        else:
            lbl = Path(str(split).replace("images", "labels"))
        label_dir = (root / lbl).resolve()
        if label_dir.is_dir():
            dirs.append(label_dir)
    return dirs


def detect_task_from_labels(data_yaml: Path) -> str:
    cfg = _load_data_yaml(data_yaml)
    root = _resolve_dataset_root(data_yaml, cfg)
    label_dirs = _label_dirs(root, cfg)
    if not label_dirs:
        raise FileNotFoundError(f"未找到 labels 目录，数据集根: {root}")

    seg_votes = det_votes = checked = 0
    for lbl_dir in label_dirs:
        for txt in sorted(lbl_dir.glob("*.txt"))[:200]:
            checked += 1
            for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                n = len(ln.split())
                if n > 5:
                    seg_votes += 1
                elif n == 5:
                    det_votes += 1

    if checked == 0:
        LOGGER.warning("标签为空，默认 detect")
        return "detect"
    task = "segment" if seg_votes > det_votes else "detect"
    LOGGER.info(f"标签抽样 segment={seg_votes} detect={det_votes} -> {task}")
    return task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练本地 YOLO 格式数据集")
    p.add_argument("--data", type=str, default="", help="data.yaml 路径")
    p.add_argument("--list-presets", action="store_true", help="列出 preset 及配置文件是否存在")
    p.add_argument("--preset", type=str, default="", help=f"preset 名称: {', '.join(PRESETS)}")
    p.add_argument("--task", type=str, default="auto", choices=["auto", "detect", "segment"])
    p.add_argument("--framework", type=str, default="", choices=["", "yolo", "rtdetr"])
    p.add_argument("--model", type=str, default="", help="自定义 .pt/.yaml（相对 ROOT 或绝对路径）")
    p.add_argument("--size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--lr0", type=float, default=-1.0, help="<=0 时用 preset 默认或 0.01")
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--optimizer", type=str, default="auto")
    p.add_argument("--cache", type=str, default="False", choices=["False", "ram", "disk"])
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--hsv-h", type=float, default=0.0)
    p.add_argument("--hsv-s", type=float, default=0.0)
    p.add_argument("--hsv-v", type=float, default=0.0)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=0.2)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=-1.0, help="<=0 时用 preset 默认或 0")
    p.add_argument("--overlap-mask", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--retina-masks", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mask-ratio", type=int, default=-1, help="<=0 时用 preset 默认或 4")
    p.add_argument("--ses-gain", type=float, default=-1.0, help="SES 增益，<=0 时用 preset 默认或 0.1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--cos-lr", action="store_true", default=True)
    p.add_argument("--defect-preprocess", type=str, default="none", choices=["none", "point", "line", "mixed"])
    p.add_argument("--fast", action="store_true", help="加速训练（关确定性、缓存、减增强/绘图开销）")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        list_presets()
        return
    if not args.data:
        raise SystemExit("请指定 --data，或使用 --list-presets")

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {data_path}")

    task = detect_task_from_labels(data_path) if args.task == "auto" else args.task
    framework = args.framework or "yolo"
    model_path, framework, preset_task = resolve_model_path(
        args.preset, args.model.strip(), task, framework, args.size
    )
    if args.preset and args.task == "auto":
        task = preset_task

    if framework == "rtdetr" and task != "detect":
        raise ValueError("RT-DETR 仅支持 detect")

    preset_info = _apply_fast_overrides(args, PRESETS.get(args.preset, {}))
    pretrained = args.pretrained or preset_info.get("pretrained", "")
    finetune = preset_info.get("finetune_from", "")
    finetune_scale = preset_info.get("finetune_scale", "n")
    used_finetune = False
    if not args.pretrained and finetune and args.size == finetune_scale:
        fp = Path(finetune).expanduser()
        if not fp.is_absolute():
            fp = ROOT / fp
        if fp.is_file():
            pretrained = str(fp)
            used_finetune = True
            LOGGER.info(f"preset 微调权重 (scale={finetune_scale}): {pretrained}")
    if not pretrained and framework == "yolo" and str(model_path).endswith((".yaml", ".yml")):
        pretrained = _default_pretrained(args.size, task)
    elif pretrained and not used_finetune:
        pretrained = _scaled_pt_name(pretrained, args.size)
    lr0 = args.lr0 if args.lr0 > 0 else preset_info.get("lr0", 0.001 if used_finetune else 0.01)

    cache: bool | str = False if args.cache == "False" else args.cache
    LOGGER.info(f"ROOT={ROOT}")
    LOGGER.info(
        f"data={data_path} task={task} framework={framework} size={args.size} "
        f"model={model_path} pretrained={pretrained or '(none)'}"
    )

    if framework == "rtdetr":
        trainer = RTDETR(model_path)
        if pretrained:
            trainer.load(pretrained)
    else:
        trainer = YOLO(model_path)
        if pretrained:
            trainer.load(pretrained)

    train_kw = dict(
        task=task,
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        cache=cache,
        lr0=lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
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
        mixup=args.mixup,
        seed=args.seed,
        amp=args.amp,
        cos_lr=args.cos_lr,
        patience=args.patience,
        resume=args.resume,
        exist_ok=True,
        deterministic=args.deterministic if args.deterministic is not None else (not args.fast),
        plots=args.plots if args.plots is not None else (not args.fast),
    )
    if framework == "yolo":
        if args.defect_preprocess != "none":
            train_kw["defect_preprocess"] = args.defect_preprocess
        elif preset_info.get("defect_preprocess"):
            train_kw["defect_preprocess"] = preset_info["defect_preprocess"]
        else:
            train_kw["defect_preprocess"] = "none"
        train_kw["copy_paste"] = args.copy_paste if args.copy_paste >= 0 else preset_info.get("copy_paste", 0.0)
        if task == "segment":
            train_kw["overlap_mask"] = (
                args.overlap_mask if args.overlap_mask is not None else preset_info.get("overlap_mask", True)
            )
            train_kw["mask_ratio"] = (
                args.mask_ratio if args.mask_ratio > 0 else preset_info.get("mask_ratio", 4)
            )
            if args.retina_masks is not None:
                train_kw["retina_masks"] = args.retina_masks
        ses_gain = args.ses_gain if args.ses_gain >= 0 else preset_info.get("ses_gain", 0.1)
        train_kw["ses_gain"] = ses_gain

    trainer.train(**train_kw)


if __name__ == "__main__":
    main()
