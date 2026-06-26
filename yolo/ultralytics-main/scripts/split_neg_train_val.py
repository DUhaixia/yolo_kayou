"""
将筛选好的负样本/微调图片按 7:3 划分 train/val，并生成空 label + data.yaml。

示例:
  python scripts/split_neg_train_val.py ^
    --images "M:/压印 - 副本/neg_edge_bg/images/train/train" ^
    --labels "M:/压印 - 副本/neg_edge_bg/labels/train" ^
    --out "M:/压印 - 副本/neg_edge_bg_split" ^
    --ratio 0.7
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_pair(img: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> None:
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_dst)
    if lbl_src.is_file():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="划分 train/val 并生成 YOLO 目录")
    p.add_argument("--images", required=True, help="筛选后的图片目录")
    p.add_argument("--labels", required=True, help="对应 label 目录（负样本可为空 txt）")
    p.add_argument("--out", required=True, help="输出数据集根目录")
    p.add_argument("--ratio", type=float, default=0.7, help="train 比例，默认 0.7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--names", default="point,line", help="data.yaml 类别名，逗号分隔")
    args = p.parse_args()

    img_dir = Path(args.images).resolve()
    lbl_dir = Path(args.labels).resolve()
    out_root = Path(args.out).resolve()
    ratio = max(0.05, min(0.95, args.ratio))

    images = list_images(img_dir)
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_train = int(round(len(images) * ratio))
    n_train = max(1, min(len(images) - 1, n_train)) if len(images) > 1 else 1
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    splits = {"train": train_imgs, "val": val_imgs}
    stats = {}

    for split, items in splits.items():
        copied = 0
        for img in items:
            lbl_src = lbl_dir / f"{img.stem}.txt"
            img_dst = out_root / "images" / split / img.name
            lbl_dst = out_root / "labels" / split / f"{img.stem}.txt"
            copy_pair(img, lbl_src, img_dst, lbl_dst)
            copied += 1
        stats[split] = copied
        print(f"{split}: {copied}")

    names = [s.strip() for s in args.names.split(",") if s.strip()]
    data = {
        "path": str(out_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    yaml_path = out_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"total={len(images)}, train={stats['train']}, val={stats['val']}")
    print(f"out -> {out_root}")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
