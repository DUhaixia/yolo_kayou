"""
正确合并：line_only 数据集 + 负样本（空 label）。
  - 不重复拷贝已在原集中的样本
  - 不混入 point 标注
  - 删除 images 中的 .npy 等非图片文件
  - 负样本 label 为空

示例:
  python scripts/merge_line_neg_clean.py ^
    --line "M:/压印 - 副本/dataSet-原始-切割-已筛选-line_only" ^
    --neg "M:/压印 - 副本/neg_edge_bg_split2" ^
    --out "M:/压印 - 副本/dataSet-line_only+neg_clean"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_SUFFIXES:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def copy_split_line(line_root: Path, neg_root: Path, dst_root: Path, split: str) -> dict:
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    n_line = n_neg = n_skip = 0

    line_lbl = line_root / "labels" / split
    line_img = line_root / "images" / split
    if line_lbl.is_dir():
        for lbl in sorted(line_lbl.glob("*.txt")):
            stem = lbl.stem
            if stem in used:
                n_skip += 1
                continue
            img = find_image(line_img, stem)
            if img is None:
                print(f"[warn] line missing image: {lbl.name}")
                continue
            shutil.copy2(img, dst_img / img.name)
            shutil.copy2(lbl, dst_lbl / lbl.name)
            used.add(stem)
            n_line += 1

    neg_lbl = neg_root / "labels" / split
    neg_img = neg_root / "images" / split
    if neg_lbl.is_dir():
        for lbl in sorted(neg_lbl.glob("*.txt")):
            stem = lbl.stem
            if stem in used:
                n_skip += 1
                continue
            img = find_image(neg_img, stem)
            if img is None:
                print(f"[warn] neg missing image: {lbl.name}")
                continue
            shutil.copy2(img, dst_img / img.name)
            (dst_lbl / lbl.name).write_text("", encoding="utf-8")
            used.add(stem)
            n_neg += 1

    return {"line": n_line, "neg": n_neg, "skip_dup": n_skip, "total": n_line + n_neg}


def main() -> None:
    p = argparse.ArgumentParser(description="干净合并 line_only + 负样本")
    p.add_argument("--line", required=True)
    p.add_argument("--neg", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    line_root = Path(args.line).resolve()
    neg_root = Path(args.neg).resolve()
    dst_root = Path(args.out).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ("train", "val"):
        if (line_root / "labels" / split).is_dir() or (neg_root / "labels" / split).is_dir():
            stats[split] = copy_split_line(line_root, neg_root, dst_root, split)
            print(f"[{split}] {stats[split]}")

    data = {
        "path": str(dst_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "line"},
    }
    with (dst_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"done -> {dst_root}")
    print(f"data.yaml -> {dst_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
