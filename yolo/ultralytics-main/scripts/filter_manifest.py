"""
根据 manifest.csv 的 keep 列过滤裁切结果，生成可训练数据集。

在 Excel 中打开 manifest.csv:
  keep=1  保留
  keep=0  删除
  留空    默认删除

示例:
  python scripts/filter_manifest.py ^
    --manifest D:/dataset_center_crops/train/manifest.csv ^
    --dst-root D:/dataset_filtered/train
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser(description="Filter cropped dataset by manifest keep column")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dst-root", required=True, help="filtered output: images/ labels/")
    p.add_argument("--src-root", default="", help="crop root with images/labels; default = manifest parent")
    args = p.parse_args()

    manifest = Path(args.manifest)
    src_root = Path(args.src_root) if args.src_root else manifest.parent
    src_img = src_root / "images"
    src_lbl = src_root / "labels"
    dst_img = Path(args.dst_root) / "images"
    dst_lbl = Path(args.dst_root) / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    with manifest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keep = str(row.get("keep", "")).strip()
            if keep not in ("1", "true", "True", "yes", "Y"):
                skipped += 1
                continue
            fname = row["filename"]
            stem = Path(fname).stem
            img_src = src_img / fname
            if not img_src.exists():
                for ext in IMAGE_SUFFIXES:
                    alt = src_img / f"{stem}{ext}"
                    if alt.exists():
                        img_src = alt
                        fname = alt.name
                        break
            lbl_src = src_lbl / f"{stem}.txt"
            if not img_src.exists() or not lbl_src.exists():
                print(f"[warn] missing pair for {stem}")
                continue
            shutil.copy2(img_src, dst_img / fname)
            shutil.copy2(lbl_src, dst_lbl / f"{stem}.txt")
            kept += 1

    print(f"kept={kept}, skipped={skipped}")
    print(f"output -> {args.dst_root}")


if __name__ == "__main__":
    main()
