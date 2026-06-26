"""
按规则自动填写 manifest keep 列，并输出筛选后数据集。

默认规则:
  - point(0): 全部保留
  - line(1): primary_len_px >= 15 保留，否则删除
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def apply_rules(row: dict, min_line_len: float) -> tuple[str, str]:
    cls_id = int(row["primary_class"])
    note = ""
    if cls_id == 0:
        return "1", "point-keep"
    seg_len = float(row.get("primary_len_px") or 0)
    if seg_len >= min_line_len:
        trunc = int(row.get("line_truncated") or 0)
        note = "line-keep" + ("-truncated" if trunc else "")
        return "1", note
    return "0", f"line-drop-len<{min_line_len}"


def process_manifest(manifest: Path, src_root: Path, dst_root: Path, split: str, min_line_len: float):
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    if not rows:
        return 0, 0

    kept = dropped = 0
    for row in rows:
        keep, note = apply_rules(row, min_line_len)
        row["keep"] = keep
        row["note"] = note
        if keep == "1":
            kept += 1
        else:
            dropped += 1

    # save annotated manifest back
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split

    for row in rows:
        if row["keep"] != "1":
            continue
        stem = Path(row["filename"]).stem
        img_src = src_img / row["filename"]
        if not img_src.exists():
            for ext in IMAGE_SUFFIXES:
                alt = src_img / f"{stem}{ext}"
                if alt.exists():
                    img_src = alt
                    break
        lbl_src = src_lbl / f"{stem}.txt"
        if img_src.exists() and lbl_src.exists():
            shutil.copy2(img_src, dst_img / img_src.name)
            shutil.copy2(lbl_src, dst_lbl / f"{stem}.txt")

    return kept, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", default=r"M:\压印 - 副本\dataSet-原始-切割")
    p.add_argument("--dst-root", default=r"M:\压印 - 副本\dataSet-原始-切割-已筛选")
    p.add_argument("--min-line-len", type=float, default=15.0)
    args = p.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val"):
        mf = src_root / f"manifest_{split}.csv"
        kept, dropped = process_manifest(mf, src_root, dst_root, split, args.min_line_len)
        summary[split] = (kept, dropped)

    yaml_text = f"""path: {dst_root}
train: images/train
val: images/val
names:
  0: point
  1: line
"""
    (dst_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("auto filter done")
    for split, (k, d) in summary.items():
        print(f"  {split}: keep={k}, drop={d}")
    print(f"output -> {dst_root}")


if __name__ == "__main__":
    main()
