"""
从已裁切数据集中找出含 line 标注的样本，复制多份以提升 line 召回训练权重。

默认只过采样 train，val 保持原样（验证指标更可信）。

示例:
  python scripts/oversample_line_dataset.py ^
    --src "M:/压印 - 副本/dataSet-原始-切割-已筛选" ^
    --out "M:/压印 - 副本/dataSet-原始-切割-已筛选-line2x" ^
    --repeat 2 ^
    --line-class 1
"""

from __future__ import annotations

import argparse
import csv
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


def label_stats(label_path: Path, line_class: int) -> tuple[bool, bool, int]:
    has_point = has_line = False
    line_n = 0
    if not label_path.is_file():
        return has_point, has_line, line_n
    for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        cls = int(float(ln.split()[0]))
        if cls == line_class:
            has_line = True
            line_n += 1
        elif cls == 0:
            has_point = True
    return has_point, has_line, line_n


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> None:
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)


def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    line_class: int,
    repeat: int,
    oversample: bool,
) -> list[dict]:
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split

    rows: list[dict] = []
    n_base = n_dup = 0
    line_related = 0

    for lbl in sorted(src_lbl.glob("*.txt")):
        stem = lbl.stem
        img = find_image(src_img, stem)
        if img is None:
            print(f"[warn] no image for {lbl.name}")
            continue

        has_point, has_line, line_n = label_stats(lbl, line_class)
        copy_pair(img, lbl, dst_img / img.name, dst_lbl / lbl.name)
        n_base += 1
        rows.append({
            "split": split,
            "file": img.name,
            "tag": "base",
            "has_point": int(has_point),
            "has_line": int(has_line),
            "line_instances": line_n,
        })

        if oversample and has_line and repeat > 1:
            line_related += 1
            for k in range(1, repeat):
                dup_stem = f"{stem}_lineDup{k:02d}"
                dup_img_name = f"{dup_stem}{img.suffix}"
                copy_pair(
                    img, lbl,
                    dst_img / dup_img_name,
                    dst_lbl / f"{dup_stem}.txt",
                )
                n_dup += 1
                rows.append({
                    "split": split,
                    "file": dup_img_name,
                    "tag": f"dup{k:02d}",
                    "has_point": int(has_point),
                    "has_line": int(has_line),
                    "line_instances": line_n,
                })

    print(f"[{split}] base={n_base}, line_related={line_related}, added_dup={n_dup}, total={n_base + n_dup}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="过采样含 line 标注的训练样本")
    p.add_argument("--src", required=True, help="原始裁切数据集根目录")
    p.add_argument("--out", required=True, help="输出数据集根目录")
    p.add_argument("--repeat", type=int, default=2, help="line 样本总份数，2=原图+复制1份")
    p.add_argument("--line-class", type=int, default=1)
    p.add_argument("--oversample-train", action="store_true", default=True)
    p.add_argument("--no-oversample-train", dest="oversample_train", action="store_false")
    p.add_argument("--oversample-val", action="store_true", default=False)
    args = p.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat 必须 >= 1")

    src_root = Path(args.src).resolve()
    dst_root = Path(args.out).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_rows.extend(process_split(
        src_root, dst_root, "train", args.line_class, args.repeat, args.oversample_train,
    ))
    all_rows.extend(process_split(
        src_root, dst_root, "val", args.line_class, args.repeat, args.oversample_val,
    ))

    src_yaml = src_root / "data.yaml"
    names = {0: "point", 1: "line"}
    if src_yaml.is_file():
        cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and cfg.get("names"):
            names = cfg["names"]

    data = {
        "path": str(dst_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml_path = dst_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    manifest = dst_root / "line_oversample_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "tag", "has_point", "has_line", "line_instances"])
        writer.writeheader()
        writer.writerows(all_rows)

    train_total = sum(1 for r in all_rows if r["split"] == "train")
    train_line = sum(1 for r in all_rows if r["split"] == "train" and r["has_line"])
    print(f"done -> {dst_root}")
    print(f"train total={train_total}, train line-related={train_line}")
    print(f"data.yaml -> {yaml_path}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
