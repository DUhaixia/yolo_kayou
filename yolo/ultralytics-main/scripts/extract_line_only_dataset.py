"""
从数据集中提取「含 line 缺陷」训练集：
  - 只保留 label 中含 line 的 patch
  - 默认 --keep-all-labels：混有 point+line 时保留全部标注
  - 可选 --line-only-labels：混有 point+line 时只留 line
  - 可选过采样 line 样本

示例:
  python scripts/extract_line_only_dataset.py ^
    --src "M:/压印 - 副本/dataSet-原始" ^
    --out "M:/压印 - 副本/dataSet-原始-line_only" ^
    --sequential --clean --keep-all-labels
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


def bbox_line_to_seg(line: str) -> str:
    parts = line.split()
    cls_id = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return (
        f"{cls_id} {x1:.6g} {y1:.6g} {x2:.6g} {y2:.6g} "
        f"{x3:.6g} {y3:.6g} {x4:.6g} {y4:.6g}"
    )


def format_label_line(line: str, seg_format: bool) -> str:
    parts = line.split()
    if not seg_format or len(parts) > 5:
        return line
    return bbox_line_to_seg(line)


def parse_labels(
    text: str,
    line_class: int,
    keep_class_id: bool = True,
    seg_format: bool = False,
    keep_all_labels: bool = True,
) -> tuple[list[str], int, int, int]:
    """Return kept lines, n_line, n_point, n_other."""
    lines_raw: list[str] = []
    n_line = n_point = n_other = 0
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        cls = int(float(parts[0]))
        if cls == line_class:
            n_line += 1
        elif cls == 0:
            n_point += 1
        else:
            n_other += 1
        if not keep_class_id and cls == line_class:
            parts[0] = "0"
            ln = " ".join(parts)
        lines_raw.append(format_label_line(ln, seg_format))

    if n_line == 0:
        return [], n_line, n_point, n_other

    if keep_all_labels:
        return lines_raw, n_line, n_point, n_other

    kept = [ln for ln in lines_raw if ln.split()[0] == ("0" if not keep_class_id else str(line_class))]
    return kept, n_line, n_point + n_other, 0


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def clean_split_dirs(dst_root: Path, split: str) -> None:
    for sub in ("images", "labels"):
        d = dst_root / sub / split
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()


def load_names(src_root: Path, keep_class_id: bool) -> dict:
    yaml_path = src_root / "data.yaml"
    if keep_class_id and yaml_path.is_file():
        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names")
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    return {0: "line"}


def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    line_class: int,
    repeat: int,
    oversample: bool,
    keep_class_id: bool,
    seg_format: bool = False,
    keep_all_labels: bool = True,
    sequential: bool = False,
    prefix: str = "line",
    digits: int = 6,
    start_idx: int = 1,
) -> tuple[list[dict], int]:
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split

    rows: list[dict] = []
    n_out = n_skip = 0
    idx = start_idx

    for lbl in sorted(src_lbl.glob("*.txt")):
        kept, n_line, n_point, n_other = parse_labels(
            lbl.read_text(encoding="utf-8", errors="ignore"),
            line_class, keep_class_id, seg_format, keep_all_labels,
        )
        if n_line == 0:
            n_skip += 1
            continue

        img = find_image(src_img, lbl.stem)
        if img is None:
            print(f"[warn] no image: {lbl.name}")
            continue

        copies = 1 if not oversample or repeat <= 1 else repeat
        for k in range(copies):
            if sequential:
                stem = f"{prefix}_{idx:0{digits}d}"
                idx += 1
            else:
                suffix = "" if k == 0 else f"_lineDup{k:02d}"
                stem = f"{lbl.stem}{suffix}"
            out_img = dst_img / f"{stem}{img.suffix}"
            out_lbl = dst_lbl / f"{stem}.txt"
            copy_image(img, out_img)
            write_label(out_lbl, kept)
            n_out += 1
            row = {
                "split": split,
                "file": out_img.name,
                "tag": "base" if k == 0 else f"dup{k:02d}",
                "line_instances": n_line,
                "point_instances": n_point,
                "other_instances": n_other,
            }
            if sequential:
                row["source"] = lbl.stem
            rows.append(row)

    print(f"[{split}] line_patches={n_out}, skipped_no_line={n_skip}")
    return rows, idx


def main() -> None:
    p = argparse.ArgumentParser(description="提取仅含 line 缺陷的数据集")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--line-class", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1, help="line patch 复制份数，2=原图+1份复制")
    p.add_argument("--oversample-train", action="store_true", default=True)
    p.add_argument("--no-oversample-train", dest="oversample_train", action="store_false")
    p.add_argument("--oversample-val", action="store_true", default=False)
    p.add_argument("--sequential", action="store_true", help="连续编号命名，如 line_000001.bmp")
    p.add_argument("--prefix", default="line", help="连续编号前缀")
    p.add_argument("--digits", type=int, default=6, help="编号位数")
    p.add_argument("--keep-class-id", action="store_true", default=True,
                   help="保留原始类别 id（line 仍为 1，不重映射为 0）")
    p.add_argument("--remap-to-zero", dest="keep_class_id", action="store_false",
                   help="将 line 重映射为 class 0（单类数据集）")
    p.add_argument("--keep-all-labels", action="store_true", default=True,
                   help="含 line 的图保留全部标注（point+line 都保留）")
    p.add_argument("--line-only-labels", dest="keep_all_labels", action="store_false",
                   help="含 line 的图只保留 line 标注，去掉 point")
    p.add_argument("--seg-format", action="store_true",
                   help="输出分割多边形格式 (矩形四角)，用于 segment 训练")
    p.add_argument("--clean", action="store_true", help="提取前清空输出目录中的 train/val 图像和标注")
    args = p.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.out).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.clean:
        clean_split_dirs(dst_root, "train")
        clean_split_dirs(dst_root, "val")
        print("[clean] removed existing files in images/labels train & val")

    rows: list[dict] = []
    train_rows, next_idx = process_split(
        src_root, dst_root, "train", args.line_class, args.repeat, args.oversample_train,
        args.keep_class_id, args.seg_format, args.keep_all_labels,
        sequential=args.sequential, prefix=args.prefix, digits=args.digits,
    )
    rows.extend(train_rows)
    val_rows, _ = process_split(
        src_root, dst_root, "val", args.line_class, args.repeat, args.oversample_val,
        args.keep_class_id, args.seg_format, args.keep_all_labels,
        sequential=args.sequential, prefix=args.prefix, digits=args.digits, start_idx=next_idx,
    )
    rows.extend(val_rows)

    names = load_names(src_root, args.keep_class_id)
    data = {
        "path": str(dst_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml_path = dst_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    manifest = dst_root / "line_only_manifest.csv"
    fieldnames = ["split", "file", "tag", "line_instances", "point_instances", "other_instances"]
    if args.sequential:
        fieldnames.append("source")
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    tr = sum(1 for r in rows if r["split"] == "train")
    va = sum(1 for r in rows if r["split"] == "val")
    cls_desc = f"names={names}" if args.keep_class_id else "classes=[line@0]"
    print(f"done -> {dst_root}")
    print(f"train={tr}, val={va}, {cls_desc}")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
