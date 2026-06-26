"""
LabelMe JSON → YOLO segmentation txt（polygon，归一化坐标）。

类别映射（默认）:
  point → 0
  line  → 1

示例:
  # 仅生成 txt
  python scripts/labelme_to_seg_txt.py ^
    --dir "G:/卡游切图/汇总_XY/AI_点压印/BIAOZHU" ^
    --out labels

  # 生成完整 YOLO seg 训练集（images/labels + data.yaml）
  python scripts/labelme_to_seg_txt.py ^
    --dir "G:/卡游切图/汇总_XY/AI_点压印/BIAOZHU" ^
    --yolo-out "G:/卡游切图/汇总_XY/AI_点压印/BIAOZHU_seg" ^
    --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

DEFAULT_CLASS_MAP = {
    "point": 0,
    "line": 1,
}


def find_image(parent: Path, stem: str, image_path_hint: str | None = None) -> Path | None:
    for ext in IMAGE_SUFFIXES:
        p = parent / f"{stem}{ext}"
        if p.is_file():
            return p
    if image_path_hint:
        p = parent / image_path_hint
        if p.is_file():
            return p
    return None


def image_hw(json_path: Path, data: dict) -> tuple[int, int] | None:
    h, w = data.get("imageHeight"), data.get("imageWidth")
    if h and w:
        return int(h), int(w)
    stem = json_path.stem
    parent = json_path.parent
    for ext in IMAGE_SUFFIXES:
        img = parent / f"{stem}{ext}"
        if img.is_file():
            im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if im is not None:
                return im.shape[0], im.shape[1]
    image_path = data.get("imagePath")
    if image_path:
        img = parent / image_path
        if img.is_file():
            im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if im is not None:
                return im.shape[0], im.shape[1]
    return None


def shape_to_seg_line(shape: dict, class_map: dict[str, int], w: int, h: int) -> str | None:
    label = str(shape.get("label", "")).strip().lower()
    if label not in class_map:
        return None
    cls_id = class_map[label]
    points = shape.get("points") or []
    if len(points) < 3:
        return None
    coords: list[str] = []
    for pt in points:
        if len(pt) < 2:
            continue
        x, y = float(pt[0]), float(pt[1])
        coords.append(f"{x / w:.6g}")
        coords.append(f"{y / h:.6g}")
    if len(coords) < 6:
        return None
    return f"{cls_id} " + " ".join(coords)


def convert_json(
    json_path: Path,
    out_dir: Path,
    class_map: dict[str, int],
    dry_run: bool = False,
) -> tuple[str, int, Counter]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return "bad_json", 0, Counter()

    shapes = data.get("shapes") or []
    if not shapes:
        return "empty", 0, Counter()

    hw = image_hw(json_path, data)
    if not hw:
        return "no_hw", 0, Counter()
    h, w = hw

    lines: list[str] = []
    lbl_cnt: Counter = Counter()
    skipped = 0
    for shape in shapes:
        line = shape_to_seg_line(shape, class_map, w, h)
        if line is None:
            skipped += 1
            continue
        lines.append(line)
        lbl = str(shape.get("label", "")).strip().lower()
        lbl_cnt[lbl] += 1

    if not lines:
        return "no_valid", skipped, lbl_cnt

    out_path = out_dir / f"{json_path.stem}.txt"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "ok", len(lines), lbl_cnt


def build_yolo_dataset(
    src: Path,
    yolo_root: Path,
    class_map: dict[str, int],
    val_ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """从 LabelMe json 生成 images/labels train/val + data.yaml。"""
    json_files = sorted(src.glob("*.json"))
    pairs: list[tuple[Path, list[str], Counter]] = []

    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        shapes = data.get("shapes") or []
        if not shapes:
            continue
        hw = image_hw(jp, data)
        if not hw:
            continue
        h, w = hw
        lines: list[str] = []
        lbl_cnt: Counter = Counter()
        for shape in shapes:
            line = shape_to_seg_line(shape, class_map, w, h)
            if line is None:
                continue
            lines.append(line)
            lbl = str(shape.get("label", "")).strip().lower()
            lbl_cnt[lbl] += 1
        if not lines:
            continue
        img = find_image(src, jp.stem, data.get("imagePath"))
        if img is None:
            print(f"[warn] no image for {jp.name}")
            continue
        pairs.append((img, lines, lbl_cnt))

    if not pairs:
        print("[error] no annotated image-label pairs")
        sys.exit(1)

    ratio = max(0.05, min(0.95, val_ratio))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(round(len(pairs) * ratio))
    if len(pairs) > 1:
        n_val = max(1, min(len(pairs) - 1, n_val))
    else:
        n_val = 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    lbl_total: Counter = Counter()
    for split, items in (("train", train_pairs), ("val", val_pairs)):
        for img, lines, lbl_cnt in items:
            lbl_total.update(lbl_cnt)
            if dry_run:
                continue
            img_dst = yolo_root / "images" / split / img.name
            lbl_dst = yolo_root / "labels" / split / f"{img.stem}.txt"
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, img_dst)
            lbl_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not dry_run:
        names = {class_map[k]: k for k in sorted(class_map, key=class_map.get)}
        data_yaml = {
            "path": str(yolo_root.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        yaml_path = yolo_root / "data.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"YOLO 数据集: {yolo_root}")
    print(f"  train images: {len(train_pairs)}")
    print(f"  val images:   {len(val_pairs)}")
    print(f"  polygons: point(0)={lbl_total.get('point', 0)}, line(1)={lbl_total.get('line', 0)}")
    if not dry_run:
        print(f"  data.yaml -> {yolo_root / 'data.yaml'}")
    if dry_run:
        print("[dry-run] 未写入文件")


def main() -> None:
    ap = argparse.ArgumentParser(description="LabelMe JSON → YOLO seg txt")
    ap.add_argument("--dir", type=Path, required=True, help="含 json/bmp 的目录")
    ap.add_argument(
        "--out",
        type=str,
        default="labels",
        help="输出子目录名（相对 --dir），或绝对路径",
    )
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写 txt")
    ap.add_argument(
        "--yolo-out",
        type=Path,
        default=None,
        help="输出 YOLO seg 训练目录（images/labels + data.yaml）",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2, help="val 比例，默认 0.2")
    ap.add_argument("--seed", type=int, default=42, help="train/val 划分随机种子")
    args = ap.parse_args()

    src = args.dir.resolve()
    class_map = dict(DEFAULT_CLASS_MAP)

    if args.yolo_out is not None:
        yolo_root = args.yolo_out.resolve()
        build_yolo_dataset(
            src,
            yolo_root,
            class_map,
            val_ratio=args.val_ratio,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        return

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = src / out_dir

    class_map = dict(DEFAULT_CLASS_MAP)
    json_files = sorted(src.glob("*.json"))
    if not json_files:
        print(f"[error] no json in {src}")
        sys.exit(1)

    stats = Counter()
    total_shapes = 0
    lbl_total: Counter = Counter()
    skipped_labels: Counter = Counter()

    for jp in json_files:
        status, n, lbl_cnt = convert_json(jp, out_dir, class_map, dry_run=args.dry_run)
        stats[status] += 1
        if status == "ok":
            total_shapes += n
            lbl_total.update(lbl_cnt)
        elif status in ("no_valid", "no_hw", "bad_json"):
            pass

    print(f"源目录: {src}")
    print(f"输出:   {out_dir}")
    print(f"JSON 总数: {len(json_files)}")
    print(f"  成功写出 txt: {stats['ok']}")
    print(f"  空 shapes:    {stats['empty']}")
    print(f"  无有效 polygon: {stats['no_valid']}")
    print(f"  无法读尺寸:   {stats['no_hw']}")
    print(f"  JSON 损坏:    {stats['bad_json']}")
    print(f"写出 polygon 总数: {total_shapes}")
    print(f"  point(0): {lbl_total.get('point', 0)}")
    print(f"  line(1):  {lbl_total.get('line', 0)}")
    if args.dry_run:
        print("[dry-run] 未写入文件")


if __name__ == "__main__":
    main()
