"""
从整图 + YOLO-Seg 标注构建固定滑窗 512 训练集（用于训练，与产线 SAHI 一致）。

设计原则:
  - 只用固定网格滑窗切图，不用缺陷中心裁切
  - 正样本 tile（含标注）与负样本 tile（空 label）分目录存放
  - 支持额外硬负样本（好品/有图案无缺陷图）
  - 可按 pos:neg 比例下采样负样本（默认 3:7）
  - 生成 merged 目录供 YOLO 直接训练 + data.yaml

目录结构:
  out_root/
    images/train          正样本 tile
    labels/train
    images_neg/train      滑窗负样本 tile
    labels_neg/train
    images_hard_neg/train 硬负样本 tile（可选）
    labels_hard_neg/train
    merged/images/train   合并后训练集（pos + 采样 neg + hard neg）
    merged/labels/train
    manifest.csv
    data.yaml

示例:
  python scripts/build_tile_seg_dataset.py ^
    --src-root "M:/压印 - 副本/dataSet-原始" ^
    --out-root "M:/压印 - 副本/dataSet-tile512" ^
    --patch 512 --stride 384 ^
    --neg-ratio 0.7 ^
    --hard-neg-images "M:/压印 - 副本/好品/images/train" ^
    --hard-neg-images-val "M:/压印 - 副本/好品/images/val"
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import slide_crop_dataset as scd  # noqa: E402


@dataclass
class SplitStats:
    source_images: int = 0
    pos_tiles: int = 0
    neg_tiles_raw: int = 0
    neg_tiles_kept: int = 0
    hard_neg_tiles: int = 0
    merged_tiles: int = 0
    label_rows: int = 0
    skipped_images: int = 0


@dataclass
class BuildStats:
    train: SplitStats = field(default_factory=SplitStats)
    val: SplitStats = field(default_factory=SplitStats)


def parse_ratio(text: str) -> float:
    text = text.strip()
    if ":" in text:
        a, b = text.split(":", 1)
        pos, neg = float(a), float(b)
        if pos <= 0 or neg < 0:
            raise ValueError(f"invalid ratio: {text}")
        return neg / (pos + neg)
    val = float(text)
    if not 0.0 <= val < 1.0:
        raise ValueError(f"neg-ratio must be in [0,1), got {val}")
    return val


def ensure_empty_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_tiles_for_image(
    img_path: Path,
    label_path: Path,
    patch: int,
    stride: int,
    line_policy: str,
    min_line_len: float,
    min_area: float,
    line_class_ids: set[int],
) -> tuple[list[tuple[str, object]], list[tuple[str, object]]]:
    """Return (positive_tiles, negative_tiles), each item = (stem, patch_bgr)."""
    im = scd.imread_unicode(img_path)
    if im is None:
        return [], []

    h, w = im.shape[:2]
    labels = scd.load_labels(label_path)
    windows = scd.sliding_coords(h, w, patch, stride)

    pos_items: list[tuple[str, object, list[str]]] = []
    neg_items: list[tuple[str, object]] = []

    for yi, xi in windows:
        tile = im[yi : yi + patch, xi : xi + patch]
        out_labels: list[str] = []
        for cls_id, fmt, coords in labels:
            row = scd.transform_label(
                cls_id,
                fmt,
                coords,
                w,
                h,
                xi,
                yi,
                patch,
                patch,
                line_policy,
                min_line_len,
                min_area,
                line_class_ids,
            )
            if row is not None:
                out_labels.append(scd.format_label(row))

        stem = f"{img_path.stem}_y{yi}_x{xi}"
        if out_labels:
            pos_items.append((stem, tile, out_labels))
        else:
            neg_items.append((stem, tile))

    return pos_items, neg_items


def subsample_negatives(
    neg_items: list[tuple[str, object]],
    pos_count: int,
    neg_ratio: float,
    max_neg_per_image: int,
    rng: random.Random,
) -> list[tuple[str, object]]:
    if not neg_items or pos_count <= 0 or neg_ratio <= 0:
        return []

    target = int(round(pos_count * neg_ratio / max(1e-6, 1.0 - neg_ratio)))
    if max_neg_per_image > 0:
        target = min(target, max_neg_per_image * max(1, pos_count))

    if len(neg_items) <= target:
        return neg_items
    return rng.sample(neg_items, target)


def save_tiles(
    items: list,
    img_dir: Path,
    lbl_dir: Path,
    *,
    with_labels: bool,
) -> int:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in items:
        if with_labels:
            stem, tile, lines = item
            scd.imwrite_unicode(img_dir / f"{stem}.png", tile)
            write_label(lbl_dir / f"{stem}.txt", lines)
            n += 1
        else:
            stem, tile = item
            scd.imwrite_unicode(img_dir / f"{stem}.png", tile)
            write_label(lbl_dir / f"{stem}.txt", [])
            n += 1
    return n


def copy_into_merged(
    src_img_dir: Path,
    src_lbl_dir: Path,
    merged_img_dir: Path,
    merged_lbl_dir: Path,
) -> int:
    if not src_img_dir.is_dir():
        return 0
    merged_img_dir.mkdir(parents=True, exist_ok=True)
    merged_lbl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_img_dir.iterdir()):
        if img.suffix.lower() not in scd.IMAGE_SUFFIXES:
            continue
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        shutil.copy2(img, merged_img_dir / img.name)
        shutil.copy2(lbl, merged_lbl_dir / lbl.name)
        count += 1
    return count


def process_labeled_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    out_root: Path,
    split: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> SplitStats:
    stats = SplitStats()
    if not src_img_dir.is_dir():
        print(f"[warn] missing labeled images: {src_img_dir}")
        return stats

    pos_root = out_root / "images" / split
    pos_lbl_root = out_root / "labels" / split
    neg_root = out_root / "images_neg" / split
    neg_lbl_root = out_root / "labels_neg" / split

    ensure_empty_dir(pos_root, args.clean)
    ensure_empty_dir(pos_lbl_root, args.clean)
    ensure_empty_dir(neg_root, args.clean)
    ensure_empty_dir(neg_lbl_root, args.clean)

    line_ids = {int(x) for x in args.line_classes.split(",") if x.strip()}

    all_pos: list = []
    all_neg_raw: list[tuple[str, object]] = []
    per_image_neg: dict[str, list[tuple[str, object]]] = {}

    images = sorted(p for p in src_img_dir.iterdir() if p.suffix.lower() in scd.IMAGE_SUFFIXES)
    stats.source_images = len(images)

    for img_path in images:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        pos_items, neg_items = collect_tiles_for_image(
            img_path,
            label_path,
            args.patch,
            args.stride,
            args.line_policy,
            args.min_line_len,
            args.min_area,
            line_ids,
        )
        if not pos_items and not neg_items:
            stats.skipped_images += 1
            continue

        all_pos.extend(pos_items)
        all_neg_raw.extend(neg_items)
        per_image_neg[img_path.stem] = neg_items

    stats.pos_tiles = len(all_pos)
    stats.neg_tiles_raw = len(all_neg_raw)
    stats.label_rows = sum(len(x[2]) for x in all_pos)

    # 按整批 pos:neg 下采样；也可改成逐图下采样
    if args.per_image_neg_cap > 0:
        kept_neg: list[tuple[str, object]] = []
        for img_path in images:
            neg_items = per_image_neg.get(img_path.stem, [])
            pos_count = sum(1 for x in all_pos if x[0].startswith(f"{img_path.stem}_y"))
            kept_neg.extend(
                subsample_negatives(
                    neg_items,
                    pos_count,
                    args.neg_ratio,
                    args.per_image_neg_cap,
                    rng,
                )
            )
        neg_kept = kept_neg
    else:
        neg_kept = subsample_negatives(
            all_neg_raw,
            stats.pos_tiles,
            args.neg_ratio,
            0,
            rng,
        )

    stats.neg_tiles_kept = len(neg_kept)

    save_tiles(all_pos, pos_root, pos_lbl_root, with_labels=True)
    save_tiles(neg_kept, neg_root, neg_lbl_root, with_labels=False)

    print(
        f"[{split}] source={stats.source_images} pos={stats.pos_tiles} "
        f"neg_raw={stats.neg_tiles_raw} neg_kept={stats.neg_tiles_kept} "
        f"labels={stats.label_rows}"
    )
    return stats


def process_hard_neg_split(
    hard_img_dir: Path,
    out_root: Path,
    split: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> int:
    if not hard_img_dir or not hard_img_dir.is_dir():
        return 0

    hard_root = out_root / "images_hard_neg" / split
    hard_lbl_root = out_root / "labels_hard_neg" / split
    ensure_empty_dir(hard_root, args.clean)
    ensure_empty_dir(hard_lbl_root, args.clean)

    line_ids = set()  # unused
    tiles: list[tuple[str, object]] = []

    images = sorted(p for p in hard_img_dir.iterdir() if p.suffix.lower() in scd.IMAGE_SUFFIXES)
    for img_path in images:
        pos_items, neg_items = collect_tiles_for_image(
            img_path,
            img_path.with_suffix(".txt"),  # empty/nonexistent -> all neg
            args.patch,
            args.stride,
            args.line_policy,
            args.min_line_len,
            args.min_area,
            line_ids,
        )
        _ = pos_items
        if args.hard_neg_max_per_image > 0 and len(neg_items) > args.hard_neg_max_per_image:
            neg_items = rng.sample(neg_items, args.hard_neg_max_per_image)
        tiles.extend(neg_items)

    n = save_tiles(tiles, hard_root, hard_lbl_root, with_labels=False)
    print(f"[{split}] hard_neg={n} from {len(images)} images")
    return n


def build_merged_split(out_root: Path, split: str, stats: SplitStats) -> int:
    merged_img = out_root / "merged" / "images" / split
    merged_lbl = out_root / "merged" / "labels" / split
    ensure_empty_dir(merged_img, True)
    ensure_empty_dir(merged_lbl, True)

    n = 0
    n += copy_into_merged(out_root / "images" / split, out_root / "labels" / split, merged_img, merged_lbl)
    n += copy_into_merged(out_root / "images_neg" / split, out_root / "labels_neg" / split, merged_img, merged_lbl)
    n += copy_into_merged(
        out_root / "images_hard_neg" / split,
        out_root / "labels_hard_neg" / split,
        merged_img,
        merged_lbl,
    )
    stats.merged_tiles = n
    print(f"[{split}] merged={n}")
    return n


def write_data_yaml(out_root: Path, class_names: list[str]) -> None:
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    content = "\n".join(
        [
            f"path: {out_root.resolve().as_posix()}",
            "train: merged/images/train",
            "val: merged/images/val",
            "names:",
            names_block,
            "",
            "# Train with:",
            "# python scripts/train_seg_defect.py --data <this_file> --imgsz 512 --mosaic 0",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(content, encoding="utf-8")


def write_manifest(out_root: Path, stats: BuildStats, args: argparse.Namespace) -> None:
    path = out_root / "manifest.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "source_images",
                "pos_tiles",
                "neg_tiles_raw",
                "neg_tiles_kept",
                "hard_neg_tiles",
                "merged_tiles",
                "label_rows",
                "patch",
                "stride",
                "neg_ratio",
            ]
        )
        for split_name, s in (("train", stats.train), ("val", stats.val)):
            w.writerow(
                [
                    split_name,
                    s.source_images,
                    s.pos_tiles,
                    s.neg_tiles_raw,
                    s.neg_tiles_kept,
                    s.hard_neg_tiles,
                    s.merged_tiles,
                    s.label_rows,
                    args.patch,
                    args.stride,
                    args.neg_ratio,
                ]
            )
    print(f"manifest -> {path}")


def resolve_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / "images" / split, root / "labels" / split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed-grid 512 tile YOLO-Seg training dataset")
    p.add_argument("--src-root", required=True, help="Source dataset root with images/{train,val} labels/{train,val}")
    p.add_argument("--out-root", required=True, help="Output dataset root")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--stride", type=int, default=384, help="Training tile stride; inference can still use 256")
    p.add_argument("--neg-ratio", default="3:7", help="Negative ratio, e.g. 3:7 or 0.7")
    p.add_argument("--per-image-neg-cap", type=int, default=0, help="Max kept neg tiles per source image (0=global cap only)")
    p.add_argument("--hard-neg-images", default="", help="Hard negative image dir for train split")
    p.add_argument("--hard-neg-images-val", default="", help="Hard negative image dir for val split")
    p.add_argument("--hard-neg-max-per-image", type=int, default=12, help="Max hard-neg tiles per source image")
    p.add_argument("--line-policy", choices=["keep", "drop"], default="keep")
    p.add_argument("--line-classes", default="1")
    p.add_argument("--min-line-len", type=float, default=4.0)
    p.add_argument("--min-area", type=float, default=2.0)
    p.add_argument("--class-names", default="point,line")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true", help="Remove existing output split dirs before writing")
    p.add_argument("--no-merge", action="store_true", help="Skip merged/ output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.neg_ratio = parse_ratio(args.neg_ratio)

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats = BuildStats()
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]

    print("=== build_tile_seg_dataset ===")
    print(f"src={src_root}")
    print(f"out={out_root}")
    print(f"patch={args.patch} stride={args.stride} neg_ratio={args.neg_ratio:.3f}")

    for split in ("train", "val"):
        src_img, src_lbl = resolve_split_dirs(src_root, split)
        s = process_labeled_split(src_img, src_lbl, out_root, split, args, rng)
        if split == "train":
            stats.train = s
        else:
            stats.val = s

    hard_train = Path(args.hard_neg_images).resolve() if args.hard_neg_images else None
    hard_val = Path(args.hard_neg_images_val).resolve() if args.hard_neg_images_val else hard_train
    stats.train.hard_neg_tiles = process_hard_neg_split(hard_train, out_root, "train", args, rng)
    stats.val.hard_neg_tiles = process_hard_neg_split(hard_val, out_root, "val", args, rng)

    if not args.no_merge:
        build_merged_split(out_root, "train", stats.train)
        build_merged_split(out_root, "val", stats.val)

    write_data_yaml(out_root, class_names)
    write_manifest(out_root, stats, args)

    total_pos = stats.train.pos_tiles + stats.val.pos_tiles
    total_neg = stats.train.neg_tiles_kept + stats.val.neg_tiles_kept
    total_hard = stats.train.hard_neg_tiles + stats.val.hard_neg_tiles
    total_merged = stats.train.merged_tiles + stats.val.merged_tiles
    print("=== done ===")
    print(f"pos={total_pos} neg_kept={total_neg} hard_neg={total_hard} merged={total_merged}")
    print(f"data.yaml -> {out_root / 'data.yaml'}")
    print("Train:")
    print(f"  python scripts/train_seg_defect.py --data \"{out_root / 'data.yaml'}\" --imgsz {args.patch} --mosaic 0")


if __name__ == "__main__":
    main()
