"""
从已切 tile 数据集（仅 train 桶或任意单桶）生成:
  - train / val 划分（按原图，避免同一原图泄漏到两个集合）
  - 缺陷(正) + 好品(负) 在 train/val 中均保持目标比例
  - 每张原图均匀聚类采样负样本
  - merged/ + data.yaml

输入（当前 dataSet-tile512 典型结构）:
  dataset-root/
    images/train      有缺陷 tile
    labels/train
    images_neg/train  无缺陷 tile
    labels_neg/train

输出:
  out-root/
    images/{train,val}          缺陷 tile
    labels/{train,val}
    images_neg/{train,val}      采样后好品 tile
    labels_neg/{train,val}
    merged/images/{train,val}   YOLO 训练用（缺陷+好品）
    merged/labels/{train,val}
    data.yaml
    split_manifest.csv

示例:
  python scripts/split_balance_tile_dataset.py ^
    --dataset-root "M:/压印 - 副本/dataSet-tile512" ^
    --out-root "M:/压印 - 副本/dataSet-tile512_balanced" ^
    --val-ratio 0.2 ^
    --pos-neg-ratio 3:7 ^
    --method cluster ^
    --neg-per-image-fallback 4
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import stat
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import sample_neg_tiles as snt  # noqa: E402


@dataclass
class SplitBucketStats:
    source_images: int = 0
    pos_tiles: int = 0
    neg_tiles_raw: int = 0
    neg_tiles_kept: int = 0


@dataclass
class JobStats:
    train: SplitBucketStats = field(default_factory=SplitBucketStats)
    val: SplitBucketStats = field(default_factory=SplitBucketStats)


def parse_pos_neg_ratio(text: str) -> tuple[float, float]:
    text = text.strip()
    if ":" in text:
        a, b = text.split(":", 1)
        pos, neg = float(a), float(b)
        if pos <= 0 or neg < 0:
            raise ValueError(f"invalid pos-neg-ratio: {text}")
        return pos, neg
    raise ValueError("use format like 3:7")


def load_tile_groups(dataset_root: Path, bucket: str) -> tuple[dict, dict]:
    """Read pos/neg tiles from images/<bucket> and images_neg/<bucket>."""
    pos_img = dataset_root / "images" / bucket
    pos_lbl = dataset_root / "labels" / bucket
    neg_img = dataset_root / "images_neg" / bucket
    neg_lbl = dataset_root / "labels_neg" / bucket

    pos_groups = snt.group_by_source(snt.list_tiles(pos_img))
    neg_groups = snt.group_by_source(snt.list_tiles(neg_img))
    return (
        {"img": pos_img, "lbl": pos_lbl, "groups": pos_groups},
        {"img": neg_img, "lbl": neg_lbl, "groups": neg_groups},
    )


def stratified_split_sources(
    all_sources: list[str],
    pos_groups: dict[str, list[Path]],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """按原图分层划分: 有缺陷原图 / 纯好品原图 各自按比例切 train/val。"""
    defect_src = [s for s in all_sources if len(pos_groups.get(s, [])) > 0]
    good_only_src = [s for s in all_sources if len(pos_groups.get(s, [])) == 0]

    def split_bucket(items: list[str]) -> tuple[list[str], list[str]]:
        items = items[:]
        rng.shuffle(items)
        if not items:
            return [], []
        if len(items) == 1:
            return items, []
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)
        val_items = items[:n_val]
        train_items = items[n_val:]
        return train_items, val_items

    tr_d, va_d = split_bucket(defect_src)
    tr_g, va_g = split_bucket(good_only_src)
    train_src = sorted(tr_d + tr_g)
    val_src = sorted(va_d + va_g)
    return train_src, val_src


def target_neg_k(pos_count: int, args: argparse.Namespace) -> int:
    if args.neg_per_image > 0:
        return args.neg_per_image
    if pos_count <= 0:
        return args.neg_per_image_fallback
    pos_w, neg_w = args.pos_neg
    return max(1, int(round(pos_count * neg_w / pos_w)))


def global_trim_neg_to_ratio(
    picked_by_src: dict[str, list[Path]],
    pos_count: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, list[Path]]:
    """若每张图独立采样后全局比例偏离目标，对负样本做二次均匀裁剪。"""
    if pos_count <= 0:
        return picked_by_src

    pos_w, neg_w = args.pos_neg
    target_neg = int(round(pos_count * neg_w / pos_w))
    all_neg = [p for paths in picked_by_src.values() for p in paths]
    if len(all_neg) <= target_neg:
        return picked_by_src

    rng.shuffle(all_neg)
    keep_set = set(all_neg[:target_neg])
    trimmed: dict[str, list[Path]] = {}
    for src, paths in picked_by_src.items():
        trimmed[src] = [p for p in paths if p in keep_set]
    return trimmed


def process_bucket(
    split_name: str,
    sources: list[str],
    pos_pack: dict,
    neg_pack: dict,
    out_root: Path,
    args: argparse.Namespace,
    rng: random.Random,
    stats: SplitBucketStats,
) -> None:
    pick_fn = snt.cluster_pick if args.method == "cluster" else snt.random_pick

    out_pos_img = out_root / "images" / split_name
    out_pos_lbl = out_root / "labels" / split_name
    out_neg_img = out_root / "images_neg" / split_name
    out_neg_lbl = out_root / "labels_neg" / split_name
    for d in (out_pos_img, out_pos_lbl, out_neg_img, out_neg_lbl):
        d.mkdir(parents=True, exist_ok=True)

    pos_groups = pos_pack["groups"]
    neg_groups = neg_pack["groups"]
    stats.source_images = len(sources)

    picked_by_src: dict[str, list[Path]] = {}
    meta_by_src: dict[str, tuple[int, int, int]] = {}

    # 1) 复制全部缺陷 tile
    for src in sources:
        for img_path in pos_groups.get(src, []):
            snt.copy_tile(img_path, pos_pack["lbl"], out_pos_img, out_pos_lbl)
            stats.pos_tiles += 1

    # 2) 每张原图聚类采样好品 tile
    for i, src in enumerate(sources, start=1):
        pos_count = len(pos_groups.get(src, []))
        neg_paths = neg_groups.get(src, [])
        stats.neg_tiles_raw += len(neg_paths)
        k = min(target_neg_k(pos_count, args), len(neg_paths)) if neg_paths else 0
        picked = pick_fn(neg_paths, k, rng) if k > 0 else []
        picked_by_src[src] = picked
        meta_by_src[src] = (pos_count, len(neg_paths), k)
        if i % 300 == 0 or i == len(sources):
            print(f"  [{split_name}] sample {i}/{len(sources)}")

    # 3) 全局比例修正（train/val 各自独立）
    if args.global_balance and stats.pos_tiles > 0:
        picked_by_src = global_trim_neg_to_ratio(picked_by_src, stats.pos_tiles, args, rng)

    manifest_rows = []
    for src in sources:
        picked = picked_by_src.get(src, [])
        pos_count, neg_raw, target_k = meta_by_src.get(src, (0, 0, 0))
        for img_path in picked:
            snt.copy_tile(img_path, neg_pack["lbl"], out_neg_img, out_neg_lbl)
            stats.neg_tiles_kept += 1
        manifest_rows.append([split_name, src, pos_count, neg_raw, target_k, len(picked)])

    manifest_path = out_root / f"manifest_{split_name}.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "source_image", "pos_tiles", "neg_raw", "target_k", "neg_kept"])
        w.writerows(manifest_rows)
    print(f"  [{split_name}] manifest -> {manifest_path}")


def _on_rm_error(func, path, _exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def safe_rmtree(path: Path, retries: int = 5) -> None:
    if not path.exists():
        return
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            if not path.exists():
                return
        except OSError:
            pass
        time.sleep(0.5 * (attempt + 1))
    if path.exists():
        raise OSError(f"failed to remove {path}")


def build_merged(out_root: Path, split_name: str) -> tuple[int, int]:
    merged_img = out_root / "merged" / "images" / split_name
    merged_lbl = out_root / "merged" / "labels" / split_name
    safe_rmtree(merged_img)
    safe_rmtree(merged_lbl)

    n_pos = snt.copy_all_tiles(
        out_root / "images" / split_name,
        out_root / "labels" / split_name,
        merged_img,
        merged_lbl,
    )
    n_neg = snt.copy_all_tiles(
        out_root / "images_neg" / split_name,
        out_root / "labels_neg" / split_name,
        merged_img,
        merged_lbl,
    )
    return n_pos, n_neg


def print_split_summary(name: str, stats: SplitBucketStats) -> None:
    total = stats.pos_tiles + stats.neg_tiles_kept
    pos_pct = 100.0 * stats.pos_tiles / max(total, 1)
    neg_pct = 100.0 * stats.neg_tiles_kept / max(total, 1)
    print(
        f"[{name}] sources={stats.source_images} "
        f"defect={stats.pos_tiles} good={stats.neg_tiles_kept} total={total} "
        f"ratio={stats.pos_tiles}:{stats.neg_tiles_kept} "
        f"({pos_pct:.1f}% / {neg_pct:.1f}%)"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split train/val and balance defect vs good tiles")
    p.add_argument("--dataset-root", default="")
    p.add_argument("--out-root", required=True)
    p.add_argument("--src-bucket", default="train",
                   help="Read tiles from images/<bucket> (default train when only one bucket exists)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--pos-neg-ratio", default="3:7", help="Target defect:good ratio, e.g. 3:7")
    p.add_argument("--neg-per-image", type=int, default=0,
                   help="Fixed good tiles per source image; overrides pos-neg-ratio if >0")
    p.add_argument("--neg-per-image-fallback", type=int, default=4,
                   help="Good-only source images (0 defect tile)")
    p.add_argument("--method", choices=["cluster", "random"], default="cluster")
    p.add_argument("--global-balance", action="store_true", default=True,
                   help="Trim negatives per split to match pos-neg-ratio globally")
    p.add_argument("--no-global-balance", dest="global_balance", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-names", default="point,line")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--merge-only", action="store_true",
                   help="Only rebuild merged/ + data.yaml from existing out-root buckets")
    return p.parse_args()


def load_stats_from_manifests(out_root: Path) -> JobStats:
    stats = JobStats()
    for split_name, bucket in (("train", stats.train), ("val", stats.val)):
        manifest = out_root / f"manifest_{split_name}.csv"
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        bucket.source_images = len(rows)
        for row in rows:
            bucket.pos_tiles += int(row["pos_tiles"])
            bucket.neg_tiles_raw += int(row["neg_raw"])
            bucket.neg_tiles_kept += int(row["neg_kept"])
    return stats


def main() -> None:
    args = parse_args()
    args.pos_neg = parse_pos_neg_ratio(args.pos_neg_ratio)
    rng = random.Random(args.seed)

    if not args.merge_only and not args.dataset_root:
        raise ValueError("--dataset-root is required unless --merge-only")

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists() and not args.merge_only:
        safe_rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== split_balance_tile_dataset ===")
    print(f"out={out_root}")

    if args.merge_only:
        stats = load_stats_from_manifests(out_root)
        print("mode=merge-only (skip split/sample)")
    else:
        pos_pack, neg_pack = load_tile_groups(dataset_root, args.src_bucket)
        all_sources = sorted(set(pos_pack["groups"]) | set(neg_pack["groups"]))
        if not all_sources:
            raise FileNotFoundError(f"no tiles under {dataset_root}")

        train_src, val_src = stratified_split_sources(
            all_sources, pos_pack["groups"], args.val_ratio, rng
        )

        print(f"in={dataset_root}")
        print(f"sources={len(all_sources)} train={len(train_src)} val={len(val_src)}")
        print(f"target ratio defect:good = {args.pos_neg_ratio}")

        stats = JobStats()
        process_bucket("train", train_src, pos_pack, neg_pack, out_root, args, rng, stats.train)
        process_bucket("val", val_src, pos_pack, neg_pack, out_root, args, rng, stats.val)

    tr_pos, tr_neg = build_merged(out_root, "train")
    va_pos, va_neg = build_merged(out_root, "val")
    print(f"merged train: defect={tr_pos} good={tr_neg} total={tr_pos+tr_neg}")
    print(f"merged val:   defect={va_pos} good={va_neg} total={va_pos+va_neg}")

    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    snt.write_data_yaml(out_root, class_names)

    summary = out_root / "split_manifest.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "source_images", "defect_tiles", "good_raw", "good_kept", "pos_neg_ratio"])
        for name, s in (("train", stats.train), ("val", stats.val)):
            w.writerow([
                name, s.source_images, s.pos_tiles, s.neg_tiles_raw, s.neg_tiles_kept, args.pos_neg_ratio,
            ])

    print_split_summary("train", stats.train)
    print_split_summary("val", stats.val)
    print(f"data.yaml -> {out_root / 'data.yaml'}")
    print(f"summary   -> {summary}")
    print("\nTrain with:")
    print(f"  python scripts/train_seg_defect.py --data \"{out_root / 'data.yaml'}\" --imgsz 512 --mosaic 0")


if __name__ == "__main__":
    main()
