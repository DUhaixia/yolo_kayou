"""
对已切好的负样本 tile 做下采样：按「每张原图」均匀配额 + 聚类选代表。

为什么用聚类而不是纯随机:
  - 同一张原图的负 tile 纹理高度相似（相邻滑窗重叠多）
  - 随机抽容易抽到几乎一样的 patch，信息冗余
  - KMeans 按纹理特征分簇后，每簇取 1 个代表 → 覆盖灰底/图案/边缘等不同区域

配额模式（二选一）:
  --neg-per-image 4        每张原图固定保留 4 个负 tile（均匀）
  --neg-ratio 3:7          按该图正样本数计算: neg_k = round(pos_k * 7/3)

示例（基于已有 dataSet-tile512）:
  python scripts/sample_neg_tiles.py ^
    --dataset-root "M:/压印 - 副本/dataSet-tile512" ^
    --split train ^
    --neg-per-image 4 ^
    --method cluster ^
    --merge ^
    --out-root "M:/压印 - 副本/dataSet-tile512/sampled"
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
STEM_RE = re.compile(r"^(?P<src>.+)_y(?P<y>\d+)_x(?P<x>\d+)$")


def parse_ratio(text: str) -> float:
    text = text.strip()
    if ":" in text:
        a, b = text.split(":", 1)
        pos, neg = float(a), float(b)
        return neg / (pos + neg)
    val = float(text)
    if not 0.0 <= val < 1.0:
        raise ValueError(f"invalid neg-ratio: {text}")
    return val


def _read_path_candidates(path: Path) -> list[str]:
    candidates = [str(path)]
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            candidates.append("\\\\?\\" + resolved)
    return candidates


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read image with unicode path; tolerate transient network-drive IO errors."""
    for attempt in range(3):
        for pstr in _read_path_candidates(path):
            try:
                data = np.fromfile(pstr, dtype=np.uint8)
            except OSError:
                continue
            if data.size == 0:
                continue
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        if attempt < 2:
            time.sleep(0.05 * (attempt + 1))
    return None


def parse_source_stem(tile_stem: str) -> str | None:
    m = STEM_RE.match(tile_stem)
    return m.group("src") if m else None


def list_tiles(img_dir: Path) -> list[Path]:
    if not img_dir.is_dir():
        return []
    return sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def group_by_source(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        src = parse_source_stem(p.stem)
        if src is None:
            continue
        groups[src].append(p)
    return groups


def extract_feature(img_bgr: np.ndarray) -> np.ndarray:
    """轻量纹理特征: 灰度下采样 + 梯度能量 + 分块均值。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.resize(cv2.magnitude(gx, gy), (16, 16), interpolation=cv2.INTER_AREA)
    grad = grad / (grad.max() + 1e-6)

    # 4x4 block mean on 64x64
    mid = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    blocks = []
    for by in range(4):
        for bx in range(4):
            block = mid[by * 16 : (by + 1) * 16, bx * 16 : (bx + 1) * 16]
            blocks.append(float(block.mean()))
    feat = np.concatenate([small.reshape(-1), grad.reshape(-1), np.array(blocks, np.float32)])
    return feat.astype(np.float32)


def tile_xy(path: Path) -> tuple[float, float] | None:
    m = STEM_RE.match(path.stem)
    if not m:
        return None
    return float(m.group("y")), float(m.group("x"))


def kmeans_labels(data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(data))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        data.astype(np.float32),
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape(-1), centers


def cluster_pick(paths: list[Path], k: int, rng: random.Random) -> list[Path]:
    """两阶段聚类: 先用 tile 坐标保证空间均匀，再在各簇内用纹理选代表。"""
    if len(paths) <= k:
        return paths

    valid: list[Path] = []
    coords: list[list[float]] = []
    for p in paths:
        xy = tile_xy(p)
        if xy is None:
            continue
        valid.append(p)
        coords.append([xy[0], xy[1]])
    if len(valid) <= k:
        return valid

    coord_arr = np.array(coords, dtype=np.float32)
    coord_n = (coord_arr - coord_arr.mean(axis=0)) / (coord_arr.std(axis=0) + 1e-6)
    labels, centers = kmeans_labels(coord_n, k)

    picked: list[Path] = []
    for cid in range(int(labels.max()) + 1):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        cands = [valid[int(i)] for i in idxs]

        # 簇内仅对候选读图提特征，选最接近簇纹理中心的代表
        if len(cands) == 1:
            picked.append(cands[0])
            continue

        feats = []
        ok = []
        for p in cands:
            im = imread_unicode(p)
            if im is None:
                continue
            feats.append(extract_feature(im))
            ok.append(p)
        if not ok:
            picked.append(rng.choice(cands))
            continue
        data = np.stack(feats, axis=0)
        data_n = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-6)
        sub_labels, sub_centers = kmeans_labels(data_n, 1)
        dist = np.linalg.norm(data_n - sub_centers[0], axis=1)
        picked.append(ok[int(np.argmin(dist))])

    if len(picked) < k:
        remain = [p for p in valid if p not in picked]
        rng.shuffle(remain)
        picked.extend(remain[: k - len(picked)])
    return picked[:k]


def random_pick(paths: list[Path], k: int, rng: random.Random) -> list[Path]:
    if len(paths) <= k:
        return paths
    return rng.sample(paths, k)


def target_k_for_image(pos_count: int, args: argparse.Namespace) -> int:
    if args.neg_per_image > 0:
        return args.neg_per_image
    if pos_count <= 0:
        return args.neg_per_image_fallback
    return max(1, int(round(pos_count * args.neg_ratio / max(1e-6, 1.0 - args.neg_ratio))))


def copy_tile(img_path: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dst_img_dir / img_path.name)
    lbl = src_lbl_dir / f"{img_path.stem}.txt"
    out_lbl = dst_lbl_dir / f"{img_path.stem}.txt"
    if lbl.exists():
        shutil.copy2(lbl, out_lbl)
    else:
        out_lbl.write_text("", encoding="utf-8")


def copy_all_tiles(img_dir: Path, lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> int:
    n = 0
    for img in list_tiles(img_dir):
        copy_tile(img, lbl_dir, dst_img_dir, dst_lbl_dir)
        n += 1
    return n


def write_data_yaml(out_root: Path, class_names: list[str]) -> None:
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    text = "\n".join(
        [
            f"path: {out_root.resolve().as_posix()}",
            "train: merged/images/train",
            "val: merged/images/val",
            "names:",
            names_block,
            "",
        ]
    )
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Per-image clustered downsampling for negative tiles")
    p.add_argument("--dataset-root", required=True, help="Existing tile dataset root")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--out-root", default="", help="Output root; default <dataset-root>/sampled")
    p.add_argument("--method", choices=["cluster", "random"], default="cluster")
    p.add_argument("--neg-per-image", type=int, default=0, help="Fixed negative quota per source image")
    p.add_argument("--neg-ratio", default="", help="Alternative: e.g. 3:7 based on pos tiles per image")
    p.add_argument("--neg-per-image-fallback", type=int, default=4,
                   help="When neg-ratio mode and image has 0 pos tile")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge", action="store_true", help="Build merged/images+labels with all pos + sampled neg")
    p.add_argument("--class-names", default="point,line")
    args = p.parse_args()

    if args.neg_per_image <= 0 and not args.neg_ratio:
        raise ValueError("set --neg-per-image or --neg-ratio")
    if args.neg_per_image > 0 and args.neg_ratio:
        print("[warn] both set; using --neg-per-image")

    args.neg_ratio = parse_ratio(args.neg_ratio) if args.neg_ratio else 0.0
    rng = random.Random(args.seed)

    root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else (root / "sampled")
    split = args.split

    pos_img_dir = root / "images" / split
    pos_lbl_dir = root / "labels" / split
    neg_img_dir = root / "images_neg" / split
    neg_lbl_dir = root / "labels_neg" / split

    pos_groups = group_by_source(list_tiles(pos_img_dir))
    neg_groups = group_by_source(list_tiles(neg_img_dir))

    all_sources = sorted(set(pos_groups) | set(neg_groups))
    pick_fn = cluster_pick if args.method == "cluster" else random_pick

    out_neg_img = out_root / "images_neg" / split
    out_neg_lbl = out_root / "labels_neg" / split
    out_neg_img.mkdir(parents=True, exist_ok=True)
    out_neg_lbl.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_raw = 0
    total_kept = 0

    for i, src in enumerate(all_sources, start=1):
        neg_paths = neg_groups.get(src, [])
        pos_count = len(pos_groups.get(src, []))
        total_raw += len(neg_paths)
        k = target_k_for_image(pos_count, args)
        k = min(k, len(neg_paths)) if neg_paths else 0
        picked = pick_fn(neg_paths, k, rng) if k > 0 else []
        for img_path in picked:
            copy_tile(img_path, neg_lbl_dir, out_neg_img, out_neg_lbl)
        total_kept += len(picked)
        manifest_rows.append([src, pos_count, len(neg_paths), k, len(picked)])
        if i % 300 == 0 or i == len(all_sources):
            print(f"  progress {i}/{len(all_sources)} kept_neg={total_kept}")

    print(f"=== sample_neg_tiles ({args.method}) ===")
    print(f"source images: {len(all_sources)}")
    print(f"neg raw={total_raw} kept={total_kept} ratio={total_kept/max(total_raw,1):.3f}")

    manifest = out_root / f"neg_sample_manifest_{split}.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_image", "pos_tiles", "neg_tiles_raw", "target_k", "neg_tiles_kept"])
        w.writerows(manifest_rows)
    print(f"manifest -> {manifest}")

    if args.merge:
        merged_img = out_root / "merged" / "images" / split
        merged_lbl = out_root / "merged" / "labels" / split
        if merged_img.exists():
            shutil.rmtree(merged_img)
        if merged_lbl.exists():
            shutil.rmtree(merged_lbl)
        n_pos = copy_all_tiles(pos_img_dir, pos_lbl_dir, merged_img, merged_lbl)
        n_neg = copy_all_tiles(out_neg_img, out_neg_lbl, merged_img, merged_lbl)
        print(f"merged -> {merged_img}  pos={n_pos} neg={n_neg} total={n_pos+n_neg}")
        class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
        write_data_yaml(out_root, class_names)
        print(f"data.yaml -> {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
