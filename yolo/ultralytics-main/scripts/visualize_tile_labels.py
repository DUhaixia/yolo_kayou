"""
随机抽取滑窗切图 tile，叠加 YOLO 分割标注可视化，用于检查切图后标签是否正确。

支持:
  - merged/ 或 images+images_neg 目录
  - 正样本(有标注) / 负样本(空 label) 分层抽样
  - 输出单张预览 + 拼图总览

示例:
  python scripts/visualize_tile_labels.py ^
    --dataset-root "M:/压印 - 副本/dataSet-tile512_balanced_11" ^
    --bucket merged --split train ^
    --num 40 --pos 20 --neg 20 ^
    --out "M:/压印 - 副本/dataSet-tile512_balanced_11/vis_check"
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
STEM_RE = re.compile(r"^(?P<src>.+)_y(?P<y>\d+)_x(?P<x>\d+)$")
CLASS_NAMES = {0: "point", 1: "line"}
CLASS_COLORS = {
    0: (255, 80, 60),   # point - blue-ish BGR
    1: (60, 60, 255),   # line - red BGR
}
NEG_COLOR = (80, 200, 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Randomly visualize tile seg labels after sliding-window crop")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--bucket", choices=["merged", "split"], default="merged",
                   help="merged=merged/images+labels; split=images+images_neg")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--num", type=int, default=40, help="Total samples")
    p.add_argument("--pos", type=int, default=0, help="Defect tiles to sample (0=auto half)")
    p.add_argument("--neg", type=int, default=0, help="Good tiles to sample (0=auto half)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--grid-cols", type=int, default=5)
    p.add_argument("--thumb", type=int, default=384, help="Thumbnail size in grid overview")
    return p.parse_args()


def _read_path_candidates(path: Path) -> list[str]:
    candidates = [str(path)]
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            candidates.append("\\\\?\\" + resolved)
    return candidates


def imread_unicode(path: Path) -> np.ndarray | None:
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


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def list_images(img_dir: Path) -> list[Path]:
    if not img_dir.is_dir():
        return []
    return sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def resolve_dirs(root: Path, bucket: str, split: str) -> tuple[Path, Path]:
    if bucket == "merged":
        return root / "merged" / "images" / split, root / "merged" / "labels" / split
    pos_img = root / "images" / split
    pos_lbl = root / "labels" / split
    neg_img = root / "images_neg" / split
    neg_lbl = root / "labels_neg" / split
    return pos_img, pos_lbl, neg_img, neg_lbl  # type: ignore[return-value]


def parse_seg_lines(label_path: Path) -> list[tuple[int, np.ndarray]]:
    if not label_path.is_file():
        return []
    rows: list[tuple[int, np.ndarray]] = []
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        cls_id = int(float(parts[0]))
        pts = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        rows.append((cls_id, pts))
    return rows


def norm_to_pixels(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    out = pts.copy()
    out[:, 0] *= w
    out[:, 1] *= h
    return out.astype(np.int32)


def parse_tile_meta(stem: str) -> tuple[str, int, int]:
    m = STEM_RE.match(stem)
    if not m:
        return stem, -1, -1
    return m.group("src"), int(m.group("y")), int(m.group("x"))


def draw_tile_labels(
    image: np.ndarray,
    labels: list[tuple[int, np.ndarray]],
    title: str,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]

    overlay = vis.copy()
    for cls_id, pts_norm in labels:
        pts = norm_to_pixels(pts_norm, w, h)
        color = CLASS_COLORS.get(cls_id, (0, 220, 0))
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        elif len(pts) >= 2:
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2)
        if len(pts) >= 1:
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.circle(vis, (cx, cy), 4, color, -1)
            name = CLASS_NAMES.get(cls_id, str(cls_id))
            cv2.putText(vis, name, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if labels:
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

    bar_h = 28
    cv2.rectangle(vis, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(vis, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def draw_neg_tile(image: np.ndarray, title: str) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (4, 4), (w - 4, h - 4), NEG_COLOR, 2)
    bar_h = 28
    cv2.rectangle(vis, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(vis, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(vis, "NEG", (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEG_COLOR, 2, cv2.LINE_AA)
    return vis


def collect_candidates(img_dir: Path, lbl_dir: Path) -> tuple[list[Path], list[Path]]:
    pos_imgs: list[Path] = []
    neg_imgs: list[Path] = []
    for img_path in list_images(img_dir):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        labels = parse_seg_lines(lbl_path)
        if labels:
            pos_imgs.append(img_path)
        else:
            neg_imgs.append(img_path)
    return pos_imgs, neg_imgs


def sample_paths(items: list[Path], k: int, rng: random.Random) -> list[Path]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return items[:]
    return rng.sample(items, k)


def make_grid(images: list[np.ndarray], cols: int, thumb: int) -> np.ndarray:
    if not images:
        return np.zeros((thumb, thumb, 3), dtype=np.uint8)
    thumbs = []
    for img in images:
        t = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        thumbs.append(t)
    rows = (len(thumbs) + cols - 1) // cols
    while len(thumbs) < rows * cols:
        thumbs.append(np.zeros((thumb, thumb, 3), dtype=np.uint8))
    row_imgs = []
    for r in range(rows):
        row = thumbs[r * cols : (r + 1) * cols]
        row_imgs.append(np.hstack(row))
    return np.vstack(row_imgs)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    root = args.dataset_root.resolve()
    out_dir = args.out.resolve() if args.out else root / "vis_tile_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bucket == "merged":
        img_dir, lbl_dir = resolve_dirs(root, "merged", args.split)  # type: ignore[misc]
        pos_lbl_dir = neg_lbl_dir = lbl_dir
        pos_imgs, neg_imgs = collect_candidates(img_dir, lbl_dir)
    else:
        pos_img_dir, pos_lbl_dir, neg_img_dir, neg_lbl_dir = resolve_dirs(root, "split", args.split)  # type: ignore[misc]
        pos_imgs, _ = collect_candidates(pos_img_dir, pos_lbl_dir)
        _, neg_imgs = collect_candidates(neg_img_dir, neg_lbl_dir)

    n_pos = args.pos if args.pos > 0 else max(0, args.num // 2)
    n_neg = args.neg if args.neg > 0 else max(0, args.num - n_pos)
    picked_pos = sample_paths(pos_imgs, n_pos, rng)
    picked_neg = sample_paths(neg_imgs, n_neg, rng)

    print("=== visualize_tile_labels ===")
    print(f"dataset={root}")
    print(f"bucket={args.bucket} split={args.split}")
    print(f"candidates: defect={len(pos_imgs)} good={len(neg_imgs)}")
    print(f"sample: defect={len(picked_pos)} good={len(picked_neg)}")
    print(f"out={out_dir}")

    grid_imgs: list[np.ndarray] = []
    manifest_rows: list[str] = []

    for tag, paths, lbl_root in (
        ("pos", picked_pos, pos_lbl_dir),
        ("neg", picked_neg, neg_lbl_dir),
    ):
        for img_path in paths:
            im = imread_unicode(img_path)
            if im is None:
                print(f"[skip] cannot read {img_path.name}")
                continue
            src, y, x = parse_tile_meta(img_path.stem)
            coord = f"y{y}_x{x}" if y >= 0 else "?"
            labels = parse_seg_lines(lbl_root / f"{img_path.stem}.txt")
            if labels:
                cls_summary = ",".join(CLASS_NAMES.get(c, str(c)) for c, _ in labels)
                title = f"{tag.upper()} | {src[:28]} | {coord} | {cls_summary}"
                vis = draw_tile_labels(im, labels, title)
            else:
                title = f"NEG | {src[:28]} | {coord}"
                vis = draw_neg_tile(im, title)

            out_name = f"{tag}_{img_path.stem}.jpg"
            out_path = out_dir / out_name
            imwrite_unicode(out_path, vis)
            grid_imgs.append(vis)
            manifest_rows.append(f"{out_name}\t{tag}\t{src}\t{y}\t{x}\t{len(labels)}")

    if grid_imgs:
        grid = make_grid(grid_imgs, args.grid_cols, args.thumb)
        grid_path = out_dir / f"overview_{args.split}.jpg"
        imwrite_unicode(grid_path, grid)
        print(f"overview -> {grid_path}")

    (out_dir / f"manifest_{args.split}.tsv").write_text(
        "file\ttag\tsource\ty\tx\tnum_labels\n" + "\n".join(manifest_rows),
        encoding="utf-8",
    )
    print(f"saved {len(grid_imgs)} previews to {out_dir}")


if __name__ == "__main__":
    main()
