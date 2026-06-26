"""
从「纯好品」原图裁 512 patch，生成无标注负样本（空 label），用于降过检微调。

重点覆盖三类易过检区域：
  1) 卡片外灰底
  2) 卡片四周边界带（压印框线/阴影）
  3) 卡片内部平坦背景（无缺陷纹理区）

输出 YOLO 格式:
  images/train/*.png
  labels/train/*.txt   (空文件)

示例:
  python scripts/crop_good_negative_patches.py ^
    --source "G:/卡游/压印testall2/好品" ^
    --out "M:/压印 - 副本/neg_edge_bg" ^
    --patch 512 --stride 256 ^
    --edge-band 80 --max-per-image 24

  # 与缺陷集合并后微调
  # 把 neg 的 images/train 复制到训练集 images/train
  # 把 neg 的 labels/train 空 txt 一并复制
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def safe_stem(name: str, max_len: int = 120) -> str:
    """Windows 文件名安全化（去掉括号/空格等易出问题字符）。"""
    s = re.sub(r'[<>:"/\\|?*]', "_", name)
    s = re.sub(r"\s+", "_", s.strip())
    s = s.strip("._")
    if not s:
        s = "neg"
    return s[:max_len]


def write_empty_label(path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def border_gray_ref(gray: np.ndarray) -> float:
    h, w = gray.shape
    b = max(5, min(h, w) // 80)
    strips = np.concatenate([
        gray[:b, :].ravel(), gray[-b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, -b:].ravel(),
    ])
    return float(np.median(strips))


def build_outer_background_mask(gray: np.ndarray, tol: int = 8) -> np.ndarray:
    from collections import deque

    h, w = gray.shape
    bg_ref = int(round(border_gray_ref(gray)))
    bg_like = np.abs(gray.astype(np.int16) - bg_ref) <= tol
    outer = np.zeros((h, w), dtype=bool)
    step = max(4, min(h, w) // 40)
    q: deque[tuple[int, int]] = deque()
    for x in range(0, w, step):
        for y in (0, h - 1):
            if bg_like[y, x]:
                outer[y, x] = True
                q.append((y, x))
    for y in range(0, h, step):
        for x in (0, w - 1):
            if bg_like[y, x] and not outer[y, x]:
                outer[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and bg_like[ny, nx] and not outer[ny, nx]:
                outer[ny, nx] = True
                q.append((ny, nx))
    return outer


def card_bbox(outer_bg: np.ndarray) -> tuple[int, int, int, int] | None:
    fg = (~outer_bg).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw * bh < 0.05 * outer_bg.size:
        return None
    return x, y, x + bw, y + bh


def sliding_coords(h: int, w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if not ys or ys[-1] + patch < h:
        ys.append(max(h - patch, 0))
    if not xs or xs[-1] + patch < w:
        xs.append(max(w - patch, 0))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def patch_tag(
    gray: np.ndarray,
    outer_bg: np.ndarray,
    card: tuple[int, int, int, int] | None,
    y0: int,
    x0: int,
    patch: int,
    edge_band: int,
    flat_std_thr: float,
) -> str:
    """给 patch 打标签：outer_bg / card_edge / card_flat / card_texture"""
    h, w = gray.shape
    y1, x1 = min(h, y0 + patch), min(w, x0 + patch)
    roi_outer = outer_bg[y0:y1, x0:x1]
    outer_ratio = float(roi_outer.mean())

    if outer_ratio > 0.55:
        return "outer_bg"

    if card is not None:
        cx1, cy1, cx2, cy2 = card
        # 与卡片外轮廓的距离
        dist = cv2.distanceTransform((~outer_bg).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        band = dist[y0:y1, x0:x1] < edge_band
        if float(band.mean()) > 0.25:
            return "card_edge"

    roi = gray[y0:y1, x0:x1]
    if float(np.std(roi)) < flat_std_thr:
        return "card_flat"
    return "card_texture"


def pick_patches(
    img: np.ndarray,
    patch: int,
    stride: int,
    edge_band: int,
    flat_std_thr: float,
    max_per_image: int,
    quotas: dict[str, int],
) -> list[tuple[int, int, str]]:
    gray = to_gray(img)
    h, w = gray.shape
    outer_bg = build_outer_background_mask(gray)
    card = card_bbox(outer_bg)
    coords = sliding_coords(h, w, patch, stride)

    buckets: dict[str, list[tuple[int, int, str]]] = {
        "outer_bg": [], "card_edge": [], "card_flat": [], "card_texture": [],
    }
    for y0, x0 in coords:
        tag = patch_tag(gray, outer_bg, card, y0, x0, patch, edge_band, flat_std_thr)
        buckets[tag].append((y0, x0, tag))

    picked: list[tuple[int, int, str]] = []
    for tag, limit in quotas.items():
        picked.extend(buckets[tag][:limit])

    if len(picked) < max_per_image:
        rest = [c for c in coords if c not in {(y, x) for y, x, _ in picked}]
        for y0, x0 in rest:
            tag = patch_tag(gray, outer_bg, card, y0, x0, patch, edge_band, flat_std_thr)
            picked.append((y0, x0, tag))
            if len(picked) >= max_per_image:
                break
    return picked[:max_per_image]


def main() -> None:
    p = argparse.ArgumentParser(description="裁好品负样本 patch（空标注）")
    p.add_argument("--source", required=True, help="纯好品原图目录")
    p.add_argument("--out", required=True, help="输出根目录")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--edge-band", type=int, default=80, help="卡片外轮廓内多少 px 算边缘带")
    p.add_argument("--flat-std", type=float, default=9.0, help="低于此视为平坦背景")
    p.add_argument("--max-per-image", type=int, default=24)
    p.add_argument("--quota-outer", type=int, default=6)
    p.add_argument("--quota-edge", type=int, default=8)
    p.add_argument("--quota-flat", type=int, default=6)
    p.add_argument("--quota-texture", type=int, default=4)
    args = p.parse_args()

    source = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    img_dir = out_root / "images" / args.split
    lbl_dir = out_root / "labels" / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if not lbl_dir.is_dir():
        raise RuntimeError(f"无法创建标签目录: {lbl_dir}")

    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"no images in {source}")

    quotas = {
        "outer_bg": args.quota_outer,
        "card_edge": args.quota_edge,
        "card_flat": args.quota_flat,
        "card_texture": args.quota_texture,
    }
    rows = []
    n_saved = 0

    for img_path in images:
        img = imread_unicode(img_path)
        if img is None:
            print(f"[skip] {img_path.name}")
            continue

        picks = pick_patches(
            img, args.patch, args.stride, args.edge_band, args.flat_std,
            args.max_per_image, quotas,
        )
        base = safe_stem(img_path.stem)
        for i, (y0, x0, tag) in enumerate(picks):
            patch = img[y0:y0 + args.patch, x0:x0 + args.patch].copy()
            stem = f"{base}_neg_{tag}_{i:02d}"
            if not imwrite_unicode(img_dir / f"{stem}.png", patch):
                print(f"[warn] image write fail: {stem}.png")
                continue
            write_empty_label(lbl_dir / f"{stem}.txt")
            rows.append({
                "src": img_path.name,
                "patch": f"{stem}.png",
                "tag": tag,
                "y0": y0,
                "x0": x0,
            })
            n_saved += 1
        print(f"[{img_path.name}] saved {len(picks)} negative patches")

    manifest = out_root / f"neg_manifest_{args.split}.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "patch", "tag", "y0", "x0"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"done: images={len(images)}, patches={n_saved}")
    print(f"images -> {img_dir}")
    print(f"labels -> {lbl_dir} (empty)")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
