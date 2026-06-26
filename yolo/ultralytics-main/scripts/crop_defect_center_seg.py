"""
YOLO-Seg 标注：按每个缺陷实例的中心裁 512 patch，供本地筛选后训练。

每张原图有 N 个缺陷 → 生成 N 个 patch（每个 patch 以该缺陷中心对齐）。
patch 内会保留所有落在窗口里的其他标注（同图多缺陷时可能同时出现）。

输出:
  images/   裁切图
  labels/   对应 YOLO-Seg 标签
  manifest.csv  筛选清单（可用 Excel 打开）
  preview/  叠加标注预览（可选，方便肉眼看）

示例:
  python scripts/crop_defect_center_seg.py ^
    --src-images D:/dataset/images/train ^
    --src-labels D:/dataset/labels/train ^
    --dst-root D:/dataset_center_crops/train ^
    --patch 512

  # 指定线缺陷 class id（用于标记线是否被裁短）
  python scripts/crop_defect_center_seg.py ^
    --src-images D:/dataset/images/train ^
    --src-labels D:/dataset/labels/train ^
    --dst-root D:/dataset_center_crops/train ^
    --line-classes 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# 预览颜色 BGR: point=绿, line=红, other=黄
PREVIEW_COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in IMAGE_SUFFIXES else ".png"
    path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def parse_seg_line(line: str) -> tuple[int, list[float]]:
    parts = line.strip().split()
    if len(parts) < 7:
        raise ValueError(f"Not YOLO-Seg line (need >=7 values): {line}")
    cls_id = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    if len(coords) % 2 != 0:
        raise ValueError(f"Odd number of polygon coords: {line}")
    return cls_id, coords


def polygon_to_pixels(coords: list[float], w: int, h: int) -> np.ndarray:
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


def pixels_to_normalized(pts: np.ndarray, w: int, h: int) -> list[float]:
    out = pts.copy()
    out[:, 0] /= max(w, 1)
    out[:, 1] /= max(h, 1)
    return np.clip(out, 0.0, 1.0).reshape(-1).tolist()


def defect_center(pts: np.ndarray) -> tuple[float, float]:
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def polygon_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(pts.astype(np.float32)))


def clip_polygon_to_rect(pts: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
    def _clip(poly, edge):
        if len(poly) == 0:
            return poly
        res = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            c_in, p_in = edge(cur), edge(prev)
            if c_in:
                if not p_in:
                    res.append(_intersect(prev, cur, edge, x0, y0, x1, y1))
                res.append(cur)
            elif p_in:
                res.append(_intersect(prev, cur, edge, x0, y0, x1, y1))
        return np.array(res, dtype=np.float64) if res else np.empty((0, 2))

    def left(p):
        return p[0] >= x0

    def right(p):
        return p[0] <= x1

    def top(p):
        return p[1] >= y0

    def bottom(p):
        return p[1] <= y1

    poly = pts.astype(np.float64)
    for edge in (left, right, top, bottom):
        poly = _clip(poly, edge)
        if len(poly) < 3:
            return None
    return poly


def _intersect(s, e, edge, x0, y0, x1, y1):
    if edge.__name__ == "left":
        x = x0
        t = (x - s[0]) / (e[0] - s[0]) if abs(e[0] - s[0]) > 1e-8 else 0.0
        return np.array([x, s[1] + t * (e[1] - s[1])])
    if edge.__name__ == "right":
        x = x1
        t = (x - s[0]) / (e[0] - s[0]) if abs(e[0] - s[0]) > 1e-8 else 0.0
        return np.array([x, s[1] + t * (e[1] - s[1])])
    if edge.__name__ == "top":
        y = y0
        t = (y - s[1]) / (e[1] - s[1]) if abs(e[1] - s[1]) > 1e-8 else 0.0
        return np.array([s[0] + t * (e[0] - s[0]), y])
    y = y1
    t = (y - s[1]) / (e[1] - s[1]) if abs(e[1] - s[1]) > 1e-8 else 0.0
    return np.array([s[0] + t * (e[0] - s[0]), y])


def crop_origin(cx: float, cy: float, patch: int, img_w: int, img_h: int) -> tuple[int, int]:
    x0 = int(round(cx - patch / 2))
    y0 = int(round(cy - patch / 2))
    x0 = max(0, min(x0, max(img_w - patch, 0)))
    y0 = max(0, min(y0, max(img_h - patch, 0)))
    return y0, x0


def transform_seg_label(
    cls_id: int,
    coords: list[float],
    img_w: int,
    img_h: int,
    x0: int,
    y0: int,
    patch: int,
    line_class_ids: set[int],
    min_line_len: float,
    min_area: float,
) -> tuple[str | None, dict]:
    pts = polygon_to_pixels(coords, img_w, img_h)
    full_len = polyline_length(pts)
    full_area = polygon_area(pts)

    bx0, by0 = pts.min(axis=0)
    bx1, by1 = pts.max(axis=0)
    if bx1 <= x0 or bx0 >= x0 + patch or by1 <= y0 or by0 >= y0 + patch:
        return None, {}

    clipped = clip_polygon_to_rect(pts, x0, y0, x0 + patch, y0 + patch)
    if clipped is None:
        return None, {}

    local = clipped - np.array([x0, y0], dtype=np.float64)
    seg_len = polyline_length(local)
    seg_area = polygon_area(local)
    is_line = cls_id in line_class_ids

    if is_line:
        if seg_len < min_line_len:
            return None, {}
    elif seg_area < min_area:
        return None, {}

    norm = pixels_to_normalized(local, patch, patch)
    text = f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm)
    meta = {
        "seg_len": round(seg_len, 2),
        "seg_area": round(seg_area, 2),
        "truncated": is_line and seg_len < full_len * 0.95,
        "full_len": round(full_len, 2),
        "full_area": round(full_area, 2),
    }
    return text, meta


def draw_preview(patch: np.ndarray, label_lines: list[str], patch_size: int) -> np.ndarray:
    vis = patch.copy()
    for line in label_lines:
        cls_id = int(line.split()[0])
        coords = [float(x) for x in line.split()[1:]]
        pts = polygon_to_pixels(coords, patch_size, patch_size).astype(np.int32)
        color = PREVIEW_COLORS[cls_id % len(PREVIEW_COLORS)]
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], isClosed=len(pts) >= 3, color=color, thickness=2)
        if len(pts) >= 1:
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.circle(vis, (cx, cy), 4, color, -1)
    return vis


def load_seg_labels(label_path: Path) -> list[tuple[int, list[float]]]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(parse_seg_line(line))
    return labels


def process(args) -> None:
    src_img_dir = Path(args.src_images)
    src_lbl_dir = Path(args.src_labels)
    dst_root = Path(args.dst_root) if args.dst_root else Path(".")
    dst_img_dir = Path(args.dst_images) if args.dst_images else dst_root / "images"
    dst_lbl_dir = Path(args.dst_labels) if args.dst_labels else dst_root / "labels"
    dst_preview_dir = Path(args.dst_preview) if args.dst_preview else dst_root / "preview"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        dst_preview_dir.mkdir(parents=True, exist_ok=True)

    line_ids = {int(x) for x in args.line_classes.split(",") if x.strip()}
    images = sorted(p for p in src_img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    manifest_rows = []
    n_patches = 0

    for img_path in images:
        im = imread_unicode(img_path)
        if im is None:
            print(f"[skip] cannot read {img_path}")
            continue

        h, w = im.shape[:2]
        if w < args.patch or h < args.patch:
            print(f"[skip] image smaller than patch: {img_path.name} ({w}x{h})")
            continue

        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        labels = load_seg_labels(label_path)
        if not labels:
            print(f"[skip] no seg labels: {img_path.name}")
            continue

        for inst_idx, (primary_cls, primary_coords) in enumerate(labels):
            pts = polygon_to_pixels(primary_coords, w, h)
            cx, cy = defect_center(pts)
            y0, x0 = crop_origin(cx, cy, args.patch, w, h)

            patch = im[y0:y0 + args.patch, x0:x0 + args.patch]
            out_labels = []
            label_meta = []

            for cls_id, coords in labels:
                text, meta = transform_seg_label(
                    cls_id, coords, w, h, x0, y0, args.patch,
                    line_ids, args.min_line_len, args.min_area,
                )
                if text:
                    out_labels.append(text)
                    label_meta.append((cls_id, meta))

            if not out_labels:
                continue

            stem = f"{img_path.stem}_c{primary_cls}_i{inst_idx:03d}_y{y0}_x{x0}"
            imwrite_unicode(dst_img_dir / f"{stem}.png", patch)
            (dst_lbl_dir / f"{stem}.txt").write_text("\n".join(out_labels) + "\n", encoding="utf-8")

            if args.preview:
                preview = draw_preview(patch, out_labels, args.patch)
                imwrite_unicode(dst_preview_dir / f"{stem}.png", preview)

            primary_meta = next((m for c, m in label_meta if c == primary_cls), label_meta[0][1])
            manifest_rows.append({
                "filename": f"{stem}.png",
                "source_image": img_path.name,
                "primary_class": primary_cls,
                "instance_index": inst_idx,
                "center_x": round(cx, 1),
                "center_y": round(cy, 1),
                "crop_x": x0,
                "crop_y": y0,
                "num_labels": len(out_labels),
                "primary_area_px": primary_meta.get("seg_area", primary_meta.get("full_area", 0)),
                "primary_len_px": primary_meta.get("seg_len", primary_meta.get("full_len", 0)),
                "line_truncated": int(primary_meta.get("truncated", False)),
                "keep": "",  # 本地筛选时填 1=保留 0=删除
                "note": "",
            })
            n_patches += 1

    manifest_path = Path(args.manifest) if args.manifest else dst_root / "manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"source images : {len(images)}")
    print(f"output patches: {n_patches}")
    print(f"images        -> {dst_img_dir}")
    print(f"labels        -> {dst_lbl_dir}")
    if args.preview:
        print(f"preview       -> {dst_preview_dir}")
    print(f"manifest      -> {manifest_path}")
    print("本地筛选: 在 manifest.csv 的 keep 列填 1(保留)/0(删除)，再用 filter_manifest.py 过滤")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLO-Seg defect-center crop for local review")
    p.add_argument("--src-images", required=True)
    p.add_argument("--src-labels", required=True)
    p.add_argument("--dst-root", default="", help="output root (optional if --dst-images/--dst-labels set)")
    p.add_argument("--dst-images", default="", help="direct output image dir")
    p.add_argument("--dst-labels", default="", help="direct output label dir")
    p.add_argument("--dst-preview", default="", help="direct preview dir")
    p.add_argument("--manifest", default="", help="manifest csv path")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--line-classes", default="1", help="line class ids, comma separated")
    p.add_argument("--min-line-len", type=float, default=6.0)
    p.add_argument("--min-area", type=float, default=4.0)
    p.add_argument("--preview", action="store_true", help="save preview/ with drawn polygons")
    process(p.parse_args())
