"""
将 1280 等大图 + YOLO 标注切成 512 训练 patch，并同步变换标签。

支持 YOLO 检测 (cls cx cy w h) 与分割 (cls x1 y1 x2 y2 ...)。

线缺陷被裁切时的策略（--line-policy）:
  keep   : 保留 patch 内的线段部分（推荐，配合 stride 重叠）
  drop   : 线被裁断就丢弃该 patch 中的该线
  center : 以线为中心生成窗口，尽量不裁断

示例:
  python scripts/slide_crop_dataset.py ^
    --src-images D:/dataset/images/train ^
    --src-labels D:/dataset/labels/train ^
    --dst-images D:/dataset_crops/images/train ^
    --dst-labels D:/dataset_crops/labels/train ^
    --patch 512 --stride 256 --line-policy keep

  # 固定滑窗 + 负样本单独目录（空 label）
  python scripts/slide_crop_dataset.py ^
    --src-images D:/dataset/images/train ^
    --src-labels D:/dataset/labels/train ^
    --dst-images D:/dataset_crops/images/train ^
    --dst-labels D:/dataset_crops/labels/train ^
    --neg-images D:/dataset_crops/images_neg/train ^
    --neg-labels D:/dataset_crops/labels_neg/train ^
    --mode sliding --patch 512 --stride 256

  # 缺陷中心裁切（点/线都适用，线不易断）
  python scripts/slide_crop_dataset.py ^
    --src-images D:/dataset/images/train ^
    --src-labels D:/dataset/labels/train ^
    --dst-images D:/dataset_crops/images/train ^
    --dst-labels D:/dataset_crops/labels/train ^
    --mode defect_center --patch 512 --pad 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


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


def parse_label_line(line: str) -> tuple[int, str, list[float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Bad label line: {line}")
    cls_id = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    fmt = "seg" if len(coords) > 4 else "det"
    return cls_id, fmt, coords


def polygon_to_pixels(coords: list[float], w: int, h: int) -> np.ndarray:
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


def pixels_to_normalized(pts: np.ndarray, w: int, h: int) -> list[float]:
    out = pts.copy()
    out[:, 0] /= max(w, 1)
    out[:, 1] /= max(h, 1)
    out = np.clip(out, 0.0, 1.0)
    return out.reshape(-1).tolist()


def clip_polygon_to_rect(pts: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
    """Sutherland-Hodgman polygon clip."""
    def _clip(poly, edge):
        if len(poly) == 0:
            return poly
        res = []
        for i in range(len(poly)):
            cur = poly[i]
            prev = poly[i - 1]
            c_in = edge(cur)
            p_in = edge(prev)
            if c_in:
                if not p_in:
                    res.append(intersect(prev, cur, edge))
                res.append(cur)
            elif p_in:
                res.append(intersect(prev, cur, edge))
        return np.array(res, dtype=np.float64) if res else np.empty((0, 2))

    def left(p):
        return p[0] >= x0

    def right(p):
        return p[0] <= x1

    def top(p):
        return p[1] >= y0

    def bottom(p):
        return p[1] <= y1

    def intersect(s, e, edge):
        # parametric; handle axis-aligned edges only
        if edge in (left, right):
            x = x0 if edge is left else x1
            if abs(e[0] - s[0]) < 1e-8:
                return np.array([x, s[1]])
            t = (x - s[0]) / (e[0] - s[0])
            return np.array([x, s[1] + t * (e[1] - s[1])])
        y = y0 if edge is top else y1
        if abs(e[1] - s[1]) < 1e-8:
            return np.array([s[0], y])
        t = (y - s[1]) / (e[1] - s[1])
        return np.array([s[0] + t * (e[0] - s[0]), y])

    poly = pts.astype(np.float64)
    for edge in (left, right, top, bottom):
        poly = _clip(poly, edge)
        if len(poly) < 3:
            return None
    return poly


def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def bbox_from_points(pts: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def transform_label(cls_id: int, fmt: str, coords: list[float],
                    img_w: int, img_h: int, x0: int, y0: int, pw: int, ph: int,
                    line_policy: str, min_line_len: float, min_area: float,
                    line_class_ids: set[int]) -> list[float] | None:
    if fmt == "det":
        cx, cy, bw, bh = coords
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_bw = bw * img_w
        abs_bh = bh * img_h
        bx0 = abs_cx - abs_bw / 2
        by0 = abs_cy - abs_bh / 2
        bx1 = abs_cx + abs_bw / 2
        by1 = abs_cy + abs_bh / 2
        ix0 = max(bx0, x0)
        iy0 = max(by0, y0)
        ix1 = min(bx1, x0 + pw)
        iy1 = min(by1, y0 + ph)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        area = (ix1 - ix0) * (iy1 - iy0)
        if area < min_area:
            return None
        ncx = ((ix0 + ix1) / 2 - x0) / pw
        ncy = ((iy0 + iy1) / 2 - y0) / ph
        nbw = (ix1 - ix0) / pw
        nbh = (iy1 - iy0) / ph
        return [cls_id, ncx, ncy, nbw, nbh]

    pts = polygon_to_pixels(coords, img_w, img_h)
    is_line = cls_id in line_class_ids

    # quick reject: bbox not intersect patch
    bx0, by0, bx1, by1 = bbox_from_points(pts)
    if bx1 <= x0 or bx0 >= x0 + pw or by1 <= y0 or by0 >= y0 + ph:
        return None

    clipped = clip_polygon_to_rect(pts, x0, y0, x0 + pw, y0 + ph)
    if clipped is None or len(clipped) < 2:
        return None

    local = clipped - np.array([x0, y0], dtype=np.float64)

    if is_line:
        seg_len = polyline_length(local)
        if line_policy == "drop" and seg_len < polyline_length(pts) * 0.95:
            return None
        if seg_len < min_line_len:
            return None
    else:
        # point-like polygon area
        if len(local) >= 3:
            area = cv2.contourArea(local.astype(np.float32))
            if area < min_area:
                return None

    if len(local) == 2 or (is_line and len(local) >= 2):
        # keep as 2-point or open polyline; YOLO seg accepts >=3 usually,
        # duplicate a tiny width for 2-point lines
        if len(local) == 2:
            p0, p1 = local
            n = np.array([p0, p1, p1 + [1.0, 0.0], p0 + [1.0, 0.0]], dtype=np.float64)
            local = n
    norm = pixels_to_normalized(local, pw, ph)
    return [cls_id] + norm


def sliding_coords(img_h: int, img_w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(img_h - patch, 0) + 1, stride))
    xs = list(range(0, max(img_w - patch, 0) + 1, stride))
    if not ys or ys[-1] + patch < img_h:
        ys.append(max(img_h - patch, 0))
    if not xs or xs[-1] + patch < img_w:
        xs.append(max(img_w - patch, 0))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    return [(y, x) for y in ys for x in xs]


def defect_center_coords(labels: list[tuple], img_w: int, img_h: int,
                         patch: int, pad: int) -> list[tuple[int, int]]:
    coords = []
    for cls_id, fmt, pts_norm in labels:
        if fmt == "det":
            cx, cy = pts_norm[0] * img_w, pts_norm[1] * img_h
        else:
            pts = polygon_to_pixels(pts_norm, img_w, img_h)
            cx, cy = pts.mean(axis=0)
        x0 = int(round(cx - patch / 2))
        y0 = int(round(cy - patch / 2))
        x0 = max(0, min(x0, img_w - patch))
        y0 = max(0, min(y0, img_h - patch))
        coords.append((y0, x0))
        # small jitter windows around defect
        for dy in (-pad, 0, pad):
            for dx in (-pad, 0, pad):
                yy = max(0, min(y0 + dy, img_h - patch))
                xx = max(0, min(x0 + dx, img_w - patch))
                coords.append((yy, xx))
    return sorted(set(coords))


def load_labels(label_path: Path) -> list[tuple[int, str, list[float]]]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(parse_label_line(line))
    return labels


def format_label(row: list[float]) -> str:
    cls_id = int(row[0])
    coords = row[1:]
    if len(coords) == 4:
        return f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)


def process_split(args) -> None:
    src_img_dir = Path(args.src_images)
    src_lbl_dir = Path(args.src_labels)
    dst_img_dir = Path(args.dst_images)
    dst_lbl_dir = Path(args.dst_labels)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    split_neg = bool(args.neg_images and args.neg_labels)
    if bool(args.neg_images) ^ bool(args.neg_labels):
        raise ValueError("--neg-images and --neg-labels must be set together")
    neg_img_dir = Path(args.neg_images) if split_neg else None
    neg_lbl_dir = Path(args.neg_labels) if split_neg else None
    if split_neg:
        neg_img_dir.mkdir(parents=True, exist_ok=True)
        neg_lbl_dir.mkdir(parents=True, exist_ok=True)

    line_ids = {int(x) for x in args.line_classes.split(",") if x.strip() != ""}
    images = sorted(p for p in src_img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    total_pos = 0
    total_neg = 0
    total_labels = 0

    for img_path in images:
        im = imread_unicode(img_path)
        if im is None:
            print(f"[skip] cannot read {img_path}")
            continue
        h, w = im.shape[:2]
        label_path = src_lbl_dir / (img_path.stem + ".txt")
        labels = load_labels(label_path)

        if args.mode == "sliding":
            windows = sliding_coords(h, w, args.patch, args.stride)
        elif args.mode == "defect_center":
            if not labels:
                continue
            windows = defect_center_coords(labels, w, h, args.patch, args.pad)
        else:
            windows = sliding_coords(h, w, args.patch, args.stride)
            if labels:
                windows = sorted(set(windows + defect_center_coords(labels, w, h, args.patch, args.pad)))

        for yi, xi in windows:
            patch = im[yi:yi + args.patch, xi:xi + args.patch]
            out_labels = []
            for cls_id, fmt, coords in labels:
                row = transform_label(
                    cls_id, fmt, coords, w, h, xi, yi, args.patch, args.patch,
                    args.line_policy, args.min_line_len, args.min_area, line_ids,
                )
                if row is not None:
                    out_labels.append(format_label(row))

            is_negative = not out_labels
            if is_negative:
                if split_neg:
                    out_img_dir, out_lbl_dir = neg_img_dir, neg_lbl_dir
                elif args.keep_empty:
                    out_img_dir, out_lbl_dir = dst_img_dir, dst_lbl_dir
                else:
                    continue
            else:
                out_img_dir, out_lbl_dir = dst_img_dir, dst_lbl_dir

            stem = f"{img_path.stem}_y{yi}_x{xi}"
            imwrite_unicode(out_img_dir / f"{stem}.png", patch)
            (out_lbl_dir / f"{stem}.txt").write_text(
                "\n".join(out_labels) + ("\n" if out_labels else ""),
                encoding="utf-8",
            )
            if is_negative:
                total_neg += 1
            else:
                total_pos += 1
                total_labels += len(out_labels)

    print(f"done: source_images={len(images)}, pos_patches={total_pos}, neg_patches={total_neg}, labels={total_labels}")
    print(f"positive images -> {dst_img_dir}")
    print(f"positive labels -> {dst_lbl_dir}")
    if split_neg:
        print(f"negative images -> {neg_img_dir}")
        print(f"negative labels -> {neg_lbl_dir}")


def main():
    p = argparse.ArgumentParser(description="Slide-crop YOLO dataset with label transform")
    p.add_argument("--src-images", required=True)
    p.add_argument("--src-labels", required=True)
    p.add_argument("--dst-images", required=True)
    p.add_argument("--dst-labels", required=True)
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--mode", choices=["sliding", "defect_center", "mixed"], default="mixed")
    p.add_argument("--pad", type=int, default=64, help="jitter for defect_center mode")
    p.add_argument("--line-policy", choices=["keep", "drop"], default="keep")
    p.add_argument("--line-classes", default="1", help="comma-separated class ids for line, e.g. 1 or 1,3")
    p.add_argument("--min-line-len", type=float, default=6.0, help="min pixel length of line segment in patch")
    p.add_argument("--min-area", type=float, default=4.0, help="min pixel area for point/det box in patch")
    p.add_argument("--keep-empty", action="store_true",
                   help="save background-only patches into dst-* (mixed with positives)")
    p.add_argument("--neg-images", default="", help="separate output dir for empty-label negative patches")
    p.add_argument("--neg-labels", default="", help="separate output dir for empty negative label files")
    args = p.parse_args()
    process_split(args)


if __name__ == "__main__":
    main()
