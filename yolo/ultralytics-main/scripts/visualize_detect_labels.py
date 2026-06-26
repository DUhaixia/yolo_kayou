"""Visualize YOLO detect bboxes; optionally overlay source seg polygons to verify conversion."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
COLORS = {
    0: (255, 80, 60),   # point - blue (BGR)
    1: (60, 60, 255),   # line - red
}
NAMES = {0: "point", 1: "line"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualize detect labels (and optional seg compare)")
    p.add_argument("--data", type=Path, default=Path(r"M:\压印 - 副本\dataSet-原始222"))
    p.add_argument(
        "--seg-src",
        type=Path,
        default=Path(r"M:\压印 - 副本\dataSet-原始"),
        help="Original seg dataset for polygon overlay (empty to skip)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output dir (default: <data>/vis_detect_check)")
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point", type=int, default=25, help="Try to pick this many images with class 0")
    p.add_argument("--line", type=int, default=25, help="Try to pick this many images with class 1")
    return p.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix if path.suffix else ".jpg", img)
    if not ok:
        return False
    enc.tofile(str(path))
    return True


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_SUFFIXES:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def parse_detect_label(txt: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not txt.is_file():
        return rows
    for line in txt.read_text(encoding="utf-8").strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        if len(p) > 5:
            continue  # skip seg lines in detect folder
        cls_id = int(float(p[0]))
        cx, cy, w, h = map(float, p[1:5])
        rows.append((cls_id, cx, cy, w, h))
    return rows


def parse_seg_label(txt: Path) -> list[tuple[int, np.ndarray]]:
    rows = []
    if not txt.is_file():
        return rows
    for line in txt.read_text(encoding="utf-8").strip().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        cls_id = int(float(p[0]))
        pts = np.array(list(map(float, p[1:])), dtype=np.float32).reshape(-1, 2)
        rows.append((cls_id, pts))
    return rows


def xywhn2xyxy(box: tuple, w: int, h: int) -> tuple[int, int, int, int]:
    _, cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2


def draw_detect(img: np.ndarray, boxes: list) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for cls_id, cx, cy, bw, bh in boxes:
        color = COLORS.get(cls_id, (0, 255, 0))
        x1, y1, x2, y2 = xywhn2xyxy((cls_id, cx, cy, bw, bh), w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{NAMES.get(cls_id, cls_id)} {bw * w:.0f}x{bh * h:.0f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def draw_seg(img: np.ndarray, segs: list) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for cls_id, pts in segs:
        color = COLORS.get(cls_id, (0, 255, 0))
        px = (pts * np.array([w, h], dtype=np.float32)).astype(np.int32)
        cv2.polylines(out, [px], True, color, 2)
        cv2.fillPoly(out, [px], tuple(int(c * 0.25) for c in color))
    return out


def label_classes(txt: Path) -> set[int]:
    return {r[0] for r in parse_detect_label(txt)}


def collect_candidates(data_root: Path) -> dict[int, list[tuple[str, Path]]]:
    """Map class_id -> list of (split, label_path)."""
    buckets: dict[int, list] = {0: [], 1: [], 2: []}  # 2 = both classes
    for split in ("train", "val"):
        lb_dir = data_root / "labels" / split
        if not lb_dir.is_dir():
            continue
        for txt in lb_dir.glob("*.txt"):
            cls = label_classes(txt)
            if not cls:
                continue
            item = (split, txt)
            if cls == {0}:
                buckets[0].append(item)
            elif cls == {1}:
                buckets[1].append(item)
            else:
                buckets[2].append(item)
    return buckets


def sample_items(buckets: dict, n_point: int, n_line: int, n_total: int, seed: int) -> list[tuple[str, Path]]:
    rng = random.Random(seed)
    picked: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def take(pool: list, k: int) -> None:
        rng.shuffle(pool)
        for item in pool:
            if k <= 0 or len(picked) >= n_total:
                return
            key = str(item[1])
            if key in seen:
                continue
            seen.add(key)
            picked.append(item)
            k -= 1

    take(list(buckets[0]), n_point)
    take(list(buckets[1]), n_line)
    rest = buckets[0] + buckets[1] + buckets[2]
    rng.shuffle(rest)
    for item in rest:
        if len(picked) >= n_total:
            break
        key = str(item[1])
        if key not in seen:
            seen.add(key)
            picked.append(item)
    return picked[:n_total]


def make_panel(img: np.ndarray, detect_boxes: list, seg_polys: list | None) -> np.ndarray:
    h, w = img.shape[:2]
    if seg_polys:
        left = draw_seg(img, seg_polys)
        right = draw_detect(img, detect_boxes)
        cv2.putText(left, "SEG (original)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "DET (converted)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panel = np.hstack([left, right])
    else:
        panel = draw_detect(img, detect_boxes)
        cv2.putText(panel, "DET bbox", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # legend
    cv2.rectangle(panel, (8, h - 50), (200, h - 8), (0, 0, 0), -1)
    cv2.putText(panel, "Blue=point  Red=line", (12, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    seg_src = args.seg_src.resolve() if args.seg_src and str(args.seg_src).strip() else None
    out_dir = (args.out or (data / "vis_detect_check")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = collect_candidates(data)
    items = sample_items(buckets, args.point, args.line, args.num, args.seed)

    stats = {"ok": 0, "missing_img": 0, "empty": 0, "point_imgs": 0, "line_imgs": 0, "both": 0}
    for split, det_txt in items:
        stem = det_txt.stem
        img_path = find_image(data / "images" / split, stem)
        if img_path is None:
            stats["missing_img"] += 1
            continue
        img = imread_unicode(img_path)
        if img is None:
            stats["missing_img"] += 1
            continue
        det_boxes = parse_detect_label(det_txt)
        if not det_boxes:
            stats["empty"] += 1
        cls_set = {b[0] for b in det_boxes}
        if cls_set == {0}:
            stats["point_imgs"] += 1
        elif cls_set == {1}:
            stats["line_imgs"] += 1
        elif 0 in cls_set and 1 in cls_set:
            stats["both"] += 1
        elif 0 in cls_set:
            stats["point_imgs"] += 1
        elif 1 in cls_set:
            stats["line_imgs"] += 1

        seg_polys = None
        if seg_src:
            seg_txt = seg_src / "labels" / split / f"{stem}.txt"
            seg_polys = parse_seg_label(seg_txt)
            if not seg_polys:
                seg_polys = None

        panel = make_panel(img, det_boxes, seg_polys)
        out_name = f"{split}_{stem}.jpg"
        imwrite_unicode(out_dir / out_name, panel)
        stats["ok"] += 1

    summary = out_dir / "_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"dataset: {data}",
                f"seg_compare: {seg_src or 'off'}",
                f"saved: {stats['ok']} / requested {args.num}",
                f"point-ish: {stats['point_imgs']}, line-ish: {stats['line_imgs']}, both: {stats['both']}",
                f"missing_img: {stats['missing_img']}, empty_labels: {stats['empty']}",
                "Legend: Blue=point, Red=line",
                "Left=SEG polygon (original), Right=DET bbox (converted)" if seg_src else "DET bbox only",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved {stats['ok']} images -> {out_dir}")
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
