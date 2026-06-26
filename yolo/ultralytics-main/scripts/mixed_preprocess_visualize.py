from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.data.defect_preprocess import mixed_preprocess


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def draw_shapes(im: np.ndarray, shapes: list[dict]) -> np.ndarray:
    out = im.copy()
    for s in shapes:
        pts = np.array(s.get("points", []), dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        label = str(s.get("label", ""))
        if "Point" in label:
            color = (0, 80, 255)
        elif "Line" in label:
            color = (0, 220, 90)
        else:
            color = (255, 255, 0)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
    return out


def shape_crop(im: np.ndarray, shape: dict, min_side: int = 160) -> np.ndarray | None:
    pts = np.array(shape.get("points", []), dtype=np.float32)
    if pts.shape[0] < 3:
        return None
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    side = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]))
    side = max(float(min_side), side * 4.0)
    cx, cy = float((min_xy[0] + max_xy[0]) * 0.5), float((min_xy[1] + max_xy[1]) * 0.5)
    x0 = int(max(0, round(cx - side * 0.5)))
    y0 = int(max(0, round(cy - side * 0.5)))
    x1 = int(min(im.shape[1], x0 + side))
    y1 = int(min(im.shape[0], y0 + side))
    if x1 <= x0 or y1 <= y0:
        return None
    return im[y0:y1, x0:x1].copy()


def save_triplet_grid(path: Path, ims: list[np.ndarray], titles: list[str], tile_w: int = 520) -> None:
    assert len(ims) == len(titles)
    h0, w0 = ims[0].shape[:2]
    tile_h = int(tile_w * h0 / max(1, w0))
    label_h = 34
    cols = len(ims)
    canvas = np.full((tile_h + label_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for i, (im, title) in enumerate(zip(ims, titles)):
        x0 = i * tile_w
        im_r = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        canvas[label_h : label_h + tile_h, x0 : x0 + tile_w] = im_r
        cv2.putText(canvas, title, (x0 + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imencode(".jpg", canvas)[1].tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser("Visualize mixed preprocess outputs")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--label-type", type=str, default="all", choices=["all", "point", "line"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.bmp"))

    picked = 0
    for fp in files:
        if picked >= args.count:
            break
        jp = fp.with_suffix(".json")
        if not jp.exists():
            continue
        try:
            ann = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        shapes = []
        for s in ann.get("shapes", []):
            label = str(s.get("label", ""))
            if args.label_type == "point" and "Point" not in label:
                continue
            if args.label_type == "line" and "Line" not in label:
                continue
            if args.label_type == "all" and ("Point" not in label and "Line" not in label):
                continue
            shapes.append(s)
        if not shapes:
            continue
        im = read_image(fp)
        if im is None:
            continue

        mixed = mixed_preprocess(im)
        b, g, r = cv2.split(mixed)
        g3 = cv2.merge((g, g, g))
        b3 = cv2.merge((b, b, b))
        tri_vis = mixed.copy()  # pseudo-color by channel semantics

        orig_overlay = draw_shapes(im, shapes)
        tri_overlay = draw_shapes(tri_vis, shapes)
        g_overlay = draw_shapes(g3, shapes)
        b_overlay = draw_shapes(b3, shapes)

        stem = fp.stem
        cv2.imencode(".png", tri_vis)[1].tofile(str(args.output_dir / f"{stem}_mixed_rgb.png"))
        cv2.imencode(".png", g3)[1].tofile(str(args.output_dir / f"{stem}_g_brightmix.png"))
        cv2.imencode(".png", b3)[1].tofile(str(args.output_dir / f"{stem}_b_darkmix.png"))

        save_triplet_grid(
            args.output_dir / f"{stem}_compare_full.jpg",
            [orig_overlay, tri_overlay, g_overlay, b_overlay],
            ["original+label", "mixed RGB+label", "G channel+label", "B channel+label"],
            tile_w=420,
        )

        # Save one ROI compare for first shape
        roi_items = [shape_crop(orig_overlay, shapes[0]), shape_crop(tri_overlay, shapes[0]), shape_crop(g_overlay, shapes[0]), shape_crop(b_overlay, shapes[0])]
        if all(x is not None for x in roi_items):
            save_triplet_grid(
                args.output_dir / f"{stem}_compare_roi.jpg",
                roi_items,  # type: ignore[arg-type]
                ["roi original", "roi mixed RGB", "roi G channel", "roi B channel"],
                tile_w=320,
            )

        picked += 1

    print(f"saved_samples={picked}")
    print(f"label_type={args.label_type}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
