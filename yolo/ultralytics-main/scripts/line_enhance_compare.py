from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray | None:
    # More robust than cv2.imread() on Windows/non-ASCII paths.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_gray(im: np.ndarray) -> np.ndarray:
    if im.ndim == 2:
        return im.astype(np.uint8)
    if im.shape[2] == 1:
        return im[..., 0].astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def shift_image(src: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = src.shape[:2]
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def line_kernel(length: int, orientation: str) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    if orientation == "h":
        k[c, :] = 1.0
    elif orientation == "v":
        k[:, c] = 1.0
    elif orientation == "d1":
        for i in range(length):
            k[i, i] = 1.0
    elif orientation == "d2":
        for i in range(length):
            k[i, length - 1 - i] = 1.0
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    return k / np.sum(k)


def enhance_local_contrast(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = cv2.GaussianBlur(gray, (21, 21), 0)
    bright = cv2.subtract(gray, bg)
    dark = cv2.subtract(bg, gray)
    return bright, dark


def enhance_morph(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    return bright, dark


def enhance_symmetric(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = gray.astype(np.float32)
    # orientation: (name, perpendicular shift vector)
    dirs = [
        ("h", (0.0, 1.0)),
        ("v", (1.0, 0.0)),
        ("d1", (0.7071, -0.7071)),
        ("d2", (0.7071, 0.7071)),
    ]
    lengths = [9, 15]
    widths = [2, 4]
    best_bright = np.zeros_like(g, dtype=np.float32)
    best_dark = np.zeros_like(g, dtype=np.float32)
    eps = 6.0

    for name, perp in dirs:
        for ln in lengths:
            k = line_kernel(ln, name)
            center = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
            for w in widths:
                dx = perp[0] * w
                dy = perp[1] * w
                side1 = shift_image(center, dx, dy)
                side2 = shift_image(center, -dx, -dy)
                bg = 0.5 * (side1 + side2)
                signed = center - bg
                contrast = np.abs(signed)
                side_diff = np.abs(side1 - side2)
                symmetry = 1.0 - side_diff / (contrast + side_diff + eps)
                symmetry = np.clip(symmetry, 0.0, 1.0)
                resp = contrast * symmetry
                bright = np.where(signed > 0, resp, 0.0)
                dark = np.where(signed < 0, resp, 0.0)
                best_bright = np.maximum(best_bright, bright)
                best_dark = np.maximum(best_dark, dark)

    bright_u8 = cv2.normalize(best_bright, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dark_u8 = cv2.normalize(best_dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return bright_u8, dark_u8


def get_methods():
    return {
        "local": enhance_local_contrast,
        "morph": enhance_morph,
        "symmetric": enhance_symmetric,
    }


def ring_mask(mask: np.ndarray, ksize: int = 21) -> np.ndarray:
    ker = np.ones((ksize, ksize), np.uint8)
    dil = cv2.dilate(mask, ker, iterations=1)
    return np.logical_and(dil > 0, mask == 0)


def load_shapes(json_path: Path) -> list[np.ndarray]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    polys = []
    for s in data.get("shapes", []):
        if "Line" not in str(s.get("label", "")):
            continue
        pts = s.get("points", [])
        if len(pts) < 3:
            continue
        polys.append(np.array(pts, dtype=np.int32))
    return polys


def eval_one_shape(bright: np.ndarray, dark: np.ndarray, poly: np.ndarray) -> dict[str, float] | None:
    h, w = bright.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    inside = mask == 1
    if inside.sum() < 20:
        return None
    outside = ring_mask(mask, 21)
    if outside.sum() < 40:
        return None

    b_in = float(bright[inside].mean())
    d_in = float(dark[inside].mean())
    b_out = float(bright[outside].mean())
    d_out = float(dark[outside].mean())
    b_diff = b_in - b_out
    d_diff = d_in - d_out
    score = max(b_diff, d_diff)
    polarity_conf = abs(b_diff - d_diff) / (abs(b_diff) + abs(d_diff) + 1e-6)
    return {
        "bright_in": b_in,
        "dark_in": d_in,
        "bright_out": b_out,
        "dark_out": d_out,
        "bright_diff": b_diff,
        "dark_diff": d_diff,
        "score": score,
        "polarity_conf": polarity_conf,
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    arr = np.array([r["score"] for r in rows], dtype=np.float32)
    conf = np.array([r["polarity_conf"] for r in rows], dtype=np.float32)
    return {
        "n": float(len(rows)),
        "score_mean": float(arr.mean()) if len(arr) else 0.0,
        "score_p75": float(np.percentile(arr, 75)) if len(arr) else 0.0,
        "score_positive_ratio": float((arr > 0).mean()) if len(arr) else 0.0,
        "polarity_conf_mean": float(conf.mean()) if len(conf) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser("Line enhancement compare")
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--count", default=80, type=int)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = get_methods()
    image_files = sorted(args.input_dir.glob("*.bmp"))
    selected = []
    for fp in image_files:
        if len(selected) >= args.count:
            break
        if not fp.with_suffix(".json").exists():
            continue
        shapes = load_shapes(fp.with_suffix(".json"))
        if shapes:
            selected.append((fp, shapes))

    detail_rows = []
    method_rows = defaultdict(list)

    valid_images = 0
    for fp, shapes in selected:
        im = read_image(fp)
        if im is None:
            continue
        valid_images += 1
        gray = to_gray(im)
        for method_name, method_fn in methods.items():
            bright, dark = method_fn(gray)
            for idx, poly in enumerate(shapes):
                row = eval_one_shape(bright, dark, poly)
                if row is None:
                    continue
                out = {"file": fp.name, "shape_idx": idx, "method": method_name}
                out.update(row)
                detail_rows.append(out)
                method_rows[method_name].append(out)

    detail_csv = args.output_dir / "line_enhance_detail.csv"
    with detail_csv.open("w", newline="", encoding="utf-8-sig") as f:
        if detail_rows:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_rows = []
    for name in methods:
        sm = summarize(method_rows[name])
        summary_rows.append({"method": name, **{k: round(v, 4) for k, v in sm.items()}})
    summary_csv = args.output_dir / "line_enhance_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"selected_images={len(selected)}")
    print(f"valid_images={valid_images}")
    print(f"detail_csv={detail_csv}")
    print(f"summary_csv={summary_csv}")
    print("summary:")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
