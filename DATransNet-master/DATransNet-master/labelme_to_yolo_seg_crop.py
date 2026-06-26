"""
LabelMe -> YOLO-seg dataset with defect-centered 512x512 crops.

- point label -> class 0
- line label  -> class 1
- crop centered on each defect; all defects inside the crop are written to txt
- dense clusters: one crop per 512 window (skip redundant overlapping crops)
- train/val split 8:2 (grouped by source image to avoid leakage)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_MAP = {"point": 0, "line": 1}
CROP_SIZE = 512


def load_labelme(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def read_image(img_path: Path) -> np.ndarray | None:
    """Read image via bytes to support non-ASCII paths on Windows."""
    try:
        data = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except OSError:
        pass
    return None


def write_image(img_path: Path, img: np.ndarray) -> bool:
    """Write image via imencode to support non-ASCII paths on Windows."""
    suffix = img_path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        return False
    img_path.write_bytes(buf.tobytes())
    return True


def find_image_path(json_path: Path, data: dict) -> Path | None:
    stem = json_path.stem
    parent = json_path.parent
    candidates = [
        parent / data.get("imagePath", ""),
        parent / f"{stem}.bmp",
        parent / f"{stem}.jpg",
        parent / f"{stem}.png",
    ]
    for p in candidates:
        if p and p.exists():
            return p
    for ext in (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        p = parent / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0, 0]), float(arr[0, 1])
    if len(arr) == 2:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    x = arr[:, 0]
    y = arr[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-6:
        return float(x.mean()), float(y.mean())
    cx = ((x + x2) * cross).sum() / (6.0 * area)
    cy = ((y + y2) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def compute_crop_box(cx: float, cy: float, img_w: int, img_h: int, size: int = CROP_SIZE) -> tuple[int, int, int, int]:
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    if img_w >= size:
        if x1 < 0:
            x1, x2 = 0, size
        elif x2 > img_w:
            x2, x1 = img_w, img_w - size
    if img_h >= size:
        if y1 < 0:
            y1, y2 = 0, size
        elif y2 > img_h:
            y2, y1 = img_h, img_h - size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def crop_image(img: np.ndarray, box: tuple[int, int, int, int], size: int = CROP_SIZE) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = img[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    if ph == size and pw == size:
        return patch
    if img.shape[0] < size or img.shape[1] < size:
        out = np.zeros((size, size, 3), dtype=img.dtype) if img.ndim == 3 else np.zeros((size, size), dtype=img.dtype)
        out[:ph, :pw] = patch
        return out
    # fallback: center pad when source image is smaller than crop target
    out = np.zeros((size, size) + ((3,) if img.ndim == 3 else ()), dtype=img.dtype)
    oy = (size - ph) // 2
    ox = (size - pw) // 2
    out[oy : oy + ph, ox : ox + pw] = patch
    return out


def clip_polygon_to_crop(
    points: list[list[float]],
    box: tuple[int, int, int, int],
    out_size: int = CROP_SIZE,
) -> list[tuple[float, float]] | None:
    x1, y1, x2, y2 = box
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    local_pts: list[tuple[float, float]] = []
    for px, py in points:
        lx = px - x1
        ly = py - y1
        if 0 <= lx < crop_w and 0 <= ly < crop_h:
            local_pts.append((lx, ly))

    if len(local_pts) < 3:
        # keep tiny defects: use bbox corners clipped to crop
        arr = np.asarray(points, dtype=np.float64)
        min_x, min_y = arr.min(axis=0)
        max_x, max_y = arr.max(axis=0)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        r = max(3.0, max(max_x - min_x, max_y - min_y) / 2.0, 2.0)
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        local_pts = []
        for t in theta:
            lx = cx + r * np.cos(t) - x1
            ly = cy + r * np.sin(t) - y1
            lx = min(max(lx, 0.0), crop_w - 1e-6)
            ly = min(max(ly, 0.0), crop_h - 1e-6)
            local_pts.append((float(lx), float(ly)))

    # scale to fixed output size if crop was edge-clamped smaller than target
    sx = out_size / crop_w
    sy = out_size / crop_h
    scaled = [(p[0] * sx, p[1] * sy) for p in local_pts]

    if len(scaled) < 3:
        return None
    return scaled


def to_yolo_seg_line(class_id: int, points: list[tuple[float, float]], size: int = CROP_SIZE) -> str:
    coords = []
    for x, y in points:
        coords.append(f"{x / size:.6f}")
        coords.append(f"{y / size:.6f}")
    return f"{class_id} " + " ".join(coords)


def centroid_in_box(cx: float, cy: float, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= cx < x2 and y1 <= cy < y2


def shape_intersects_crop(points: list[list[float]], box: tuple[int, int, int, int]) -> bool:
    arr = np.asarray(points, dtype=np.float64)
    min_x, min_y = arr.min(axis=0)
    max_x, max_y = arr.max(axis=0)
    x1, y1, x2, y2 = box
    return not (max_x < x1 or min_x >= x2 or max_y < y1 or min_y >= y2)


def collect_annotations_in_crop(
    shapes: list[dict],
    box: tuple[int, int, int, int],
    out_size: int = CROP_SIZE,
) -> list[tuple[int, list[tuple[float, float]]]]:
    """Return all defects whose polygon intersects the crop, as (class_id, local_poly)."""
    annotations: list[tuple[int, list[tuple[float, float]]]] = []
    for shape in shapes:
        if not shape_intersects_crop(shape["points"], box):
            continue
        local_poly = clip_polygon_to_crop(shape["points"], box, out_size)
        if local_poly is None:
            continue
        annotations.append((shape["class_id"], local_poly))
    return annotations


def parse_valid_shapes(data: dict) -> list[dict]:
    shapes: list[dict] = []
    for idx, shape in enumerate(data.get("shapes", [])):
        label = (shape.get("label") or "").strip().lower()
        if label not in CLASS_MAP:
            continue
        points = shape.get("points") or []
        if not points:
            continue
        cx, cy = polygon_centroid(points)
        shapes.append(
            {
                "idx": idx,
                "label": label,
                "class_id": CLASS_MAP[label],
                "points": points,
                "cx": cx,
                "cy": cy,
            }
        )
    return shapes


def process_one_json(
    json_path: Path,
    out_images: Path,
    out_labels: Path,
    name_prefix: str,
) -> list[dict]:
    data = load_labelme(json_path)
    img_path = find_image_path(json_path, data)
    if img_path is None:
        print(f"[WARN] image not found for {json_path}")
        return []

    img = read_image(img_path)
    if img is None:
        print(f"[WARN] failed to read image {img_path}")
        return []

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_h, img_w = img.shape[:2]
    valid_shapes = parse_valid_shapes(data)
    if not valid_shapes:
        return []

    records: list[dict] = []
    covered: set[int] = set()
    crop_idx = 0

    for seed in valid_shapes:
        if seed["idx"] in covered:
            continue

        box = compute_crop_box(seed["cx"], seed["cy"], img_w, img_h, CROP_SIZE)
        annotations = collect_annotations_in_crop(valid_shapes, box, CROP_SIZE)
        if not annotations:
            continue

        crop = crop_image(img, box, CROP_SIZE)
        stem = f"{name_prefix}_{crop_idx:03d}"
        img_out = out_images / f"{stem}.jpg"
        lbl_out = out_labels / f"{stem}.txt"

        write_image(img_out, crop)
        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write("\n".join(to_yolo_seg_line(cid, poly, CROP_SIZE) for cid, poly in annotations) + "\n")

        for shape in valid_shapes:
            if centroid_in_box(shape["cx"], shape["cy"], box):
                covered.add(shape["idx"])

        records.append(
            {
                "source": json_path.stem,
                "stem": stem,
                "seed_label": seed["label"],
                "num_ann": len(annotations),
            }
        )
        crop_idx += 1

    return records


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["source"], []).append(r)

    sources = list(groups.keys())
    random.Random(seed).shuffle(sources)
    val_count = max(1, int(round(len(sources) * val_ratio))) if len(sources) > 1 else 0
    val_sources = set(sources[:val_count])

    train_recs, val_recs = [], []
    for src, items in groups.items():
        if src in val_sources:
            val_recs.extend(items)
        else:
            train_recs.extend(items)
    return train_recs, val_recs


def write_data_yaml(out_dir: Path) -> None:
    yaml_path = out_dir / "data.yaml"
    content = f"""path: {out_dir.as_posix()}
train: train/images
val: val/images

names:
  0: point
  1: line
"""
    yaml_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LabelMe polygon -> YOLO-seg defect-centered crops")
    parser.add_argument("--src", type=str, default=r"G:\卡游\新建文件夹", help="source folder with images+json")
    parser.add_argument(
        "--dst",
        type=str,
        default=r"G:\卡游\yolo_seg_512",
        help="output dataset root",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global CROP_SIZE
    CROP_SIZE = args.crop_size

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    tmp_dir = dst_dir / "_tmp_all"
    tmp_images = tmp_dir / "images"
    tmp_labels = tmp_dir / "labels"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_images.mkdir(parents=True)
    tmp_labels.mkdir(parents=True)

    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No json files found in {src_dir}")

    all_records: list[dict] = []
    for jf in json_files:
        all_records.extend(process_one_json(jf, tmp_images, tmp_labels, jf.stem))

    if not all_records:
        raise SystemExit("No valid defects exported.")

    train_recs, val_recs = split_records(all_records, args.val_ratio, args.seed)

    for split_name, recs in ("train", train_recs), ("val", val_recs):
        img_dir = dst_dir / split_name / "images"
        lbl_dir = dst_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in recs:
            stem = r["stem"]
            shutil.move(str(tmp_images / f"{stem}.jpg"), str(img_dir / f"{stem}.jpg"))
            shutil.move(str(tmp_labels / f"{stem}.txt"), str(lbl_dir / f"{stem}.txt"))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    write_data_yaml(dst_dir)

    multi_ann = sum(1 for r in all_records if r["num_ann"] > 1)
    print("Done.")
    print(f"Source json: {len(json_files)}")
    print(f"Total crops: {len(all_records)} (multi-annotation crops: {multi_ann})")
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
