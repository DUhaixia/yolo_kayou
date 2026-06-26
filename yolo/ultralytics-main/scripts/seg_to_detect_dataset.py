"""Convert YOLO segmentation labels to detection (xywh bbox) and copy dataset to a new root."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils.ops import segments2boxes

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert seg polygon labels to detect bbox labels")
    p.add_argument("--src", type=Path, default=Path(r"M:\压印 - 副本\dataSet-原始"))
    p.add_argument("--dst", type=Path, default=Path(r"M:\压印 - 副本\dataSet-原始222"))
    p.add_argument(
        "--min-wh",
        type=float,
        default=0.004,
        help="Minimum normalized box width/height (e.g. 0.004 ~ 4px at 1024)",
    )
    p.add_argument("--overwrite", action="store_true", help="Remove dst if it exists before copy")
    p.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copy (saves disk)")
    return p.parse_args()


def _is_segment_line(parts: list[str]) -> bool:
    return len(parts) > 5


def seg_label_to_detect_lines(text: str, min_wh: float) -> list[str]:
    """Convert one label file content from seg polygons to detect xywh lines."""
    out: list[str] = []
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if not _is_segment_line(parts):
            # Already detect format: cls cx cy w h
            if len(parts) >= 5:
                cx, cy, w, h = map(float, parts[1:5])
                w, h = max(w, min_wh), max(h, min_wh)
                out.append(f"{cls_id} {cx:.6g} {cy:.6g} {w:.6g} {h:.6g}")
            continue
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        xywh = segments2boxes([coords])[0]
        cx, cy, w, h = xywh.tolist()
        w, h = max(float(w), min_wh), max(float(h), min_wh)
        cx = float(np.clip(cx, w / 2, 1 - w / 2))
        cy = float(np.clip(cy, h / 2, 1 - h / 2))
        out.append(f"{cls_id} {cx:.6g} {cy:.6g} {w:.6g} {h:.6g}")
    return out


def convert_label_file(src_txt: Path, dst_txt: Path, min_wh: float) -> tuple[int, int]:
    """Return (num_seg_instances, num_detect_lines)."""
    raw = src_txt.read_text(encoding="utf-8")
    lines = seg_label_to_detect_lines(raw, min_wh=min_wh)
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    seg_n = sum(1 for ln in raw.strip().splitlines() if _is_segment_line(ln.strip().split()))
    return seg_n, len(lines)


def copy_or_link_file(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def update_data_yaml(src_yaml: Path, dst_yaml: Path, dst_root: Path) -> None:
    data = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) if src_yaml.is_file() else {}
    if not isinstance(data, dict):
        data = {}
    data["path"] = str(dst_root)
    for key in ("train", "val", "test"):
        if key not in data:
            continue
        p = Path(str(data[key]))
        if p.is_absolute():
            try:
                rel = p.relative_to(src_yaml.parent)
                data[key] = str(dst_root / rel)
            except ValueError:
                data[key] = str(dst_root / "images" / key)
        else:
            data[key] = str(dst_root / p)
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source not found: {src}")
    if dst.exists() and args.overwrite:
        shutil.rmtree(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Destination exists (use --overwrite): {dst}")

    label_files = list((src / "labels").rglob("*.txt")) if (src / "labels").is_dir() else []
    if not label_files:
        raise FileNotFoundError(f"No labels under {src / 'labels'}")

    stats = {"files": 0, "seg_inst": 0, "det_inst": 0, "empty": 0}

    for src_txt in label_files:
        rel = src_txt.relative_to(src)
        dst_txt = dst / rel
        seg_n, det_n = convert_label_file(src_txt, dst_txt, min_wh=args.min_wh)
        stats["files"] += 1
        stats["seg_inst"] += seg_n
        stats["det_inst"] += det_n
        if det_n == 0:
            stats["empty"] += 1

    # Copy / symlink images and other non-label files
    copied, linked, skipped = 0, 0, 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        if rel.parts[:1] == ("labels",) and src_file.suffix.lower() == ".txt":
            continue
        dst_file = dst / rel
        if rel.parts[:1] == ("images",) and src_file.suffix.lower() in IMAGE_SUFFIXES:
            copy_or_link_file(src_file, dst_file, symlink=args.symlink_images)
            linked += args.symlink_images
            copied += not args.symlink_images
        else:
            if src_file.name == "data.yaml":
                continue
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied += 1

    src_yaml = src / "data.yaml"
    update_data_yaml(src_yaml, dst / "data.yaml", dst)

    print(f"Source : {src}")
    print(f"Output : {dst}")
    print(f"Labels converted: {stats['files']} files")
    print(f"  seg instances -> detect boxes: {stats['seg_inst']} -> {stats['det_inst']}")
    print(f"  empty label files: {stats['empty']}")
    print(f"  min_wh (normalized): {args.min_wh}")
    print(f"Images: {'symlinked' if args.symlink_images else 'copied'}")
    print(f"data.yaml written: {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
