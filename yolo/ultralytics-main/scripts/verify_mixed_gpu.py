from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.defect_preprocess import mixed_preprocess, mixed_preprocess_torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare CPU vs GPU mixed_preprocess numerically")
    parser.add_argument("--source", type=str, required=True, help="Image file or folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for GPU version")
    parser.add_argument("--count", type=int, default=10, help="Max images to compare")
    parser.add_argument("--save-diff", type=str, default="", help="Optional folder to save diff heatmaps")
    return parser.parse_args()


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted([p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def main() -> None:
    args = parse_args()
    src = Path(args.source).expanduser().resolve()
    images = list_images(src)[: args.count]
    if not images:
        print(f"No images found in: {src}")
        return

    save_dir = Path(args.save_diff).expanduser().resolve() if args.save_diff else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    max_diffs = []
    mean_diffs = []
    for img_path in images:
        im = imread_unicode(img_path)
        if im is None:
            print(f"Skip unreadable: {img_path}")
            continue
        cpu_out = mixed_preprocess(im).astype(np.int16)
        gpu_out = mixed_preprocess_torch(im, device=args.device).astype(np.int16)
        if cpu_out.shape != gpu_out.shape:
            print(f"Shape mismatch on {img_path.name}: cpu={cpu_out.shape} gpu={gpu_out.shape}")
            continue
        diff = np.abs(cpu_out - gpu_out)
        max_d = int(diff.max())
        mean_d = float(diff.mean())
        max_diffs.append(max_d)
        mean_diffs.append(mean_d)
        print(f"{img_path.name}: max_abs_diff={max_d}, mean_abs_diff={mean_d:.4f}")

        if save_dir is not None:
            heat = np.clip(diff.max(axis=2), 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            cv2.imencode(".png", heat)[1].tofile(str(save_dir / f"{img_path.stem}_diff.png"))

    if max_diffs:
        print("\nSummary over", len(max_diffs), "images")
        print(f"  worst max_abs_diff = {max(max_diffs)}")
        print(f"  avg  max_abs_diff = {np.mean(max_diffs):.3f}")
        print(f"  avg  mean_abs_diff = {np.mean(mean_diffs):.4f}")


if __name__ == "__main__":
    main()
