"""
离线批量 MIXED 预处理（GPU 优先），训练时用 defect_preprocess=none，大幅提速。

示例:
  python scripts/batch_offline_mixed.py ^
    --src-root "M:\压印 - 副本\dataSet-原始-切割-已筛选" ^
    --dst-root "M:\压印 - 副本\dataSet-原始-切割-已筛选-mixed" ^
    --mode mixed --device cuda
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.defect_preprocess import (
    apply_defect_preprocess,
    mixed_preprocess_batch_torch,
    mixed_preprocess_torch,
)

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


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def process_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path,
                  mode: str, device: str, batch_size: int, use_gpu: bool) -> tuple[int, int]:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    files = list_images(src_img)
    ok = fail = 0
    t0 = time.time()

    if use_gpu and mode == "mixed":
        for i in range(0, len(files), batch_size):
            chunk = files[i:i + batch_size]
            images = []
            valid = []
            for fp in chunk:
                im = imread_unicode(fp)
                if im is None:
                    fail += 1
                    continue
                images.append(im)
                valid.append(fp)
            if not images:
                continue
            try:
                outs = mixed_preprocess_batch_torch(images, device=device)
            except Exception:
                outs = [apply_defect_preprocess(im, mode) for im in images]
            for fp, out in zip(valid, outs):
                if imwrite_unicode(dst_img / fp.name, out):
                    lbl = src_lbl / f"{fp.stem}.txt"
                    if lbl.exists():
                        shutil.copy2(lbl, dst_lbl / lbl.name)
                    ok += 1
                else:
                    fail += 1
            if (i + batch_size) % 200 == 0 or i + batch_size >= len(files):
                print(f"  progress {min(i + batch_size, len(files))}/{len(files)}")
    else:
        for n, fp in enumerate(files, 1):
            im = imread_unicode(fp)
            if im is None:
                fail += 1
                continue
            if use_gpu and mode == "mixed":
                try:
                    out = mixed_preprocess_torch(im, device=device)
                except Exception:
                    out = apply_defect_preprocess(im, mode)
            else:
                out = apply_defect_preprocess(im, mode)
            if imwrite_unicode(dst_img / fp.name, out):
                lbl = src_lbl / f"{fp.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, dst_lbl / lbl.name)
                ok += 1
            else:
                fail += 1
            if n % 500 == 0:
                print(f"  progress {n}/{len(files)}")

    print(f"  done {ok} ok, {fail} fail, {time.time() - t0:.1f}s")
    return ok, fail


def main():
    p = argparse.ArgumentParser(description="Offline batch MIXED preprocess for faster training")
    p.add_argument("--src-root", default=r"M:\压印 - 副本\dataSet-原始-切割-已筛选")
    p.add_argument("--dst-root", default=r"M:\压印 - 副本\dataSet-原始-切割-已筛选-mixed")
    p.add_argument("--mode", default="mixed", choices=["mixed", "point", "line"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true", help="force CPU OpenCV preprocess")
    args = p.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    use_gpu = not args.cpu

    total_ok = total_fail = 0
    for split in ("train", "val"):
        print(f"=== {split} ===")
        ok, fail = process_split(
            src / "images" / split,
            src / "labels" / split,
            dst / "images" / split,
            dst / "labels" / split,
            args.mode,
            args.device,
            args.batch_size,
            use_gpu,
        )
        total_ok += ok
        total_fail += fail

    yaml_text = f"""path: {dst}
train: images/train
val: images/val
names:
  0: point
  1: line
"""
    (dst / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"total: ok={total_ok}, fail={total_fail}")
    print(f"output -> {dst}")
    print("训练命令: --data 上述路径 --defect-preprocess none")


if __name__ == "__main__":
    main()
