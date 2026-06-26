"""
批量 YOLO 整图分割推理（非滑窗）：遍历各小类目录，按相同结构保存，文件名不变。

每张图直接 model.predict 一次，不涉及 patch / stride / merge。
  有预测结果 → 保存带标注的可视化图（原文件名）
  无预测结果 → 复制原图（原文件名）

滑窗推理请用 infer_sliding_mirror.py。

示例:
  python scripts/infer_batch_mirror.py ^
    --source "H:/卡游/压印testall2/骑行" ^
    --out "G:/卡游/压印testall2/骑行_pred" ^
    --model H:/Python_cls/YOLO1111111/yolo/runs/segment/runs/seg-weitiao06077/defect_seg/weights/best.pt ^
    --conf 0.5 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.defect_preprocess import apply_defect_preprocess, mixed_preprocess_batch_torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch YOLO full-image seg/detect infer (no sliding window)")
    p.add_argument(
        "--model",
        type=str,
        default=r"H:\Python_cls\YOLO1111111\yolo\runs\segment\runs\seg-weitiao0613\defect_seg\weights\best.pt",
        help="Model weights path",
    )
    p.add_argument("--source", type=str, default="G:\卡游切图\汇总_XY\好品\XY", help="Input root folder")
    p.add_argument(
        "--out",
        type=str,
        default="G:/卡游/xy",
        help="Output root; sub-folders mirror --source relative paths",
    )
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    p.add_argument("--device", type=str, default="0", help="Device, e.g. 0 or cpu")
    p.add_argument("--task", type=str, default="segment", choices=["detect", "segment"], help="Model task type")
    p.add_argument("--batch", type=int, default=16, help="Batch size for GPU inference")
    p.add_argument("--workers", type=int, default=4, help="Threads for parallel image read+preprocess")
    p.add_argument(
        "--defect-preprocess",
        type=str,
        default="none",
        choices=["none", "point", "line", "mixed"],
        help="none=raw image; mixed=legacy OpenCV preprocess",
    )
    p.add_argument(
        "--preprocess-device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Where to run mixed preprocess (only affects mixed mode)",
    )
    p.add_argument(
        "--draw",
        action="store_true",
        default=True,
        help="Draw boxes/masks on images that have detections; no-detection images keep original",
    )
    p.add_argument("--no-draw", dest="draw", action="store_false", help="Copy originals for all images (no overlay)")
    p.add_argument("--half", action="store_true", default=True, help="GPU FP16 推理（默认开启）")
    p.add_argument("--no-half", dest="half", action="store_false", help="关闭 FP16")
    p.add_argument(
        "--summary",
        type=str,
        default="预测汇总.csv",
        help="Per-subfolder summary filename written under --out",
    )
    return p.parse_args()


def imwrite_unicode(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".bmp"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images_here(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_subsets(source: Path) -> list[Path]:
    """Every directory under source (incl. source) that directly holds images."""
    subsets = []
    for d in sorted(p for p in source.rglob("*") if p.is_dir()):
        if list_images_here(d):
            subsets.append(d)
    if list_images_here(source):
        subsets.insert(0, source)
    return subsets


def has_prediction(result, task: str) -> bool:
    if task == "segment":
        if result.masks is not None:
            try:
                return len(result.masks) > 0
            except TypeError:
                return False
        if result.boxes is not None:
            return len(result.boxes) > 0
        return False
    if result.boxes is None:
        return False
    return len(result.boxes) > 0


def preprocess_one(img_path: Path, mode: str, defer_gpu: bool):
    raw = imread_unicode(img_path)
    if raw is None:
        return img_path, None
    if mode == "none" or defer_gpu:
        return img_path, raw
    return img_path, apply_defect_preprocess(raw, mode=mode)


def prefetch_preprocess(images: list[Path], mode: str, pool: ThreadPoolExecutor, max_inflight: int, defer_gpu: bool):
    futures: deque = deque()
    img_iter = iter(images)
    for _ in range(max_inflight):
        try:
            p = next(img_iter)
        except StopIteration:
            break
        futures.append(pool.submit(preprocess_one, p, mode, defer_gpu))

    while futures:
        fut = futures.popleft()
        try:
            p = next(img_iter)
            futures.append(pool.submit(preprocess_one, p, mode, defer_gpu))
        except StopIteration:
            pass
        yield fut.result()


def chunked(iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def save_result(
    img_path: Path,
    rel_dir: Path,
    out_root: Path,
    result,
    task: str,
    draw: bool,
) -> bool:
    """Save under out_root/rel_dir with the same filename as input."""
    out_dir = out_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    detected = has_prediction(result, task)

    if detected and draw:
        imwrite_unicode(out_path, result.plot())
    else:
        shutil.copy2(img_path, out_path)

    return detected


def process_subset(
    model,
    images: list[Path],
    source: Path,
    subset: Path,
    out_root: Path,
    args,
    label: str,
) -> dict:
    rel_dir = subset.relative_to(source)
    rel_str = "" if str(rel_dir) == "." else str(rel_dir)
    counts = {"total": len(images), "detected": 0, "no_defect": 0}
    done = 0
    batch_size = max(1, args.batch)
    workers = max(1, args.workers)
    max_inflight = batch_size * 2

    use_gpu_mixed = args.preprocess_device == "cuda" and args.defect_preprocess == "mixed"
    torch_device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepped = prefetch_preprocess(images, args.defect_preprocess, pool, max_inflight, defer_gpu=use_gpu_mixed)
        for batch in chunked(prepped, batch_size):
            valid = []
            for img_path, payload in batch:
                if payload is None:
                    print(f"  Skip unreadable: {img_path}")
                    continue
                valid.append((img_path, payload))
            done += len(batch)
            if not valid:
                continue

            if use_gpu_mixed:
                inputs = mixed_preprocess_batch_torch([payload for _, payload in valid], device=torch_device)
            else:
                inputs = [payload for _, payload in valid]

            results = model.predict(
                source=inputs,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                half=args.half and args.device != "cpu",
            )

            for (img_path, _), r in zip(valid, results):
                detected = save_result(img_path, rel_dir, out_root, r, args.task, args.draw)
                if detected:
                    counts["detected"] += 1
                else:
                    counts["no_defect"] += 1

            print(
                f"  [{label}] [{done}/{counts['total']}] "
                f"detected={counts['detected']}, no_defect={counts['no_defect']}"
            )

    return counts


def write_summary(rows: list[dict], out_root: Path, summary_name: str) -> None:
    headers = ["小类", "总数", "有缺陷", "无缺陷"]
    cols = ["subset", "total", "detected", "no_defect"]
    table = [[row[c] for c in cols] for row in rows]
    grand = {k: sum(r[k] for r in rows) for k in ("total", "detected", "no_defect")}
    table.append(["合计", grand["total"], grand["detected"], grand["no_defect"]])

    csv_path = out_root / summary_name
    out_root.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(table)
    print(f"Summary: {csv_path}")


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    subsets = find_subsets(source)
    if not subsets:
        print(f"No image-containing folders found in: {source}")
        return

    print(f"Source : {source}")
    print(f"Output : {out_root}")
    print(f"Found {len(subsets)} sub-folder(s):")
    for s in subsets:
        rel = s.relative_to(source)
        print(f"  - {rel if str(rel) != '.' else '<root>'}  ({len(list_images_here(s))} images)")
    print()

    use_gpu_mixed = args.preprocess_device == "cuda" and args.defect_preprocess == "mixed"
    if args.preprocess_device == "cuda" and not use_gpu_mixed and args.defect_preprocess != "none":
        print(f"Note: GPU preprocess only supports 'mixed'; using CPU for '{args.defect_preprocess}'.")

    model = YOLO(args.model)
    try:
        model.fuse()
    except Exception:  # noqa: BLE001
        pass
    use_half = args.half and args.device != "cpu"
    if use_half:
        dummy = np.zeros((640, 640, 3), np.uint8)
        model.predict([dummy], imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, half=True)
    print(f"[batch-mirror] batch={args.batch} half={use_half} imgsz={args.imgsz}\n")
    rows: list[dict] = []

    for idx, subset in enumerate(subsets, 1):
        rel = subset.relative_to(source)
        sub_name = str(rel) if str(rel) != "." else "<root>"
        images = list_images_here(subset)
        print(f"==== [{idx}/{len(subsets)}] {sub_name}  ({len(images)} images) ====")
        counts = process_subset(model, images, source, subset, out_root, args, label=sub_name)
        rows.append({"subset": sub_name, **counts})
        print(f"     -> detected={counts['detected']}, no_defect={counts['no_defect']}, total={counts['total']}\n")

    write_summary(rows, out_root, args.summary)
    print("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        sys.exit(1)
