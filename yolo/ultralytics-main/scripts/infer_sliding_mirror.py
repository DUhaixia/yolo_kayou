"""
原图滑窗推理 + 按输入目录结构保存，文件名不变。

流程：滑窗 → 同位置 mask 合并 → NMS → 保存
  有预测结果 → 保存带标注的可视化图（原文件名）
  无预测结果 → 复制原图（原文件名）

示例:
  python scripts/infer_sliding_mirror.py ^
    --model H:/Python_cls/YOLO1111111/yolo/runs/segment/runs/seg-weitiao06077/defect_seg/weights/best.pt ^
    --source "H:/卡游/压印testall2/骑行" ^
    --out "G:/卡游/压印testall2/骑行_sliding_pred" ^
    --patch 512 --stride 384 --conf 0.35 --batch 32 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.defect_preprocess import apply_defect_preprocess
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import select_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
DEFAULT_COLOR = (0, 255, 255)


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".bmp"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def list_images_here(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_subsets(source: Path) -> list[Path]:
    subsets = []
    for d in sorted(p for p in source.rglob("*") if p.is_dir()):
        if list_images_here(d):
            subsets.append(d)
    if list_images_here(source):
        subsets.insert(0, source)
    return subsets


def get_sliding_coords(img_h: int, img_w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(img_h - patch, 0) + 1, stride))
    xs = list(range(0, max(img_w - patch, 0) + 1, stride))
    if not ys or ys[-1] + patch < img_h:
        ys.append(max(img_h - patch, 0))
    if not xs or xs[-1] + patch < img_w:
        xs.append(max(img_w - patch, 0))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def nms_by_class(dets: list[dict], iou_thr: float) -> list[dict]:
    if not dets:
        return []
    out: list[dict] = []
    by_cls: dict[int, list[dict]] = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)

    for items in by_cls.values():
        boxes = np.array([d["xyxy"] for d in items], dtype=np.float32)
        scores = np.array([d["conf"] for d in items], dtype=np.float32)
        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thr)
        if len(idxs) == 0:
            continue
        for i in idxs.flatten():
            out.append(items[int(i)])
    return out


def preprocess_patch(patch: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return patch
    return apply_defect_preprocess(patch, mode)


def extract_patch_mask(result, index: int) -> np.ndarray | None:
    if result.masks is None or len(result.masks) <= index:
        return None
    masks = result.masks.data
    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks, dtype=torch.float32)
    else:
        masks = masks.float().cpu()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    scaled = ops.scale_masks(masks[None], result.masks.orig_shape)[0]
    mb = (scaled[index].numpy() > 0.5).astype(np.uint8)
    return mb if mb.any() else None


def xyxy_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def sync_xyxy_from_mask(det: dict, h: int, w: int) -> None:
    mb = det_to_mask(det, h, w)
    xyxy = xyxy_from_mask((mb > 0).astype(np.uint8))
    if xyxy is not None:
        det["xyxy"] = xyxy


def sliding_predict_one(
    model: YOLO,
    img_bgr: np.ndarray,
    patch: int,
    stride: int,
    conf: float,
    imgsz: int,
    defect_preprocess: str,
    batch_size: int,
    names: dict,
    device,
    use_half: bool,
) -> tuple[list[dict], int]:
    h, w = img_bgr.shape[:2]
    coords = get_sliding_coords(h, w, patch, stride)
    all_dets: list[dict] = []
    need_copy = defect_preprocess != "none"

    with torch.inference_mode():
        for start in range(0, len(coords), batch_size):
            chunk = coords[start:start + batch_size]
            if need_copy:
                patches = [
                    preprocess_patch(img_bgr[y0:y0 + patch, x0:x0 + patch].copy(), defect_preprocess)
                    for y0, x0 in chunk
                ]
            else:
                patches = [img_bgr[y0:y0 + patch, x0:x0 + patch] for y0, x0 in chunk]
            results = model.predict(
                patches, imgsz=imgsz, conf=conf, verbose=False, device=device, half=use_half,
            )

            for (y0, x0), r in zip(chunk, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i].item())
                    score = float(r.boxes.conf[i].item())

                    mb = extract_patch_mask(r, i)
                    if mb is None:
                        continue

                    xyxy_local = xyxy_from_mask(mb)
                    if xyxy_local is None:
                        continue
                    xyxy = xyxy_local.copy()
                    xyxy[[0, 2]] += x0
                    xyxy[[1, 3]] += y0

                    all_dets.append({
                        "xyxy": xyxy,
                        "cls": cls_id,
                        "conf": score,
                        "patch_y": y0,
                        "patch_x": x0,
                        "mask_local": mb,
                        "name": names.get(cls_id, str(cls_id)),
                    })

    return all_dets, len(coords)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)


def det_to_mask(det: dict, h: int, w: int) -> np.ndarray:
    mask = det.get("mask")
    if mask is not None and mask.shape[:2] == (h, w):
        return (mask > 0).astype(np.uint8) * 255
    mb = det.get("mask_local")
    if mb is not None:
        y0, x0 = int(det["patch_y"]), int(det["patch_x"])
        ph, pw = mb.shape
        y2, x2 = min(y0 + ph, h), min(x0 + pw, w)
        mh, mw = y2 - y0, x2 - x0
        full = np.zeros((h, w), np.uint8)
        full[y0:y2, x0:x2] = mb[:mh, :mw] * 255
        return full
    return np.zeros((h, w), np.uint8)


def det_mask_cached(det: dict, h: int, w: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    key = id(det)
    if key not in cache:
        cache[key] = det_to_mask(det, h, w)
    return cache[key]


def is_line_det(d: dict) -> bool:
    return "line" in str(d.get("name", "")).lower()


def axis_adjacent(a: dict, b: dict, axis: str, overlap_ratio: float, gap_thr: float) -> bool:
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    if axis == "y":
        cross_ov = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        cross_min = max(1e-6, min(ax2 - ax1, bx2 - bx1))
        along_ov = max(0.0, min(ay2, by2) - max(ay1, by1))
        along_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
        along_span = max(ay2, by2) - min(ay1, by1)
    else:
        cross_ov = max(0.0, min(ay2, by2) - max(ay1, by1))
        cross_min = max(1e-6, min(ay2 - ay1, by2 - by1))
        along_ov = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        along_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
        along_span = max(ax2, bx2) - min(ax1, bx1)

    if cross_ov / cross_min < overlap_ratio:
        return False
    return along_ov > 0 or along_gap <= max(gap_thr, 0.15 * along_span)


def dilated_masks_touch(a: dict, b: dict, h: int, w: int, radius: int) -> bool:
    if radius <= 0:
        return False
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ma = cv2.dilate(det_to_mask(a, h, w), kernel)
    mb = cv2.dilate(det_to_mask(b, h, w), kernel)
    return bool(np.logical_and(ma > 0, mb > 0).any())


def should_merge_dets(
    a: dict,
    b: dict,
    h: int,
    w: int,
    merge_iou: float,
    stride: int = 256,
    mask_cache: dict[int, np.ndarray] | None = None,
    fast: bool = False,
) -> bool:
    if box_iou(a["xyxy"], b["xyxy"]) >= merge_iou:
        return True
    if fast:
        return False

    if mask_cache is not None:
        if mask_iou_roi(a, b, h, w, mask_cache) >= merge_iou * 0.5:
            return True
    else:
        ma = det_to_mask(a, h, w)
        mb = det_to_mask(b, h, w)
        inter = int(np.logical_and(ma > 0, mb > 0).sum())
        union = int(np.logical_or(ma > 0, mb > 0).sum())
        if union > 0 and inter / union >= merge_iou * 0.5:
            return True

    both_line = is_line_det(a) and is_line_det(b)
    if both_line:
        gap_thr = max(stride * 0.55, 48.0)
        if axis_adjacent(a, b, "y", 0.30, gap_thr):
            return True
        if axis_adjacent(a, b, "x", 0.30, gap_thr):
            return True
    elif axis_adjacent(a, b, "y", 0.50, 20.0):
        return True

    if both_line and mask_cache is not None:
        dilate_r = max(stride // 6, 12)
        if dilated_masks_touch(a, b, h, w, dilate_r):
            return True

    return False


def _union_find_groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def close_line_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def merge_overlapping_dets(
    dets: list[dict],
    img_shape: tuple[int, ...],
    merge_iou: float,
    stride: int = 256,
    max_items: int = 80,
) -> list[dict]:
    if not dets:
        return []

    h, w = int(img_shape[0]), int(img_shape[1])
    by_cls: dict[int, list[dict]] = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)

    fast = len(dets) > max_items
    mask_cache: dict[int, np.ndarray] = {}
    merged: list[dict] = []
    for cls_id, items in by_cls.items():
        n = len(items)
        pairs: list[tuple[int, int]] = []
        use_fast = fast or n > max_items
        for i in range(n):
            for j in range(i + 1, n):
                if should_merge_dets(
                    items[i], items[j], h, w, merge_iou, stride=stride,
                    mask_cache=mask_cache, fast=use_fast,
                ):
                    pairs.append((i, j))

        for group_idx in _union_find_groups(n, pairs):
            group = [items[i] for i in group_idx]
            union_mask = np.zeros((h, w), np.uint8)
            for g in group:
                union_mask = cv2.bitwise_or(union_mask, det_mask_cached(g, h, w, mask_cache))

            if is_line_det(group[0]) and len(group) > 1:
                union_mask = close_line_mask(union_mask, max(stride // 8, 8))

            if not union_mask.any():
                continue

            ys, xs = np.where(union_mask > 0)
            xyxy = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

            merged.append({
                "xyxy": xyxy,
                "cls": cls_id,
                "conf": max(g["conf"] for g in group),
                "mask": union_mask,
                "name": group[0]["name"],
            })
    for m in merged:
        sync_xyxy_from_mask(m, h, w)
    return merged


def det_mask_area(det: dict, h: int, w: int) -> int:
    mb = det.get("mask_local")
    if mb is not None:
        return int(np.count_nonzero(mb))
    mask = det.get("mask")
    if mask is not None:
        return int(np.count_nonzero(mask))
    return int(np.count_nonzero(det_to_mask(det, h, w)))


def mask_iou_roi(a: dict, b: dict, h: int, w: int, cache: dict[int, np.ndarray]) -> float:
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    x1 = max(0, int(min(ax1, bx1)))
    y1 = max(0, int(min(ay1, by1)))
    x2 = min(w, int(max(ax2, bx2)) + 1)
    y2 = min(h, int(max(ay2, by2)) + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ma = det_mask_cached(a, h, w, cache)[y1:y2, x1:x2] > 0
    mb = det_mask_cached(b, h, w, cache)[y1:y2, x1:x2] > 0
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union > 0 else 0.0


def is_valid_det(det: dict, h: int, w: int, min_mask_area: int) -> bool:
    return det_mask_area(det, h, w) >= min_mask_area


def filter_valid_dets(dets: list[dict], img_shape: tuple[int, ...], min_mask_area: int) -> list[dict]:
    h, w = int(img_shape[0]), int(img_shape[1])
    return [d for d in dets if is_valid_det(d, h, w, min_mask_area)]


def draw_on_full_image(img: np.ndarray, dets: list[dict], mask_alpha: float = 0.35) -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    h, w = vis.shape[:2]

    for d in dets:
        color = CLASS_COLORS.get(d["cls"], DEFAULT_COLOR)
        label = f"{d['name']} {d['conf']:.2f}"

        mask = det_to_mask(d, h, w)
        if mask.any():
            overlay[mask > 0] = color

        x1, y1, x2, y2 = d["xyxy"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(vis, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if mask_alpha > 0 and dets:
        vis = cv2.addWeighted(overlay, mask_alpha, vis, 1.0 - mask_alpha, 0)
    return vis


def finalize_dets(
    dets: list[dict],
    img_shape: tuple[int, ...],
    min_mask_area: int,
) -> list[dict]:
    return filter_valid_dets(dets, img_shape, min_mask_area)


def run_postprocess(
    raw_dets: list[dict],
    img_shape: tuple[int, ...],
    conf_point: float,
    conf_line: float,
    min_mask_area: int,
    merge_iou: float,
    stride: int,
    iou: float,
    no_merge: bool,
    max_merge_dets: int,
    max_merge_items: int,
) -> tuple[list[dict], int, int]:
    """先置信度过滤再合并，避免 raw 过多时 merge 卡死。"""
    after_conf = filter_by_class_conf(raw_dets, conf_point, conf_line)
    dets = after_conf
    if dets and not no_merge:
        if len(dets) > max_merge_dets:
            dets = nms_by_class(dets, iou)
        else:
            dets = merge_overlapping_dets(
                dets, img_shape, merge_iou, stride=stride, max_items=max_merge_items,
            )
            dets = nms_by_class(dets, iou)
    elif dets:
        dets = nms_by_class(dets, iou)
    final = finalize_dets(dets, img_shape, min_mask_area)
    return final, len(raw_dets), len(after_conf)


def filter_by_class_conf(dets: list[dict], conf_point: float, conf_line: float) -> list[dict]:
    kept = []
    for d in dets:
        name = str(d["name"]).lower()
        thr = conf_line if "line" in name else conf_point if "point" in name else min(conf_point, conf_line)
        if d["conf"] >= thr:
            kept.append(d)
    return kept


def save_result(
    img_path: Path,
    rel_dir: Path,
    out_root: Path,
    img: np.ndarray,
    dets: list[dict],
    draw: bool,
    mask_alpha: float,
) -> bool:
    """Save under out_root/rel_dir with the same filename as input."""
    out_dir = out_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    detected = bool(dets)

    if detected and draw:
        imwrite_unicode(out_path, draw_on_full_image(img, dets, mask_alpha=mask_alpha))
    else:
        shutil.copy2(img_path, out_path)

    return detected


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


def process_subset(
    model: YOLO,
    images: list[Path],
    source: Path,
    subset: Path,
    out_root: Path,
    args,
    names: dict,
    device,
    use_half: bool,
    label: str,
) -> dict:
    rel_dir = subset.relative_to(source)
    conf_point = args.conf if args.conf_point < 0 else args.conf_point
    conf_line = args.conf if args.conf_line < 0 else args.conf_line
    model_conf = min(args.conf, conf_point, conf_line)

    counts = {"total": len(images), "detected": 0, "no_defect": 0}
    total = len(images)
    t_subset = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = pool.submit(imread_unicode, images[0]) if images else None
        for done, img_path in enumerate(images, 1):
            img = pending.result() if pending is not None else None
            if done < len(images):
                pending = pool.submit(imread_unicode, images[done])
            else:
                pending = None

            if img is None:
                print(f"  Skip unreadable image: {img_path}")
                continue

            t0 = time.time()
            raw_dets, n_patches = sliding_predict_one(
                model, img, args.patch, args.stride, model_conf, args.imgsz,
                args.defect_preprocess, args.batch, names, device, use_half,
            )
            t_infer = time.time() - t0

            t1 = time.time()
            dets, raw_n, after_conf_n = run_postprocess(
                raw_dets, img.shape, conf_point, conf_line, args.min_mask_area,
                args.merge_iou, args.stride, args.iou, args.no_merge,
                args.max_merge_dets, args.max_merge_items,
            )
            t_post = time.time() - t1

            if raw_n > args.max_merge_dets and t_post > 2.0:
                print(f"  [warn] {img_path.name}: raw={raw_n} -> conf={after_conf_n}, post={t_post:.1f}s (heavy)")

            detected = save_result(img_path, rel_dir, out_root, img, dets, args.draw, args.mask_alpha)
            if detected:
                counts["detected"] += 1
            else:
                counts["no_defect"] += 1

            if done == 1 or done == total or done % args.log_every == 0 or t_post > 2.0:
                print(
                    f"  [{label}] [{done}/{total}] {img_path.name} -> "
                    f"{'detected' if detected else 'no_defect'} "
                    f"(patches={n_patches} raw={raw_n} conf={after_conf_n} final={len(dets)} "
                    f"infer={t_infer:.2f}s post={t_post:.2f}s) | "
                    f"detected={counts['detected']}, no_defect={counts['no_defect']}"
                )

    subset_elapsed = time.time() - t_subset
    if total:
        print(f"  [{label}] done in {subset_elapsed:.1f}s, avg {subset_elapsed / total:.2f}s/img")
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sliding-window inference with mirrored output directory structure")
    p.add_argument(
        "--model",
        default=r"H:\Python_cls\YOLO1111111\yolo\ultralytics-main\runs\segment\runs\seg-weitiao06078\defect_seg\weights\best.pt",
    )
    p.add_argument("--source", default="G:/BaiduNetdiskDownload/新一批图像/好品")
    p.add_argument("--out", default="G:/卡游/haopindu3")
    p.add_argument("--summary", type=str, default="预测汇总.csv", help="Summary filename written under --out")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--stride", type=int, default=512, help="滑窗步长，越大越快（默认384，原256）")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--conf-point", type=float, default=-1.0)
    p.add_argument("--conf-line", type=float, default=-1.0)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    p.add_argument("--merge-iou", type=float, default=0.30, help="同位置合并 IoU 阈值")
    p.add_argument("--max-merge-dets", type=int, default=40, help="conf 过滤后超过此数量则跳过 mask 合并，直接 NMS")
    p.add_argument("--max-merge-items", type=int, default=80, help="单类检测数超过此值时 merge 仅使用 bbox 快速模式")
    p.add_argument("--no-merge", action="store_true", help="关闭同位置 mask 合并")
    p.add_argument("--batch", type=int, default=32, help="滑窗 patch 批大小，GPU 显存够可设 64")
    p.add_argument("--workers", type=int, default=4, help="并行读图线程数")
    p.add_argument("--log-every", type=int, default=20, help="每 N 张打印一次进度")
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", default=True, help="GPU FP16 推理（默认开启）")
    p.add_argument("--no-half", dest="half", action="store_false", help="关闭 FP16")
    p.add_argument("--defect-preprocess", default="none", choices=["none", "point", "line", "mixed"])
    p.add_argument("--mask-alpha", type=float, default=0.35)
    p.add_argument(
        "--min-mask-area",
        type=int,
        default=16,
        help="Minimum mask pixel area; smaller regions are treated as noise and ignored",
    )
    p.add_argument(
        "--draw",
        action="store_true",
        default=True,
        help="Draw boxes/masks on images that have detections; no-detection images keep original",
    )
    p.add_argument("--no-draw", dest="draw", action="store_false", help="Copy originals for all images (no overlay)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    subsets = find_subsets(source)
    if not subsets:
        print(f"No image-containing sub-folders found in: {source}")
        return

    print(f"Source : {source}")
    print(f"Output : {out_root}")
    print(f"Found {len(subsets)} sub-folder(s):")
    for s in subsets:
        rel = s.relative_to(source)
        print(f"  - {rel if str(rel) != '.' else '<root>'}  ({len(list_images_here(s))} images)")
    print()

    device = select_device(args.device)
    use_half = args.half and str(device).startswith("cuda")
    model = YOLO(args.model)
    try:
        model.fuse()
    except Exception:  # noqa: BLE001
        pass
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    conf_point = args.conf if args.conf_point < 0 else args.conf_point
    conf_line = args.conf if args.conf_line < 0 else args.conf_line
    model_conf = min(args.conf, conf_point, conf_line)

    # GPU 预热，避免首张图偏慢
    with torch.inference_mode():
        warmup = np.zeros((args.patch, args.patch, 3), np.uint8)
        model.predict([warmup], imgsz=args.imgsz, conf=model_conf, verbose=False, device=device, half=use_half)

    print(
        f"[sliding-mirror] patch={args.patch} stride={args.stride} conf={model_conf} "
        f"batch={args.batch} half={use_half} workers={args.workers} "
        f"merge_iou={args.merge_iou} merge={not args.no_merge}\n"
    )

    rows: list[dict] = []
    for idx, subset in enumerate(subsets, 1):
        rel = subset.relative_to(source)
        sub_name = str(rel) if str(rel) != "." else "<root>"
        images = list_images_here(subset)
        print(f"==== [{idx}/{len(subsets)}] {sub_name}  ({len(images)} images) ====")

        counts = process_subset(model, images, source, subset, out_root, args, names, device, use_half, label=sub_name)
        rows.append({"subset": sub_name, **counts})
        print(f"     -> detected={counts['detected']}, no_defect={counts['no_defect']}, total={counts['total']}\n")

    write_summary(rows, out_root, args.summary)
    print("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"Error: {e}")
        sys.exit(1)
