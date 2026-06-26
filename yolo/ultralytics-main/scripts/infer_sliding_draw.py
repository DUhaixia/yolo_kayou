"""
原图滑窗推理：按单张图统计 point/line 缺陷数量，并绘制到原图上。

每张原图独立滑窗 → 坐标映射回全图 → 按类别 NMS 去重 → 画框/多边形 + 计数。

示例:
  python scripts/infer_sliding_draw.py ^
    --model runs/segment/.../weights/best.pt ^
    --source "M:\压印 - 副本\dataSet-原始\images\val" ^
    --out "M:\压印 - 副本\sliding_infer_val" ^
    --patch 512 --stride 256 ^
    --defect-preprocess none ^
    --conf 0.35
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.defect_preprocess import apply_defect_preprocess
from ultralytics.utils.torch_utils import select_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_COLORS = {
    0: (0, 255, 0),    # point 绿
    1: (0, 0, 255),    # line 红
}
DEFAULT_COLOR = (0, 255, 255)


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


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
    out = []
    by_cls: dict[int, list[dict]] = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)

    for cls_id, items in by_cls.items():
        boxes = np.array([d["xyxy"] for d in items], dtype=np.float32)
        scores = np.array([d["conf"] for d in items], dtype=np.float32)
        if len(boxes) == 0:
            continue
        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_thr)
        if len(idxs) == 0:
            continue
        for i in idxs.flatten():
            out.append(items[int(i)])
    return out


def preprocess_patch(patch: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return patch
    return apply_defect_preprocess(patch, mode)


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
) -> tuple[list[dict], int]:
    h, w = img_bgr.shape[:2]
    coords = get_sliding_coords(h, w, patch, stride)
    all_dets: list[dict] = []

    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        patches = []
        for y0, x0 in chunk:
            p = img_bgr[y0:y0 + patch, x0:x0 + patch].copy()
            patches.append(preprocess_patch(p, defect_preprocess))

        results = model.predict(patches, imgsz=imgsz, conf=conf, verbose=False, device=device)

        for (y0, x0), r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                xyxy = r.boxes.xyxy[i].cpu().numpy().astype(np.float32).copy()
                xyxy[[0, 2]] += x0
                xyxy[[1, 3]] += y0
                cls_id = int(r.boxes.cls[i].item())
                score = float(r.boxes.conf[i].item())
                poly = None
                if r.masks is not None and len(r.masks) > i:
                    poly = r.masks.xy[i].astype(np.float32).copy()
                    poly[:, 0] += x0
                    poly[:, 1] += y0
                all_dets.append({
                    "xyxy": xyxy,
                    "cls": cls_id,
                    "conf": score,
                    "poly": poly,
                    "name": names.get(cls_id, str(cls_id)),
                })

    return all_dets, len(coords)


def draw_detections(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for d in dets:
        cls_id = d["cls"]
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        name = d["name"]
        conf = d["conf"]
        label = f"{name} {conf:.2f}"

        if d["poly"] is not None and len(d["poly"]) >= 2:
            pts = d["poly"].astype(np.int32)
            if len(pts) >= 3:
                cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
            else:
                cv2.polylines(vis, [pts], False, color, 2, cv2.LINE_AA)

        x1, y1, x2, y2 = d["xyxy"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return vis


def draw_summary_header(vis: np.ndarray, point_n: int, line_n: int, total_n: int) -> np.ndarray:
    text = f"point={point_n}  line={line_n}  total={total_n}"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 320), 36), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def count_by_class(dets: list[dict], names: dict) -> tuple[int, int, int]:
    point_n = line_n = 0
    for d in dets:
        n = str(d["name"]).lower()
        if "point" in n:
            point_n += 1
        elif "line" in n:
            line_n += 1
    return point_n, line_n, len(dets)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_card_roi(img: np.ndarray, diff_thr: int = 12) -> tuple[int, int, int, int] | None:
    """
    自动找卡片区域：外圈灰底 + 中间有纹理的矩形。
    返回内缩前的 (x1, y1, x2, y2)，失败返回 None。
    """
    gray = to_gray(img)
    h, w = gray.shape
    border = max(3, min(h, w) // 100)
    strips = np.concatenate([
        gray[:border, :].ravel(),
        gray[-border:, :].ravel(),
        gray[:, :border].ravel(),
        gray[:, -border:].ravel(),
    ])
    bg = float(np.median(strips))
    mask = (np.abs(gray.astype(np.float32) - bg) > diff_thr).astype(np.uint8) * 255
    k = max(5, min(h, w) // 80) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < h * w * 0.05:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    return x, y, x + bw, y + bh


def shrink_rect(rect: tuple[int, int, int, int], margin: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    x1 = min(max(0, x1 + margin), img_w - 1)
    y1 = min(max(0, y1 + margin), img_h - 1)
    x2 = min(max(x1 + 1, x2 - margin), img_w)
    y2 = min(max(y1 + 1, y2 - margin), img_h)
    return x1, y1, x2, y2


def det_center(d: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = d["xyxy"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(d: dict) -> float:
    x1, y1, x2, y2 = d["xyxy"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_wh(d: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = d["xyxy"]
    return abs(x2 - x1), abs(y2 - y1)


def box_aspect(d: dict) -> float:
    bw, bh = box_wh(d)
    return max(bw, bh) / max(min(bw, bh), 1e-6)


def poly_mask_area(d: dict, shape: tuple[int, int, int]) -> float:
    poly = d.get("poly")
    if poly is None or len(poly) < 2:
        return 0.0
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = poly.astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    else:
        cv2.polylines(mask, [pts], False, 1, thickness=6)
    return float(mask.sum())


def local_gray_std(gray: np.ndarray, cx: float, cy: float, radius: int = 18) -> float:
    h, w = gray.shape
    ix, iy = int(round(cx)), int(round(cy))
    x1, x2 = max(0, ix - radius), min(w, ix + radius)
    y1, y2 = max(0, iy - radius), min(h, iy + radius)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.std(patch))


def filter_point_quality(
    dets: list[dict],
    gray: np.ndarray,
    outer_bg: np.ndarray | None,
    min_area: float,
    max_area: float,
    max_aspect: float,
    min_conf_flat: float,
    flat_std_thr: float,
) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    h, w = gray.shape
    for d in dets:
        name = str(d["name"]).lower()
        if "point" not in name:
            kept.append(d)
            continue

        cx, cy = det_center(d)
        ix, iy = int(round(cx)), int(round(cy))
        area = box_area(d)
        aspect = box_aspect(d)
        conf = d["conf"]

        if outer_bg is not None and 0 <= ix < w and 0 <= iy < h and outer_bg[iy, ix]:
            dropped += 1
            continue
        if area < min_area or area > max_area or aspect > max_aspect:
            dropped += 1
            continue
        if conf < min_conf_flat and local_gray_std(gray, cx, cy) < flat_std_thr:
            dropped += 1
            continue
        kept.append(d)
    return kept, dropped


def filter_line_quality(
    dets: list[dict],
    shape: tuple[int, int, int],
    max_box_area: float,
    min_fill_ratio: float,
) -> tuple[list[dict], int]:
    """仅去掉超大空框类 line 误检。"""
    kept, dropped = [], 0
    for d in dets:
        name = str(d["name"]).lower()
        if "line" not in name:
            kept.append(d)
            continue

        area = box_area(d)
        aspect = box_aspect(d)
        min_side = min(box_wh(d))
        mask_area = poly_mask_area(d, shape)
        fill = mask_area / max(area, 1.0)

        huge_hollow = area > max_box_area and fill < min_fill_ratio
        squat_block = area > 12000 and aspect < 1.6 and min_side > 70
        if huge_hollow or squat_block:
            dropped += 1
            continue
        kept.append(d)
    return kept, dropped


def filter_detections(
    dets: list[dict],
    img_shape: tuple[int, int],
    card_roi: tuple[int, int, int, int] | None,
    edge_margin: int,
    min_area: float,
    max_area: float,
    max_point_aspect: float,
) -> tuple[list[dict], int]:
    """过滤边缘/背景过检。返回 (保留列表, 被过滤数量)。"""
    h, w = img_shape[:2]
    dropped = 0
    kept = []

    if card_roi is not None:
        inner = shrink_rect(card_roi, edge_margin, w, h)
    else:
        inner = (edge_margin, edge_margin, w - edge_margin, h - edge_margin)

    ix1, iy1, ix2, iy2 = inner

    for d in dets:
        cx, cy = det_center(d)
        area = box_area(d)
        x1, y1, x2, y2 = d["xyxy"]
        bw, bh = x2 - x1, y2 - y1
        aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
        name = str(d["name"]).lower()

        # 1) 中心须在卡片内缩 ROI 内
        if not (ix1 <= cx <= ix2 and iy1 <= cy <= iy2):
            dropped += 1
            continue

        # 2) 点：面积 + 形状（排除细长边缘响应）
        if "point" in name:
            if area < min_area or area > max_area:
                dropped += 1
                continue
            if aspect > max_point_aspect:
                dropped += 1
                continue

        kept.append(d)

    return kept, dropped


def draw_roi_overlay(vis: np.ndarray, card_roi: tuple[int, int, int, int] | None, inner: tuple[int, int, int, int]) -> np.ndarray:
    ix1, iy1, ix2, iy2 = inner
    cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (255, 128, 0), 1, cv2.LINE_AA)
    if card_roi is not None:
        x1, y1, x2, y2 = card_roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def border_gray_ref(gray: np.ndarray) -> float:
    h, w = gray.shape
    b = max(5, min(h, w) // 80)
    strips = np.concatenate([
        gray[:b, :].ravel(), gray[-b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, -b:].ravel(),
    ])
    return float(np.median(strips))


def build_outer_background_mask(gray: np.ndarray, tol: int = 8) -> np.ndarray:
    """
    仅从图像四边洪泛填充，标记「外圈灰底」。
    不侵蚀卡片内部，避免浮雕图灰度范围窄时整图被误判。
    """
    h, w = gray.shape
    outer = np.zeros((h, w), dtype=bool)
    step = max(4, min(h, w) // 40)
    seeds = []
    for x in range(0, w, step):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(0, h, step):
        seeds.append((0, y))
        seeds.append((w - 1, y))

    for sx, sy in seeds:
        work = gray.copy()
        fmask = np.zeros((h + 2, w + 2), np.uint8)
        # 灰度单通道：loDiff/upDiff 用位置参数，OpenCV 4.13 不支持 hiDiff 关键字
        cv2.floodFill(work, fmask, (sx, sy), 255, tol, tol, cv2.FLOODFILL_FIXED_RANGE)
        outer |= work == 255
    return outer


def filter_background_edge(
    dets: list[dict],
    gray: np.ndarray,
    outer_bg: np.ndarray | None,
    min_conf_point: float,
    drop_outer_only: bool,
) -> tuple[list[dict], int]:
    """轻量过滤：默认只去掉落在外圈灰底上的低置信点。"""
    h, w = gray.shape
    dropped = 0
    kept = []

    for d in dets:
        cx, cy = det_center(d)
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < w and 0 <= iy < h):
            dropped += 1
            continue

        name = str(d["name"]).lower()
        conf = d["conf"]
        on_outer = outer_bg is not None and outer_bg[iy, ix]

        if drop_outer_only and on_outer:
            # 只删「明确在灰底上」的检测；真缺陷在卡片上不受影响
            if "point" in name and (min_conf_point <= 0 or conf < min_conf_point):
                dropped += 1
                continue
            if "line" in name and conf < 0.5:
                dropped += 1
                continue

        kept.append(d)

    return kept, dropped


def build_card_edge_distance(gray: np.ndarray, bg_tol: int = 8) -> np.ndarray:
    """卡片内部每像素到「卡片外轮廓」的距离(px)。轮廓处≈0，越往里越大。"""
    outer_bg = build_outer_background_mask(gray, tol=bg_tol)
    fg = (~outer_bg).astype(np.uint8) * 255
    return cv2.distanceTransform(fg, cv2.DIST_L2, 5)


def line_distances(d: dict, dist_map: np.ndarray) -> tuple[float, float]:
    h, w = dist_map.shape
    pts = d["poly"] if d.get("poly") is not None and len(d["poly"]) >= 2 else None
    if pts is None:
        cx, cy = det_center(d)
        pts = np.array([[cx, cy]], dtype=np.float32)
    vals = []
    for px, py in pts:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w and 0 <= iy < h:
            vals.append(float(dist_map[iy, ix]))
    if not vals:
        return 0.0, 0.0
    return min(vals), float(np.mean(vals))


def center_edge_distance(d: dict, dist_map: np.ndarray) -> float:
    h, w = dist_map.shape
    cx, cy = det_center(d)
    ix, iy = int(round(cx)), int(round(cy))
    if 0 <= ix < w and 0 <= iy < h:
        return float(dist_map[iy, ix])
    return 0.0


def line_oriented_angle_aspect(d: dict) -> tuple[float, float]:
    poly = d.get("poly")
    if poly is not None and len(poly) >= 2:
        rect = cv2.minAreaRect(poly.astype(np.float32))
        rw, rh = rect[1]
        ang = rect[2]
        if rw < rh:
            ang += 90.0
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        return abs(ang) % 90.0, aspect
    bw, bh = box_wh(d)
    aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
    return (0.0 if bw >= bh else 90.0), aspect


def is_axis_aligned(angle: float, tol: float = 14.0) -> bool:
    return angle <= tol or angle >= (90.0 - tol)


def is_line_edge_false_positive(
    d: dict,
    dist_map: np.ndarray,
    shape: tuple[int, int, int],
    min_dist: float,
    card_roi: tuple[int, int, int, int] | None = None,
) -> bool:
    """贴外轮廓/底边横条 line 误检；斜向内部真线保留。"""
    min_d, mean_d = line_distances(d, dist_map)
    if min_d < 10:
        return True

    if center_edge_distance(d, dist_map) > min_dist * 1.2:
        return False

    conf = d["conf"]
    aspect = box_aspect(d)
    area = box_area(d)
    fill = poly_mask_area(d, shape) / max(area, 1.0)
    bw, bh = box_wh(d)
    h, w = shape[:2]
    _, cy = det_center(d)

    hugs_silhouette = min_d < 15 and mean_d < min_dist * 0.42
    band_edge_line = (
        min_d < min_dist
        and mean_d < min_dist * 0.55
        and aspect > 2.5
        and conf < 0.68
    )
    vertical_edge = min_d < 22 and mean_d < 30 and aspect > 2.8 and conf < 0.68

    bottom_band = False
    if bw > bh * 2.0 and min_d < 32 and mean_d < 48:
        if card_roi is not None:
            _, y1, _, y2 = card_roi
            bottom_band = cy > y2 - max(28, int((y2 - y1) * 0.06))
        else:
            bottom_band = cy > h * 0.80

    top_band = False
    if bw > bh * 2.0 and min_d < 32 and mean_d < 48:
        if card_roi is not None:
            _, y1, _, _ = card_roi
            top_band = cy < y1 + max(28, int((card_roi[3] - y1) * 0.06))
        else:
            top_band = cy < h * 0.12

    edge_corner_blob = min_d < 24 and area > 5000 and fill < 0.07 and conf < 0.75
    return hugs_silhouette or band_edge_line or vertical_edge or bottom_band or top_band or edge_corner_blob


def is_printed_line_false_positive(
    d: dict,
    max_conf: float = 0.72,
    min_aspect: float = 3.2,
    min_long_side: float = 55.0,
    angle_tol: float = 14.0,
) -> bool:
    """笔直横/竖印刷压纹线；斜向划痕保留。"""
    conf = d["conf"]
    if conf >= max_conf:
        return False
    angle, aspect = line_oriented_angle_aspect(d)
    if aspect < min_aspect or not is_axis_aligned(angle, angle_tol):
        return False
    if max(box_wh(d)) < min_long_side:
        return False
    return conf < max_conf


def filter_printed_lines(
    dets: list[dict],
    max_conf: float,
) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for d in dets:
        name = str(d["name"]).lower()
        if "line" not in name:
            kept.append(d)
            continue
        if is_printed_line_false_positive(d, max_conf=max_conf):
            dropped += 1
            continue
        kept.append(d)
    return kept, dropped


def filter_line_card_edge(
    dets: list[dict],
    dist_map: np.ndarray,
    shape: tuple[int, int, int],
    min_dist: float,
    card_roi: tuple[int, int, int, int] | None = None,
) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for d in dets:
        name = str(d["name"]).lower()
        if "line" not in name:
            kept.append(d)
            continue
        if is_line_edge_false_positive(d, dist_map, shape, min_dist, card_roi):
            dropped += 1
            continue
        kept.append(d)
    return kept, dropped


def main():
    p = argparse.ArgumentParser(description="Sliding-window inference per image with draw + count")
    p.add_argument("--model", default='H:/Python_cls/YOLO1111111/yolo/ultralytics-main/runs/segment/runs/seg-mixed/defect_seg/weights/best.pt', help="YOLO weights .pt")
    p.add_argument("--source", default='H:/卡游/压印testall2/骑行/线', help="folder of original images")
    p.add_argument("--out", default="G:/卡游/压印testall2/骑行2/线", help="output root")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.40, help="模型置信度阈值，漏检降到0.35，过检升到0.5")
    p.add_argument("--min-conf-point", type=float, default=0.0,
                   help="仅配合--bg-filter：灰底上的点低于此值才删，0=不按置信度删")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU across sliding windows")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--defect-preprocess", default="mixed", choices=["none", "point", "line", "mixed"])
    p.add_argument("--bg-filter", action="store_true", default=False,
                   help="可选：仅去掉落在外圈灰底上的检测（不画框，不伤卡片内部）")
    p.add_argument("--bg-tol", type=int, default=8, help="灰底洪泛填充容差，过大易误删")
    p.add_argument("--filter-point", action="store_true", default=True,
                   help="默认开：点缺陷面积/形状/灰底/平坦背景过滤")
    p.add_argument("--no-filter-point", dest="filter_point", action="store_false")
    p.add_argument("--point-min-area", type=float, default=18.0)
    p.add_argument("--point-max-area", type=float, default=1800.0)
    p.add_argument("--point-max-aspect", type=float, default=2.8)
    p.add_argument("--point-flat-conf", type=float, default=0.49)
    p.add_argument("--point-flat-std", type=float, default=9.0)
    p.add_argument("--filter-line-edge", action="store_true", default=True,
                   help="默认开：删贴外轮廓/底边横条 line")
    p.add_argument("--no-filter-line-edge", dest="filter_line_edge", action="store_false")
    p.add_argument("--line-edge-dist", type=float, default=35.0)
    p.add_argument("--filter-printed-line", action="store_true", default=True,
                   help="默认开：删笔直横竖印刷压纹线，保留斜线")
    p.add_argument("--no-filter-printed-line", dest="filter_printed_line", action="store_false")
    p.add_argument("--printed-line-max-conf", type=float, default=0.72)
    p.add_argument("--filter-line-blob", action="store_true", default=False,
                   help="默认关：仅去掉超大空框 line；真线测试建议保持关闭")
    p.add_argument("--no-filter-line-blob", dest="filter_line_blob", action="store_false")
    p.add_argument("--line-max-area", type=float, default=22000.0)
    p.add_argument("--line-min-fill", type=float, default=0.04)
    p.add_argument("--interior-shrink", type=int, default=0, help="已弃用，保留兼容")
    p.add_argument("--spatial-filter", action="store_true", help="额外面积/长宽比过滤")
    p.add_argument("--card-roi", action="store_true", help="spatial-filter 时用矩形 ROI（一般不用）")
    p.add_argument("--edge-margin", type=int, default=0)
    p.add_argument("--min-area", type=float, default=0.0)
    p.add_argument("--max-area", type=float, default=1e9)
    p.add_argument("--max-point-aspect", type=float, default=1e9)
    p.add_argument("--draw-roi", action="store_true", help="调试：画 ROI 框")
    args = p.parse_args()

    source = Path(args.source)
    out_root = Path(args.out)
    vis_dir = out_root / "vis"
    orig_dir = out_root / "orig_copy"
    vis_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"no images in {source}")

    summary_rows = []
    print(f"images={len(images)}, patch={args.patch}, stride={args.stride}, preprocess={args.defect_preprocess}")

    for img_path in images:
        img = imread_unicode(img_path)
        if img is None:
            print(f"[skip] read fail: {img_path.name}")
            continue

        t0 = time.time()
        raw_dets, n_win = sliding_predict_one(
            model, img, args.patch, args.stride, args.conf, args.imgsz,
            args.defect_preprocess, args.batch, names, device,
        )
        dets = nms_by_class(raw_dets, args.iou)

        n_filtered = 0
        card_rect = None
        gray = to_gray(img)
        outer_bg = build_outer_background_mask(gray, tol=args.bg_tol)

        if args.bg_filter:
            min_cp = args.min_conf_point if args.min_conf_point > 0 else 0.48
            dets, n_bg = filter_background_edge(
                dets, gray, outer_bg, min_cp, drop_outer_only=True,
            )
            n_filtered += n_bg

        if args.filter_point:
            dets, n_pt = filter_point_quality(
                dets, gray, outer_bg,
                args.point_min_area, args.point_max_area, args.point_max_aspect,
                args.point_flat_conf, args.point_flat_std,
            )
            n_filtered += n_pt

        if args.filter_line_blob:
            dets, n_blob = filter_line_quality(
                dets, img.shape, args.line_max_area, args.line_min_fill,
            )
            n_filtered += n_blob

        if args.filter_line_edge:
            card_rect = detect_card_roi(img)
            dist_map = build_card_edge_distance(gray, bg_tol=args.bg_tol)
            dets, n_line = filter_line_card_edge(
                dets, dist_map, img.shape, args.line_edge_dist, card_rect,
            )
            n_filtered += n_line

        if args.filter_printed_line:
            dets, n_printed = filter_printed_lines(dets, args.printed_line_max_conf)
            n_filtered += n_printed

        if args.spatial_filter:
            card_rect = detect_card_roi(img) if args.card_roi else None
            dets, n_sp = filter_detections(
                dets, img.shape, card_rect, args.edge_margin,
                args.min_area, args.max_area, args.max_point_aspect,
            )
            n_filtered += n_sp

        elapsed = time.time() - t0

        point_n, line_n, total_n = count_by_class(dets, names)
        vis = draw_detections(img, dets)
        if args.draw_roi and card_rect is not None:
            inner = shrink_rect(card_rect, args.edge_margin, img.shape[1], img.shape[0])
            vis = draw_roi_overlay(vis, card_rect, inner)
        vis = draw_summary_header(vis, point_n, line_n, total_n)

        stem = img_path.stem
        imwrite_unicode(vis_dir / f"{stem}_pred.png", vis)
        imwrite_unicode(orig_dir / f"{stem}_orig{img_path.suffix}", img)

        summary_rows.append({
            "filename": img_path.name,
            "width": img.shape[1],
            "height": img.shape[0],
            "windows": n_win,
            "raw_detections": len(raw_dets),
            "filtered": n_filtered,
            "after_nms": total_n,
            "point": point_n,
            "line": line_n,
            "time_sec": round(elapsed, 3),
        })
        print(f"[{img_path.name}] win={n_win} raw={len(raw_dets)} nms={total_n} "
              f"point={point_n} line={line_n} {elapsed:.2f}s")

    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    total_point = sum(r["point"] for r in summary_rows)
    total_line = sum(r["line"] for r in summary_rows)
    print(f"done: images={len(summary_rows)}, total_point={total_point}, total_line={total_line}")
    print(f"vis      -> {vis_dir}")
    print(f"summary  -> {summary_path}")


if __name__ == "__main__":
    main()
