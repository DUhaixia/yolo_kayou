"""
大批量推理后，自动筛选「边缘线误检」并裁 512 patch，生成空 label 硬负样本。

流程:
  原图滑窗推理 → 识别 edge line 误检 → 以检测中心裁 512 → 保存 YOLO 格式

输出:
  out/images/train/*.png
  out/labels/train/*.txt   (空)
  out/review/vis/*_fp.png  (可选，标出误检框)
  out/manifest.csv

示例:
  cd H:/Python_cls/YOLO1111111/yolo/ultralytics-main

  python scripts/harvest_line_edge_fp_patches.py ^
    --model runs/segment/runs/seg-mixed/defect_seg/weights/best.pt ^
    --source "G:/卡游/压印testall2/好品大批量" ^
    --out "M:/压印 - 副本/line_edge_fp_patches" ^
    --patch 512 --stride 256 --crop 512 ^
    --conf 0.35 --max-per-image 8
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_draw_module():
    spec = importlib.util.spec_from_file_location("infer_sliding_draw", SCRIPT_DIR / "infer_sliding_draw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRAW = _load_draw_module()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_stem(name: str, max_len: int = 80) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", name)
    s = re.sub(r"\s+", "_", s.strip())
    s = s.strip("._") or "img"
    return s[:max_len]


def crop_origin(cx: float, cy: float, patch: int, img_w: int, img_h: int) -> tuple[int, int]:
    x0 = int(round(cx - patch / 2))
    y0 = int(round(cy - patch / 2))
    x0 = max(0, min(x0, max(img_w - patch, 0)))
    y0 = max(0, min(y0, max(img_h - patch, 0)))
    return y0, x0


def patch_key(y0: int, x0: int, step: int = 64) -> tuple[int, int]:
    return y0 // step, x0 // step


def classify_line_fp(
    d: dict,
    gray: np.ndarray,
    shape: tuple[int, int, int],
    card_roi: tuple[int, int, int, int] | None,
    dist_map: np.ndarray,
    line_edge_dist: float,
    include_printed: bool,
    printed_max_conf: float,
) -> tuple[bool, str]:
    name = str(d.get("name", "")).lower()
    if "line" not in name:
        return False, ""

    if DRAW.is_line_edge_false_positive(d, dist_map, shape, line_edge_dist, card_roi):
        min_d, mean_d = DRAW.line_distances(d, dist_map)
        if min_d < 10:
            return True, "hug_edge"
        bw, bh = DRAW.box_wh(d)
        _, cy = DRAW.det_center(d)
        if bw > bh * 2.0:
            if card_roi is not None:
                _, y1, _, y2 = card_roi
                if cy > y2 - max(28, int((y2 - y1) * 0.06)):
                    return True, "bottom_band"
                if cy < y1 + max(28, int((y2 - y1) * 0.06)):
                    return True, "top_band"
            else:
                h = shape[0]
                if cy > h * 0.80:
                    return True, "bottom_band"
                if cy < h * 0.12:
                    return True, "top_band"
        return True, "edge_line"

    if include_printed and DRAW.is_printed_line_false_positive(d, max_conf=printed_max_conf):
        return True, "printed_line"

    return False, ""


def draw_fp_review(img: np.ndarray, fps: list[dict], shape: tuple[int, int, int]) -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for d in fps:
        x1, y1, x2, y2 = d["xyxy"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        label = f"fp_{d.get('fp_reason', 'edge')} {d['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cx, cy = DRAW.det_center(d)
        y0, x0 = crop_origin(cx, cy, d["crop_patch"], shape[1], shape[0])
        cv2.rectangle(vis, (x0, y0), (x0 + d["crop_patch"], y0 + d["crop_patch"]), (255, 128, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"edge_fp={len(fps)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest 512 patches from edge line false positives")
    p.add_argument("--model", required=True, help="YOLO seg weights")
    p.add_argument("--source", required=True, help="原图目录，大批量推理输入")
    p.add_argument("--out", required=True, help="输出数据集根目录")
    p.add_argument("--patch", type=int, default=512, help="滑窗 patch")
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--crop", type=int, default=512, help="误检中心裁切尺寸")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--conf-line", type=float, default=-1.0)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--defect-preprocess", default="none", choices=["none", "point", "line", "mixed"])
    p.add_argument("--line-edge-dist", type=float, default=35.0)
    p.add_argument("--bg-tol", type=int, default=8)
    p.add_argument("--include-printed", action="store_true", help="除边缘外，也收笔直印刷压纹线误检")
    p.add_argument("--printed-line-max-conf", type=float, default=0.72)
    p.add_argument("--max-per-image", type=int, default=8, help="每张原图最多保存几个 patch")
    p.add_argument("--dedup-step", type=int, default=64, help="裁切位置去重步长(px)")
    p.add_argument("--save-vis", action="store_true", default=True, help="保存 review 可视化")
    p.add_argument("--no-save-vis", dest="save_vis", action="store_false")
    p.add_argument("--split", default="train", choices=["train", "val"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    img_dir = out_root / "images" / args.split
    lbl_dir = out_root / "labels" / args.split
    review_dir = out_root / "review" / "vis"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        review_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    images = DRAW.list_images(source)
    if not images:
        raise FileNotFoundError(f"no images in {source}")

    conf_line = args.conf if args.conf_line < 0 else args.conf_line
    model_conf = min(args.conf, conf_line)

    manifest_rows: list[dict] = []
    total_fp = 0
    total_saved = 0

    print(f"[harvest] images={len(images)}, model_conf={model_conf}, crop={args.crop}, "
          f"edge_only={not args.include_printed}, max_per_image={args.max_per_image}")

    for img_path in images:
        img = DRAW.imread_unicode(img_path)
        if img is None:
            print(f"[skip] read fail: {img_path.name}")
            continue

        t0 = time.time()
        raw_dets, n_win = DRAW.sliding_predict_one(
            model, img, args.patch, args.stride, model_conf, args.imgsz,
            args.defect_preprocess, args.batch, names, device,
        )
        dets = DRAW.nms_by_class(raw_dets, args.iou)

        gray = DRAW.to_gray(img)
        card_roi = DRAW.detect_card_roi(img)
        dist_map = DRAW.build_card_edge_distance(gray, bg_tol=args.bg_tol)

        fps: list[dict] = []
        for d in dets:
            if "line" not in str(d["name"]).lower():
                continue
            if d["conf"] < conf_line:
                continue
            is_fp, reason = classify_line_fp(
                d, gray, img.shape, card_roi, dist_map,
                args.line_edge_dist, args.include_printed, args.printed_line_max_conf,
            )
            if not is_fp:
                continue
            min_d, mean_d = DRAW.line_distances(d, dist_map)
            cx, cy = DRAW.det_center(d)
            item = dict(d)
            item["fp_reason"] = reason
            item["min_edge_dist"] = round(min_d, 1)
            item["mean_edge_dist"] = round(mean_d, 1)
            item["center_x"] = round(cx, 1)
            item["center_y"] = round(cy, 1)
            item["crop_patch"] = args.crop
            fps.append(item)

        fps.sort(key=lambda x: (-x["conf"], x["min_edge_dist"]))
        used_keys: set[tuple[int, int]] = set()
        saved_this = 0
        stem = safe_stem(img_path.stem)

        for i, d in enumerate(fps):
            if saved_this >= args.max_per_image:
                break
            cx, cy = d["center_x"], d["center_y"]
            y0, x0 = crop_origin(cx, cy, args.crop, img.shape[1], img.shape[0])
            key = patch_key(y0, x0, args.dedup_step)
            if key in used_keys:
                continue
            used_keys.add(key)

            out_name = f"fpedge_{stem}_{saved_this:02d}_y{y0}_x{x0}.png"
            out_img = img[y0:y0 + args.crop, x0:x0 + args.crop]
            if out_img.shape[0] != args.crop or out_img.shape[1] != args.crop:
                continue

            out_img_path = img_dir / out_name
            out_lbl_path = lbl_dir / f"{out_img_path.stem}.txt"
            DRAW.imwrite_unicode(out_img_path, out_img)
            out_lbl_path.write_text("", encoding="utf-8")

            manifest_rows.append({
                "file": out_name,
                "source_image": img_path.name,
                "fp_reason": d["fp_reason"],
                "conf": round(d["conf"], 4),
                "min_edge_dist": d["min_edge_dist"],
                "mean_edge_dist": d["mean_edge_dist"],
                "center_x": d["center_x"],
                "center_y": d["center_y"],
                "crop_x": x0,
                "crop_y": y0,
                "crop_size": args.crop,
            })
            saved_this += 1
            total_saved += 1

        total_fp += len(fps)
        elapsed = time.time() - t0

        if args.save_vis and fps:
            vis = draw_fp_review(img, fps, img.shape)
            DRAW.imwrite_unicode(review_dir / f"{stem}_fp.png", vis)

        if fps or saved_this:
            print(f"[{img_path.name}] win={n_win} line_fp={len(fps)} saved={saved_this} {elapsed:.2f}s")

    manifest_path = out_root / "manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {str(out_root).replace(chr(92), '/')}\n"
        f"train: images/{args.split}\n"
        f"val: images/{args.split}\n"
        f"names:\n  0: point\n  1: line\n",
        encoding="utf-8",
    )

    print("=== done ===")
    print(f"images scanned : {len(images)}")
    print(f"line fp found  : {total_fp}")
    print(f"patches saved  : {total_saved}")
    print(f"images dir     : {img_dir}")
    print(f"labels dir     : {lbl_dir}")
    if args.save_vis:
        print(f"review vis     : {review_dir}")
    print(f"manifest       : {manifest_path}")


if __name__ == "__main__":
    main()
