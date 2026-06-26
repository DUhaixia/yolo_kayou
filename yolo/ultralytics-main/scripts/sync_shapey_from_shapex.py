"""
从 ShapeX 的 LabelMe 标注同步到配对 ShapeY：
  - ShapeY 已有 shapes → 保持不动
  - ShapeX 无 shapes → 不生成 ShapeY 标注
  - 否则将 ShapeX 的 shapes（同类 id/位置）复制到 ShapeY json

示例:
  python scripts/sync_shapey_from_shapex.py ^
    --dir "G:/卡游切图/汇总_XY/AI_点压印/BIAOZHU"
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] bad json {path.name}: {e}")
        return None


def shape_count(data: dict | None) -> int:
    if not data:
        return 0
    return len(data.get("shapes", []))


def image_hw(bmp: Path, fallback_json: dict | None = None) -> tuple[int, int] | None:
    if bmp.is_file():
        im = cv2.imread(str(bmp), cv2.IMREAD_UNCHANGED)
        if im is not None:
            return im.shape[0], im.shape[1]
    if fallback_json:
        h, w = fallback_json.get("imageHeight"), fallback_json.get("imageWidth")
        if h and w:
            return int(h), int(w)
    return None


def save_shapey_json(
    path: Path,
    shapes: list[dict],
    bmp: Path,
    template: dict,
    hw: tuple[int, int] | None,
) -> None:
    h, w = hw or (template.get("imageHeight"), template.get("imageWidth"))
    if not h or not w:
        raise ValueError(f"cannot determine image size for {bmp.name}")
    payload = {
        "version": template.get("version", "5.1.1"),
        "flags": copy.deepcopy(template.get("flags", {})),
        "shapes": shapes,
        "imagePath": bmp.name,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_dir(root: Path, dry_run: bool = False) -> dict[str, int]:
    stats = {"copied": 0, "kept_y": 0, "skip_no_x": 0, "skip_no_y_img": 0, "err": 0}
    shapex_jsons = sorted(root.glob("*ShapeX.json"))

    for x_json in shapex_jsons:
        y_json = Path(str(x_json).replace("ShapeX", "ShapeY"))
        y_bmp = Path(str(x_json).replace("ShapeX.json", "ShapeY.bmp"))

        x_data = load_json(x_json)
        if x_data is None:
            stats["err"] += 1
            continue
        if shape_count(x_data) == 0:
            stats["skip_no_x"] += 1
            continue

        y_data = load_json(y_json)
        if shape_count(y_data) > 0:
            stats["kept_y"] += 1
            continue

        if not y_bmp.is_file():
            stats["skip_no_y_img"] += 1
            print(f"[warn] missing ShapeY image: {y_bmp.name}")
            continue

        shapes = copy.deepcopy(x_data.get("shapes", []))
        x_bmp = Path(str(x_json).replace("ShapeX.json", "ShapeX.bmp"))
        hw = image_hw(y_bmp, x_data) or image_hw(x_bmp, x_data)
        if y_data:
            template = y_data
        else:
            template = x_data

        if dry_run:
            stats["copied"] += 1
            continue

        try:
            save_shapey_json(y_json, shapes, y_bmp, template, hw)
            stats["copied"] += 1
        except Exception as e:
            stats["err"] += 1
            print(f"[err] {y_json.name}: {e}")

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="ShapeX 标注同步到 ShapeY")
    p.add_argument("--dir", type=Path, required=True, help="BIAOZHU 目录")
    p.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = p.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    stats = sync_dir(root, dry_run=args.dry_run)
    print(f"dir: {root}")
    print(
        f"copied={stats['copied']}, kept_shapey={stats['kept_y']}, "
        f"skip_shapex_empty={stats['skip_no_x']}, skip_no_y_bmp={stats['skip_no_y_img']}, err={stats['err']}"
    )


if __name__ == "__main__":
    main()
