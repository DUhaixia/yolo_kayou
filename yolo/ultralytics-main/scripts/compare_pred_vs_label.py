"""Build side-by-side comparison images: Original | GT annotation | Prediction.

For every ``<stem>_pred.bmp`` found under the split-inference root (organised in
defect sub-folders), this finds the matching ``<stem>_orig.bmp`` (original used
for inference) and the LabelMe ``<stem>.json`` in the annotation folder, then
renders a 3-panel triptych so prediction vs. ground-truth differences are easy
to eyeball.

Panels:
    1. Original image
    2. Ground-truth annotation (polygons drawn from the JSON)
    3. Model prediction (the baked ``_pred.bmp`` visualisation)

Usage (PowerShell):
    python scripts/compare_pred_vs_label.py \
        --infer-root "H:/卡游/runs/split_infer" \
        --ann-dir   "M:/压印 - 副本/明山二次优化/Images/压印线干扰" \
        --dst       "H:/卡游/runs/compare_gt_pred"

    # quick preview of a few images per subset:
    python scripts/compare_pred_vs_label.py ... --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# label -> RGB colour for the ground-truth overlay
LABEL_COLORS = {
    "YaYinLine": (255, 60, 60),     # 压印线 - red
    "YaYinPoint": (60, 160, 255),   # 压印点 - blue
}
DEFAULT_COLOR = (0, 220, 0)

PANEL_TITLES = ("原始图 Original", "标注 GT", "预测 Prediction")

PRED_SUFFIX = "_pred.bmp"
ORIG_SUFFIX = "_orig.bmp"


def load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_annotations(base: Image.Image, shapes: list, scale_x: float, scale_y: float,
                     font) -> Image.Image:
    """Return a copy of *base* with polygon annotations drawn on it."""
    img = base.convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    line_w = max(2, round(img.width / 500))

    for s in shapes:
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        color = LABEL_COLORS.get(s.get("label"), DEFAULT_COLOR)
        poly = [(p[0] * scale_x, p[1] * scale_y) for p in pts]
        stype = s.get("shape_type", "polygon")
        if stype in ("polygon", "linestrip") and len(poly) >= 3:
            od.polygon(poly, fill=color + (70,), outline=color + (255,), width=line_w)
        elif stype == "rectangle" and len(poly) >= 2:
            (x0, y0), (x1, y1) = poly[0], poly[1]
            od.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                         fill=color + (70,), outline=color + (255,), width=line_w)
        else:
            od.line(poly + [poly[0]], fill=color + (255,), width=line_w)
        # label text near the first point
        lx, ly = poly[0]
        od.text((lx + 3, ly - 18), str(s.get("label", "")), fill=color + (255,), font=font)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


def resize_h(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    w = round(img.width * target_h / img.height)
    return img.resize((w, target_h), Image.LANCZOS)


def compose(orig: Image.Image, gt: Image.Image, pred: Image.Image,
            stem: str, subset: str, gt_summary: str,
            panel_h: int, title_font, head_font, gap: int = 12) -> Image.Image:
    panels = [resize_h(im, panel_h) for im in (orig, gt, pred)]
    title_band = 34
    head_band = 30
    total_w = sum(p.width for p in panels) + gap * (len(panels) + 1)
    total_h = head_band + title_band + panel_h + gap * 2

    canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    # top header: file name + subset + GT summary
    d.text((gap, 6), f"[{subset}] {stem}    GT: {gt_summary}", fill=(0, 0, 0), font=head_font)

    x = gap
    y_title = head_band
    y_img = head_band + title_band
    for title, p in zip(PANEL_TITLES, panels):
        d.text((x + 4, y_title + 4), title, fill=(20, 20, 20), font=title_font)
        canvas.paste(p, (x, y_img))
        d.rectangle([x, y_img, x + p.width - 1, y_img + p.height - 1], outline=(180, 180, 180))
        x += p.width + gap
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="only process these sub-folders (default: all with predictions)")
    ap.add_argument("--panel-h", type=int, default=900, help="height of each panel in px")
    ap.add_argument("--limit", type=int, default=0, help="max images per subset (0 = all)")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    infer_root = Path(args.infer_root)
    ann_dir = Path(args.ann_dir)
    dst = Path(args.dst)
    if not infer_root.is_dir():
        sys.exit(f"ERROR: infer-root not found: {infer_root}")
    if not ann_dir.is_dir():
        sys.exit(f"ERROR: ann-dir not found: {ann_dir}")
    dst.mkdir(parents=True, exist_ok=True)

    title_font = load_font(22)
    head_font = load_font(20)
    ann_font = load_font(20)

    subsets = args.subsets or [d.name for d in sorted(infer_root.iterdir()) if d.is_dir()]

    total_ok = total_miss_orig = total_miss_json = 0
    t0 = time.time()
    for subset in subsets:
        sdir = infer_root / subset
        if not sdir.is_dir():
            print(f"-- skip (not a folder): {subset}")
            continue
        preds = sorted(p for p in sdir.iterdir() if p.name.endswith(PRED_SUFFIX))
        if not preds:
            print(f"-- {subset}: no *_pred.bmp, skipping")
            continue
        out_sub = dst / subset
        out_sub.mkdir(parents=True, exist_ok=True)

        done = 0
        print(f"== {subset}: {len(preds)} prediction(s)")
        for pred_path in preds:
            stem = pred_path.name[: -len(PRED_SUFFIX)]
            out_path = out_sub / f"{stem}.jpg"
            if out_path.exists() and not args.overwrite:
                done += 1
                if args.limit and done >= args.limit:
                    break
                continue

            orig_path = sdir / f"{stem}{ORIG_SUFFIX}"
            json_path = ann_dir / f"{stem}.json"
            if not orig_path.exists():
                print(f"   ! missing orig: {orig_path.name}")
                total_miss_orig += 1
                continue
            if not json_path.exists():
                print(f"   ! missing json: {json_path.name}")
                total_miss_json += 1
                continue

            try:
                orig = Image.open(orig_path).convert("RGB")
                pred = Image.open(pred_path).convert("RGB")
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                print(f"   ! read error {stem}: {e}")
                continue

            shapes = data.get("shapes", [])
            jw = data.get("imageWidth") or orig.width
            jh = data.get("imageHeight") or orig.height
            sx = orig.width / jw
            sy = orig.height / jh

            gt_img = draw_annotations(orig, shapes, sx, sy, ann_font)
            # summarise GT labels
            counts: dict[str, int] = {}
            for s in shapes:
                counts[s.get("label", "?")] = counts.get(s.get("label", "?"), 0) + 1
            gt_summary = ", ".join(f"{k}×{v}" for k, v in counts.items()) or "none"

            canvas = compose(orig, gt_img, pred, stem, subset, gt_summary,
                             args.panel_h, title_font, head_font)
            canvas.save(out_path, quality=args.quality)
            total_ok += 1
            done += 1
            if total_ok % 50 == 0:
                print(f"   ... {total_ok} done ({time.time()-t0:.0f}s)")
            if args.limit and done >= args.limit:
                break

    print("=" * 60)
    print(f"DONE. saved={total_ok}  missing_orig={total_miss_orig}  "
          f"missing_json={total_miss_json}  time={time.time()-t0:.0f}s")
    print("Output:", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
