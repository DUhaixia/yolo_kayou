"""
Defect-preprocess helpers for annotation (compare / optional offline label_view).

Recommended: annotate on ORIGINAL, use enhanced view only as reference (same pixels):
     python scripts/dual_view_annotate.py --dir D:/raw --mode mixed

Outputs LabelMe JSON on original images. Also openable in labelme on the same folder.

Optional side-by-side check:
     python scripts/preprocess_for_labelme.py --compare --src D:/raw --image foo.bmp --mode mixed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.defect_preprocess import apply_defect_preprocess

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix.lower() if path.suffix.lower() in IMAGE_SUFFIXES else ".png"
    path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_label_view(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = list_images(src)
    if not files:
        print(f"No images in {src}")
        return
    ok, fail = 0, 0
    for fp in files:
        out = dst / fp.name
        if out.exists() and not overwrite:
            ok += 1
            continue
        im = imread_unicode(fp)
        if im is None:
            fail += 1
            continue
        proc = apply_defect_preprocess(im, mode=mode)
        if imwrite_unicode(out, proc):
            ok += 1
        else:
            fail += 1
    print(f"label_view: {dst}")
    print(f"mode={mode}, written={ok}, failed={fail}, skipped_existing={len(files) - ok - fail}")


def sync_json(src: Path, dst: Path, direction: str) -> None:
    """
    Copy LabelMe JSON between folders. Polygon 'points' are not modified.

    direction=to_raw   : dst/*.json -> src/*.json  (after labeling on label_view)
    direction=to_view   : src/*.json -> dst/*.json  (resume labeling on existing raw labels)
    """
    if direction == "to_raw":
        from_dir, to_dir = dst, src
    else:
        from_dir, to_dir = src, dst

    copied = 0
    for jp in sorted(from_dir.glob("*.json")):
        stem = jp.stem
        if not any((from_dir / f"{stem}{s}").exists() for s in IMAGE_SUFFIXES):
            continue
        target = to_dir / jp.name
        shutil.copy2(jp, target)
        # Keep imagePath consistent with destination image filename.
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            img_name = data.get("imagePath") or f"{stem}.jpg"
            # Use same stem; extension follows whatever image exists in to_dir.
            for ext in IMAGE_SUFFIXES:
                if (to_dir / f"{stem}{ext}").exists():
                    data["imagePath"] = f"{stem}{ext}"
                    break
            else:
                data["imagePath"] = img_name
            data["imageHeight"] = data.get("imageHeight")
            data["imageWidth"] = data.get("imageWidth")
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        copied += 1
    print(f"sync_json {direction}: {copied} files from {from_dir} -> {to_dir}")


def draw_shapes(im: np.ndarray, shapes: list[dict]) -> np.ndarray:
    out = im.copy()
    for s in shapes:
        pts = np.array(s.get("points", []), dtype=np.int32)
        if pts.shape[0] < 2:
            continue
        closed = pts.shape[0] >= 3
        cv2.polylines(out, [pts], closed, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def compare_one(src: Path, image_name: str, mode: str, out_dir: Path | None) -> None:
    fp = src / image_name
    if not fp.exists():
        raise FileNotFoundError(fp)
    im = imread_unicode(fp)
    if im is None:
        raise RuntimeError(f"Cannot read {fp}")
    proc = apply_defect_preprocess(im, mode=mode)
    h, w = im.shape[:2]
    assert proc.shape[:2] == (h, w), "preprocess must not change image size"

    jp = fp.with_suffix(".json")
    shapes = []
    if jp.exists():
        try:
            ann = json.loads(jp.read_text(encoding="utf-8"))
            shapes = ann.get("shapes", [])
        except Exception:
            pass

    orig_o = draw_shapes(im, shapes) if shapes else im
    proc_o = draw_shapes(proc, shapes) if shapes else proc

    bar_h = 36
    gap = 8
    canvas_w = w * 2 + gap
    canvas = np.full((h + bar_h, canvas_w, 3), 32, dtype=np.uint8)
    canvas[bar_h : bar_h + h, 0:w] = orig_o
    canvas[bar_h : bar_h + h, w + gap : w + gap + w] = proc_o
    cv2.putText(canvas, "original", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"preprocess ({mode})", (w + gap + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fp.stem}_compare_{mode}.jpg"
        cv2.imencode(".jpg", canvas)[1].tofile(str(out_path))
        print(f"saved {out_path}")
    else:
        cv2.imshow("original | preprocess (press key)", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defect preprocess for LabelMe + JSON sync")
    p.add_argument("--src", type=Path, required=True, help="Original images (and optional JSON)")
    p.add_argument("--dst", type=Path, default=None, help="Label-view preprocessed images folder")
    p.add_argument("--mode", type=str, default="mixed", choices=["none", "point", "line", "mixed"])
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing label-view images")
    p.add_argument("--sync-json", action="store_true", help="Copy JSON between src and dst")
    p.add_argument(
        "--sync-direction",
        type=str,
        default="to_raw",
        choices=["to_raw", "to_view"],
        help="to_raw: label on dst, copy JSON to src; to_view: copy existing src JSON to dst",
    )
    p.add_argument("--compare", action="store_true", help="Show/save side-by-side original vs preprocess")
    p.add_argument("--image", type=str, default=None, help="Image filename for --compare")
    p.add_argument("--compare-out", type=Path, default=None, help="Save compare image here instead of imshow")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()

    if args.compare:
        if not args.image:
            imgs = list_images(src)
            if not imgs:
                raise SystemExit(f"No images in {src}")
            args.image = imgs[0].name
        compare_one(src, args.image, args.mode, args.compare_out)
        return

    if args.sync_json:
        if not args.dst:
            raise SystemExit("--dst required for --sync-json")
        sync_json(src, args.dst.expanduser().resolve(), args.sync_direction)
        return

    if not args.dst:
        raise SystemExit("Use --dst to build label-view, or --sync-json / --compare")
    build_label_view(src, args.dst.expanduser().resolve(), args.mode, args.overwrite)


if __name__ == "__main__":
    main()
