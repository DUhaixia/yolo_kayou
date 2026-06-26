"""从数据集中筛出无标注 txt 的好品图（及空 txt），移到单独目录。"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def filter_split(
    root: Path,
    out_root: Path,
    split: str,
    include_empty_txt: bool,
    move: bool,
) -> list[dict]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    dst_img = out_root / "images" / split
    dst_img.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    op = shutil.move if move else shutil.copy2

    for img in sorted(img_dir.iterdir()):
        if not is_image(img):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        reason = ""
        if not lbl.is_file():
            reason = "no_txt"
        elif include_empty_txt and not lbl.read_text(encoding="utf-8", errors="ignore").strip():
            reason = "empty_txt"
        else:
            continue

        dst = dst_img / img.name
        op(img, dst)
        npy = img.with_suffix(".npy")
        if npy.is_file():
            (shutil.move if move else shutil.copy2)(npy, dst.with_suffix(".npy"))
        if lbl.is_file() and reason == "empty_txt":
            if move:
                lbl.unlink()
            else:
                shutil.copy2(lbl, out_root / "labels" / split / lbl.name)

        rows.append({"split": split, "file": img.name, "reason": reason})

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="筛出无 txt 的好品图")
    p.add_argument("--root", type=Path, required=True, help="数据集根目录")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="好品输出目录，默认 <root>/good_no_label",
    )
    p.add_argument("--include-empty-txt", action="store_true", default=True)
    p.add_argument("--no-include-empty-txt", dest="include_empty_txt", action="store_false")
    p.add_argument("--move", action="store_true", default=True, help="移动（默认）；不加则复制")
    p.add_argument("--copy", dest="move", action="store_false")
    args = p.parse_args()

    root = args.root.resolve()
    out_root = (args.out or root / "good_no_label").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split in ("train", "val"):
        rows.extend(filter_split(root, out_root, split, args.include_empty_txt, args.move))

    manifest = out_root / "good_no_label_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "file", "reason"])
        w.writeheader()
        w.writerows(rows)

    tr = sum(1 for r in rows if r["split"] == "train")
    va = sum(1 for r in rows if r["split"] == "val")
    action = "moved" if args.move else "copied"
    print(f"done -> {out_root}")
    print(f"{action}: train={tr}, val={va}, total={len(rows)}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
