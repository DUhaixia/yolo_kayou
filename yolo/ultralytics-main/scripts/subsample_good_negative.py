"""缩减数据集中的好品负样本（空 txt），按缺陷样本比例保留子集。"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_good_images(img_dir: Path, lbl_dir: Path) -> list[Path]:
    good: list[Path] = []
    for img in sorted(img_dir.iterdir()):
        if not is_image(img):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.is_file() and not lbl.read_text(encoding="utf-8", errors="ignore").strip():
            good.append(img)
    return good


def list_defect_images(img_dir: Path, lbl_dir: Path) -> list[Path]:
    defect: list[Path] = []
    for img in sorted(img_dir.iterdir()):
        if not is_image(img):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.is_file() and lbl.read_text(encoding="utf-8", errors="ignore").strip():
            defect.append(img)
    return defect


def restore_from_excess(root: Path, removed_root: Path, split: str) -> int:
    src_img = removed_root / "images" / split
    src_lbl = removed_root / "labels" / split
    dst_img = root / "images" / split
    dst_lbl = root / "labels" / split
    if not src_img.is_dir():
        return 0
    n = 0
    for img in sorted(src_img.iterdir()):
        if not is_image(img):
            continue
        lbl = src_lbl / f"{img.stem}.txt"
        shutil.move(img, dst_img / img.name)
        if lbl.is_file():
            shutil.move(lbl, dst_lbl / lbl.name)
        else:
            (dst_lbl / f"{img.stem}.txt").write_text("", encoding="utf-8")
        npy = src_img / f"{img.stem}.npy"
        if npy.is_file():
            shutil.move(npy, (dst_img / img.name).with_suffix(".npy"))
        n += 1
    return n


def subsample_split(
    root: Path,
    removed_root: Path,
    split: str,
    ratio: float,
    max_good: int,
    seed: int,
    restore_excess: bool,
) -> list[dict]:
    if restore_excess:
        restored = restore_from_excess(root, removed_root, split)
        if restored:
            print(f"[{split}] restored_from_excess={restored}")
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    dst_img = removed_root / "images" / split
    dst_lbl = removed_root / "labels" / split

    good = list_good_images(img_dir, lbl_dir)
    defect_n = len(list_defect_images(img_dir, lbl_dir))
    if not good:
        print(f"[{split}] no good samples")
        return []

    target = int(round(defect_n * ratio))
    if max_good > 0:
        target = min(target, max_good)
    target = max(0, min(len(good), target))

    rng = random.Random(seed + (0 if split == "train" else 1))
    rng.shuffle(good)
    keep = set(p.stem for p in good[:target])
    removed_rows: list[dict] = []

    for img in good:
        if img.stem in keep:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        shutil.move(img, dst_img / img.name)
        if lbl.is_file():
            shutil.move(lbl, dst_lbl / lbl.name)
        npy = img.with_suffix(".npy")
        if npy.is_file():
            shutil.move(npy, (dst_img / img.name).with_suffix(".npy"))
        removed_rows.append({"split": split, "file": img.name, "action": "removed"})

    print(
        f"[{split}] defect={defect_n}, good_before={len(good)}, "
        f"good_keep={target}, good_removed={len(removed_rows)}"
    )
    return removed_rows


def main() -> None:
    p = argparse.ArgumentParser(description="按缺陷比例缩减好品负样本")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--removed-dir",
        type=Path,
        default=None,
        help="移出的好品目录，默认 <root>/good_excess",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.05,
        help="默认 train/val 共用比例",
    )
    p.add_argument("--train-ratio", type=float, default=None, help="train 好品比例，默认用 --ratio")
    p.add_argument("--val-ratio", type=float, default=None, help="val 好品比例，可设更高")
    p.add_argument("--splits", default="train,val", help="处理的划分，如 val")
    p.add_argument("--restore-excess", action="store_true", help="子采样前先把 excess 中的样本移回")
    p.add_argument("--max-good", type=int, default=0, help="每个 split 好品上限，0=不限制")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = args.root.resolve()
    removed_root = (args.removed_dir or root / "good_excess").resolve()
    removed_root.mkdir(parents=True, exist_ok=True)

    split_ratios = {
        "train": args.train_ratio if args.train_ratio is not None else args.ratio,
        "val": args.val_ratio if args.val_ratio is not None else args.ratio,
    }
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    rows: list[dict] = []
    for split in splits:
        rows.extend(
            subsample_split(
                root, removed_root, split, split_ratios[split],
                args.max_good, args.seed, args.restore_excess,
            )
        )

    manifest = removed_root / "removed_good_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "file", "action"])
        w.writeheader()
        w.writerows(rows)

    for cache in (root / "labels").glob("*.cache"):
        cache.unlink()
        print(f"removed cache: {cache.name}")

    print(f"done -> kept in {root}, removed -> {removed_root}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
