"""为 line_only 数据集图片与标签同步重命名，避免合并时文件名冲突。"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(r"M:\压印 - 副本\dataSet-原始-切割-已筛选-line_only - 副本")
PREFIX = "lineonly_"
SPLITS = ("train", "val")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def rename_pair(img_path: Path, lbl_path: Path, new_stem: str) -> tuple[str, str]:
    new_img = img_path.parent / f"{new_stem}{img_path.suffix}"
    new_lbl = lbl_path.parent / f"{new_stem}.txt"
    img_path.rename(new_img)
    lbl_path.rename(new_lbl)
    return new_img.name, new_lbl.name


def update_csv(csv_path: Path, old_to_new: dict[str, str]) -> None:
    if not csv_path.exists():
        return

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            for key in ("file",):
                if key in row and row[key] in old_to_new:
                    row[key] = old_to_new[row[key]]
            rows.append(row)

    backup = csv_path.with_suffix(csv_path.suffix + ".bak")
    shutil.copy2(csv_path, backup)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rename_map: list[tuple[str, str, str, str]] = []
    old_to_new_file: dict[str, str] = {}
    renamed = 0
    skipped = 0

    for split in SPLITS:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split

        stems = sorted(
            p.stem
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )

        for stem in stems:
            img_path = find_image(img_dir, stem)
            lbl_path = lbl_dir / f"{stem}.txt"

            if img_path is None:
                print(f"[WARN] missing image: {split}/{stem}")
                skipped += 1
                continue
            if not lbl_path.exists():
                print(f"[WARN] missing label: {split}/{stem}")
                skipped += 1
                continue

            if stem.startswith(PREFIX):
                skipped += 1
                continue

            new_stem = f"{PREFIX}{stem}"
            new_img_name, new_lbl_name = rename_pair(img_path, lbl_path, new_stem)
            rename_map.append((split, img_path.name, new_img_name, new_lbl_name))
            old_to_new_file[img_path.name] = new_img_name
            renamed += 1

    map_path = ROOT / "rename_map.csv"
    with map_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "old_image", "new_image", "new_label"])
        writer.writerows(rename_map)

    update_csv(ROOT / "line_only_manifest.csv", old_to_new_file)

    # verify
    missing = 0
    for split in SPLITS:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        for p in img_dir.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if not (lbl_dir / f"{p.stem}.txt").exists():
                missing += 1

    print(f"root: {ROOT}")
    print(f"prefix: {PREFIX}")
    print(f"renamed pairs: {renamed}")
    print(f"skipped: {skipped}")
    print(f"missing pairs after rename: {missing}")
    print(f"rename_map: {map_path}")


if __name__ == "__main__":
    main()
