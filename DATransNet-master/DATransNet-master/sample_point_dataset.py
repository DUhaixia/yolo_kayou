"""从 point 子集随机抽取 1000 张，8:2 划分 train/val，同步复制并重命名图片与标签。"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

SRC_ROOT = Path(r"M:\压印 - 副本\dataSet-原始-切割-已筛选+neg_细分\point")
DST_ROOT = Path(r"M:\压印 - 副本\dataSet-原始-切割-已筛选+neg_细分\point_sample1000_82")
PREFIX = "pt1000"
SAMPLE_SIZE = 1000
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def make_new_stem(target_split: str, index: int) -> str:
    return f"{PREFIX}_{target_split}_{index:06d}"

SPLITS = ("train", "val")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_pairs() -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for split in SPLITS:
        img_dir = SRC_ROOT / "images" / split
        lbl_dir = SRC_ROOT / "labels" / split
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                raise FileNotFoundError(f"missing label: {lbl_path}")
            pairs.append(
                {
                    "source_split": split,
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "old_stem": img_path.stem,
                    "old_image": img_path.name,
                }
            )
    return pairs


def main() -> None:
    all_pairs = collect_pairs()
    if len(all_pairs) < SAMPLE_SIZE:
        raise ValueError(f"not enough samples: {len(all_pairs)} < {SAMPLE_SIZE}")

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(all_pairs, SAMPLE_SIZE)
    train_count = int(SAMPLE_SIZE * TRAIN_RATIO)
    val_count = SAMPLE_SIZE - train_count
    train_items = sampled[:train_count]
    val_items = sampled[train_count:]

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    manifest_rows: list[list[str]] = []
    used_stems: set[str] = set()

    for target_split, items in (("train", train_items), ("val", val_items)):
        dst_img_dir = DST_ROOT / "images" / target_split
        dst_lbl_dir = DST_ROOT / "labels" / target_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(items, start=1):
            img_path: Path = item["img_path"]  # type: ignore[assignment]
            lbl_path: Path = item["lbl_path"]  # type: ignore[assignment]

            new_stem = make_new_stem(target_split, idx)
            if new_stem in used_stems:
                raise ValueError(f"duplicate new stem: {new_stem}")
            used_stems.add(new_stem)

            new_img = dst_img_dir / f"{new_stem}{img_path.suffix.lower()}"
            new_lbl = dst_lbl_dir / f"{new_stem}.txt"

            shutil.copy2(img_path, new_img)
            shutil.copy2(lbl_path, new_lbl)

            manifest_rows.append(
                [
                    target_split,
                    str(item["source_split"]),
                    str(item["old_image"]),
                    str(item["old_stem"]),
                    new_stem,
                    new_img.name,
                    new_lbl.name,
                ]
            )

    manifest_path = DST_ROOT / "sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["split", "source_split", "old_image", "old_stem", "new_stem", "new_image", "new_label"]
        )
        writer.writerows(manifest_rows)

    data_yaml = DST_ROOT / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {DST_ROOT.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: point",
                "  1: line",
                "",
                f"sample_size: {SAMPLE_SIZE}",
                f"train: {train_count}",
                f"val: {val_count}",
                f"random_seed: {RANDOM_SEED}",
                f"naming: {PREFIX}_{{split}}_{{index:06d}}",
            ]
        ),
        encoding="utf-8",
    )

    missing = 0
    for split in SPLITS:
        img_dir = DST_ROOT / "images" / split
        lbl_dir = DST_ROOT / "labels" / split
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if not (lbl_dir / f"{img.stem}.txt").exists():
                missing += 1

    print(f"src: {SRC_ROOT}")
    print(f"dst: {DST_ROOT}")
    print(f"total available: {len(all_pairs)}")
    print(f"sampled: {SAMPLE_SIZE}")
    print(f"train: {train_count}")
    print(f"val: {val_count}")
    print(f"naming: {PREFIX}_{{split}}_{{index:06d}}")
    print(f"missing pairs: {missing}")
    print(f"manifest: {manifest_path}")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
