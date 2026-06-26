"""
合并数据集（去重、可验证）:
  1) 原始训练集 dataSet-原始-切割-已筛选（含 point+line）
  2) line_only 子集：重命名后追加（class 0→1，用于线缺陷过采样）
  3) 好品负样本 neg_edge_bg/images/train/train，7:3 划分 train/val，空 label 追加

示例:
  python scripts/merge_orig_neg_dataset.py ^
    --orig "M:/压印 - 副本/dataSet-原始-切割-已筛选" ^
    --line-only "M:/压印 - 副本/dataSet-原始-切割-已筛选-line_only" ^
    --neg-images "M:/压印 - 副本/neg_edge_bg/images/train/train" ^
    --out "M:/压印 - 副本/dataSet-原始-切割-已筛选+neg" ^
    --neg-ratio 0.7
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def copy_orig_split(src_root: Path, dst_root: Path, split: str) -> tuple[int, set[str]]:
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    stems: set[str] = set()
    n = 0
    for lbl in sorted(src_lbl.glob("*.txt")):
        stem = lbl.stem
        img = None
        for ext in IMAGE_SUFFIXES:
            cand = src_img / f"{stem}{ext}"
            if cand.is_file():
                img = cand
                break
        if img is None:
            print(f"[warn] orig missing image: {lbl.name}")
            continue
        shutil.copy2(img, dst_img / img.name)
        shutil.copy2(lbl, dst_lbl / lbl.name)
        stems.add(stem)
        n += 1
    return n, stems


def remap_label_class(text: str, from_cls: int, to_cls: int) -> str:
    out: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if int(parts[0]) == from_cls:
            parts[0] = str(to_cls)
        out.append(" ".join(parts))
    return "\n".join(out) + ("\n" if out else "")


def add_line_only_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    used_stems: set[str],
    prefix: str = "line_",
    from_cls: int = 0,
    to_cls: int = 1,
) -> tuple[int, list[dict]]:
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n = 0
    for lbl in sorted(src_lbl.glob("*.txt")):
        stem = lbl.stem
        img = None
        for ext in IMAGE_SUFFIXES:
            cand = src_img / f"{stem}{ext}"
            if cand.is_file():
                img = cand
                break
        if img is None:
            print(f"[warn] line_only missing image: {lbl.name}")
            continue

        new_stem = f"{prefix}{stem}"
        if new_stem in used_stems:
            k = 1
            while f"{prefix}{stem}_{k:03d}" in used_stems:
                k += 1
            new_stem = f"{prefix}{stem}_{k:03d}"

        out_img = dst_img / f"{new_stem}{img.suffix}"
        out_lbl = dst_lbl / f"{new_stem}.txt"
        shutil.copy2(img, out_img)
        out_lbl.write_text(
            remap_label_class(lbl.read_text(encoding="utf-8", errors="ignore"), from_cls, to_cls),
            encoding="utf-8",
        )
        used_stems.add(new_stem)
        n += 1
        rows.append({"split": split, "file": out_img.name, "source": img.name, "tag": "line_oversample"})
    return n, rows


def add_neg_split(
    neg_images: Path,
    dst_root: Path,
    split: str,
    items: list[Path],
    used_stems: set[str],
    prefix: str = "good_",
) -> tuple[int, list[dict]]:
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n = 0
    for img in items:
        new_stem = f"{prefix}{img.stem}"
        if new_stem in used_stems:
            new_stem = f"{prefix}{img.stem}_neg"
            k = 1
            while new_stem in used_stems:
                new_stem = f"{prefix}{img.stem}_neg{k:02d}"
                k += 1
        out_img = dst_img / f"{new_stem}{img.suffix}"
        out_lbl = dst_lbl / f"{new_stem}.txt"
        shutil.copy2(img, out_img)
        out_lbl.write_text("", encoding="utf-8")
        used_stems.add(new_stem)
        n += 1
        rows.append({"split": split, "file": out_img.name, "source": img.name, "tag": "neg"})
    return n, rows


def verify_dataset(root: Path) -> dict:
    report: dict = {"ok": True, "issues": []}
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        lbls = list(lbl_dir.glob("*.txt"))
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}

        bad_ext = [p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES]
        miss_lbl = sorted(img_stems - lbl_stems)
        miss_img = sorted(lbl_stems - img_stems)
        dup_img = len(imgs) - len(img_stems)

        empty = annotated = 0
        cls: set[str] = set()
        for t in lbls:
            txt = t.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                empty += 1
            else:
                annotated += 1
                for ln in txt.splitlines():
                    if ln.strip():
                        cls.add(ln.split()[0])

        rep = {
            "imgs": len(imgs),
            "lbls": len(lbls),
            "dup_img": dup_img,
            "miss_lbl": len(miss_lbl),
            "miss_img": len(miss_img),
            "bad_ext": len(bad_ext),
            "empty": empty,
            "annotated": annotated,
            "classes": sorted(cls),
        }
        report[split] = rep
        if miss_lbl or miss_img or dup_img or bad_ext:
            report["ok"] = False
            report["issues"].append(f"{split}: pairing/dup/ext problem")

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="合并原始训练集 + 好品负样本")
    p.add_argument("--orig", required=True)
    p.add_argument("--neg-images", required=True, help="好品图目录，如 neg_edge_bg/images/train/train")
    p.add_argument("--out", required=True)
    p.add_argument("--neg-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--line-only", default="", help="线子集目录，重命名后追加（class 0→1）")
    p.add_argument("--line-prefix", default="line_", help="line_only 重命名前缀")
    p.add_argument("--skip-line-only", action="store_true", help="不追加 line_only")
    args = p.parse_args()

    orig_root = Path(args.orig).resolve()
    neg_dir = Path(args.neg_images).resolve()
    out_root = Path(args.out).resolve()
    line_only_root = Path(args.line_only).resolve() if args.line_only else None

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) 原始集
    used_train: set[str] = set()
    used_val: set[str] = set()
    n_orig_train, used_train = copy_orig_split(orig_root, out_root, "train")
    n_orig_val, used_val = copy_orig_split(orig_root, out_root, "val")
    print(f"[orig] train={n_orig_train}, val={n_orig_val}")

    rows_line: list[dict] = []
    n_line_tr = n_line_va = 0
    if line_only_root and line_only_root.is_dir() and not args.skip_line_only:
        lo_train = {p.stem for p in (line_only_root / "labels" / "train").glob("*.txt")}
        lo_not_in_orig = lo_train - {s for s in used_train}
        if lo_not_in_orig:
            print(f"[warn] line_only 有 {len(lo_not_in_orig)} 个样本不在 orig 中")
        overlap = len(lo_train & used_train)
        print(f"[line_only] overlap with orig train stems: {overlap}/{len(lo_train)}")
        n_line_tr, rows_tr_lo = add_line_only_split(
            line_only_root, out_root, "train", used_train, prefix=args.line_prefix,
        )
        n_line_va, rows_va_lo = add_line_only_split(
            line_only_root, out_root, "val", used_val, prefix=args.line_prefix,
        )
        rows_line = rows_tr_lo + rows_va_lo
        print(f"[line_only] added train={n_line_tr}, val={n_line_va} (renamed + class 0->1)")

    # 2) 好品负样本 7:3
    neg_imgs = list_images(neg_dir)
    if not neg_imgs:
        raise FileNotFoundError(f"no neg images in {neg_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(neg_imgs)
    ratio = max(0.05, min(0.95, args.neg_ratio))
    n_tr = int(round(len(neg_imgs) * ratio))
    n_tr = max(1, min(len(neg_imgs) - 1, n_tr)) if len(neg_imgs) > 1 else 1
    neg_train = neg_imgs[:n_tr]
    neg_val = neg_imgs[n_tr:]

    n_neg_tr, rows_tr = add_neg_split(neg_dir, out_root, "train", neg_train, used_train)
    n_neg_val, rows_val = add_neg_split(neg_dir, out_root, "val", neg_val, used_val)
    print(f"[neg] total={len(neg_imgs)}, train={n_neg_tr}, val={n_neg_val}")

    # 3) data.yaml
    src_yaml = orig_root / "data.yaml"
    names = {0: "point", 1: "line"}
    if src_yaml.is_file():
        cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and cfg.get("names"):
            names = cfg["names"]
    data = {
        "path": str(out_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    # 4) manifest + verify
    manifest = out_root / "merge_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "file", "source", "tag"])
        w.writeheader()
        for r in rows_line + rows_tr + rows_val:
            w.writerow(r)

    report = verify_dataset(out_root)
    print("=== verify ===")
    for split in ("train", "val"):
        if split in report:
            print(split, report[split])
    print("verify_ok:", report["ok"])
    if report["issues"]:
        print("issues:", report["issues"])

    with (out_root / "merge_report.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({
            "orig_train": n_orig_train,
            "orig_val": n_orig_val,
            "line_only_train": n_line_tr,
            "line_only_val": n_line_va,
            "neg_train": n_neg_tr,
            "neg_val": n_neg_val,
            "verify": report,
        }, f, allow_unicode=True, sort_keys=False)

    print(f"done -> {out_root}")
    print(
        f"train total = {n_orig_train + n_line_tr + n_neg_tr}, "
        f"val total = {n_orig_val + n_line_va + n_neg_val}"
    )


if __name__ == "__main__":
    main()
