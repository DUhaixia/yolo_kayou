"""
从 YOLO seg 训练集中按类别抽取高质量子集（用于微调/补强）。

类别:
  point (id 0) — 含 point 标注的图
  line  (id 1) — 含 line 标注的图
  good  — 空 txt 好品

质量评分（越高优先）:
  - 单类缺陷（仅 point 或仅 line）优于混合
  - 实例数 1~3 优于过多
  - polygon 归一化面积在合理范围（非极小噪点）

示例:
  python scripts/sample_quality_seg_subset.py ^
    --src "M:/压印 - 副本/dataSet-原始-line_only+XY标注" ^
    --out "M:/压印 - 副本/dataSet-ft-quality1100" ^
    --train-point 500 --train-line 500 --train-good 100 ^
    --val-point 130 --val-line 130 --val-good 60

  # 在已有微调集上追加样本
  python scripts/sample_quality_seg_subset.py ^
    --src "M:/压印 - 副本/dataSet-原始-line_only+XY标注" ^
    --base "M:/压印 - 副本/dataSet-ft-quality1100" ^
    --out "M:/压印 - 副本/dataSet-ft-expanded" ^
    --add-point 500 --add-line 500 --add-good 3000 --add-val-ratio 0.2
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    img: Path
    lbl: Path
    split: str
    category: str  # point | line | good
    score: float
    n_point: int
    n_line: int


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES


def polygon_bbox_area(parts: list[str]) -> float:
    coords = [float(x) for x in parts[1:]]
    if len(coords) < 6:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def analyze_label(lbl: Path) -> tuple[int, int, list[float], bool]:
    text = lbl.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return 0, 0, [], True
    n_point = n_line = 0
    areas: list[float] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        cls = int(float(parts[0]))
        area = polygon_bbox_area(parts)
        if cls == 0:
            n_point += 1
        elif cls == 1:
            n_line += 1
        if area > 0:
            areas.append(area)
    return n_point, n_line, areas, False


def quality_score(n_point: int, n_line: int, areas: list[float], is_good: bool) -> float:
    if is_good:
        return 10.0

    score = 0.0
    total = n_point + n_line
    only_one_class = (n_point > 0 and n_line == 0) or (n_line > 0 and n_point == 0)
    if only_one_class:
        score += 20.0
    else:
        score += 5.0

    if total == 1:
        score += 15.0
    elif 2 <= total <= 3:
        score += 12.0
    elif 4 <= total <= 5:
        score += 6.0
    else:
        score += 1.0

    if areas:
        med = sorted(areas)[len(areas) // 2]
        if 0.0003 <= med <= 0.08:
            score += 10.0
        elif 0.0001 <= med <= 0.15:
            score += 6.0
        elif med > 0.00005:
            score += 2.0
        # 极小框扣分
        if med < 0.0001:
            score -= 5.0

    return score


def classify_sample(img: Path, lbl: Path, split: str) -> list[Sample]:
    n_point, n_line, areas, is_good = analyze_label(lbl)
    score = quality_score(n_point, n_line, areas, is_good)
    out: list[Sample] = []

    if is_good:
        out.append(Sample(img, lbl, split, "good", score, 0, 0))
        return out

    if n_point > 0:
        out.append(Sample(img, lbl, split, "point", score, n_point, n_line))
    if n_line > 0:
        line_score = score
        if n_point == 0:
            line_score += 2.0  # 纯 line 再加分
        out.append(Sample(img, lbl, split, "line", line_score, n_point, n_line))
    return out


def collect_pool(src: Path, split: str) -> dict[str, list[Sample]]:
    pool: dict[str, list[Sample]] = {"point": [], "line": [], "good": []}
    img_dir = src / "images" / split
    lbl_dir = src / "labels" / split
    seen: set[str] = set()

    for img in sorted(img_dir.iterdir()):
        if not is_image(img):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.is_file():
            continue
        for s in classify_sample(img, lbl, split):
            key = f"{s.category}:{img.stem}"
            if key in seen:
                continue
            seen.add(key)
            pool[s.category].append(s)
    for cat in pool:
        pool[cat].sort(key=lambda x: (-x.score, x.img.name))
    return pool


def pick_samples(
    pool: list[Sample],
    n: int,
    used_stems: set[str],
    rng: random.Random,
) -> list[Sample]:
    """先取高分，同分块内随机，避免总取同一批。"""
    available = [s for s in pool if s.img.stem not in used_stems]
    if len(available) <= n:
        return available

    # 按 score 分桶，桶内 shuffle 后按分数降序拼接
    buckets: dict[float, list[Sample]] = {}
    for s in available:
        buckets.setdefault(s.score, []).append(s)
    ordered: list[Sample] = []
    for sc in sorted(buckets.keys(), reverse=True):
        batch = buckets[sc]
        rng.shuffle(batch)
        ordered.extend(batch)
    return ordered[:n]


def collect_pool_all(src: Path, exclude_stems: set[str]) -> dict[str, list[Sample]]:
    """从 train+val 汇总候选池，排除已用 stem。"""
    pool: dict[str, list[Sample]] = {"point": [], "line": [], "good": []}
    seen: set[str] = set()
    for split in ("train", "val"):
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not is_image(img) or img.stem in exclude_stems:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            for s in classify_sample(img, lbl, split):
                key = f"{s.category}:{img.stem}"
                if key in seen:
                    continue
                seen.add(key)
                pool[s.category].append(s)
    for cat in pool:
        pool[cat].sort(key=lambda x: (-x.score, x.img.name))
    return pool


def copy_dataset_tree(base: Path, out: Path) -> set[str]:
    """复制已有数据集到 out，返回已有 stem 集合。"""
    stems: set[str] = set()
    for split in ("train", "val"):
        src_img = base / "images" / split
        src_lbl = base / "labels" / split
        if not src_img.is_dir():
            continue
        for img in sorted(src_img.iterdir()):
            if not is_image(img):
                continue
            lbl = src_lbl / f"{img.stem}.txt"
            dst_img = out / "images" / split / img.name
            dst_lbl = out / "labels" / split / f"{img.stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst_img)
            if lbl.is_file():
                shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
            stems.add(img.stem)
    return stems


def split_add_counts(total: int, val_ratio: float) -> tuple[int, int]:
    total = max(0, total)
    if total == 0:
        return 0, 0
    n_val = int(round(total * val_ratio))
    if total > 1:
        n_val = max(1, min(total - 1, n_val))
    else:
        n_val = 0
    return total - n_val, n_val


def copy_sample(s: Sample, out: Path) -> None:
    dst_img = out / "images" / s.split / s.img.name
    dst_lbl = out / "labels" / s.split / s.lbl.name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s.img, dst_img)
    shutil.copy2(s.lbl, dst_lbl)


def pick_and_copy_additions(
    pool: list[Sample],
    n_train: int,
    n_val: int,
    used_stems: set[str],
    out: Path,
    rng: random.Random,
    category: str,
    rows: list[dict],
) -> tuple[int, int]:
    picked_train = pick_samples(pool, n_train, used_stems, rng)
    for s in picked_train:
        s.split = "train"
        copy_sample(s, out)
        used_stems.add(s.img.stem)
        rows.append({
            "split": "train",
            "category": category,
            "file": s.img.name,
            "score": f"{s.score:.1f}",
            "n_point": s.n_point,
            "n_line": s.n_line,
            "action": "add",
        })
    picked_val = pick_samples(pool, n_val, used_stems, rng)
    for s in picked_val:
        s.split = "val"
        copy_sample(s, out)
        used_stems.add(s.img.stem)
        rows.append({
            "split": "val",
            "category": category,
            "file": s.img.name,
            "score": f"{s.score:.1f}",
            "n_point": s.n_point,
            "n_line": s.n_line,
            "action": "add",
        })
    return len(picked_train), len(picked_val)


    dst_img = out / "images" / s.split / s.img.name
    dst_lbl = out / "labels" / s.split / s.lbl.name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s.img, dst_img)
    shutil.copy2(s.lbl, dst_lbl)


def main() -> None:
    ap = argparse.ArgumentParser(description="按类别抽取高质量 seg 子集")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-point", type=int, default=500)
    ap.add_argument("--train-line", type=int, default=500)
    ap.add_argument("--train-good", type=int, default=100)
    ap.add_argument("--val-point", type=int, default=130)
    ap.add_argument("--val-line", type=int, default=130)
    ap.add_argument("--val-good", type=int, default=60)
    ap.add_argument("--base", type=Path, default=None, help="已有微调集，先复制再追加")
    ap.add_argument("--add-point", type=int, default=0, help="追加 point 总数（按 val 比例划分 train/val）")
    ap.add_argument("--add-line", type=int, default=0, help="追加 line 总数")
    ap.add_argument("--add-good", type=int, default=0, help="追加好品总数")
    ap.add_argument("--add-val-ratio", type=float, default=0.2, help="追加样本 val 比例，默认 0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="清空输出目录后重建")
    args = ap.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    expand_mode = args.base is not None or args.add_point or args.add_line or args.add_good

    if expand_mode:
        if args.clean and out.exists():
            shutil.rmtree(out)
        used_stems: set[str] = set()
        if args.base is not None:
            base = args.base.resolve()
            print(f"copy base -> {out} from {base}")
            used_stems = copy_dataset_tree(base, out)
            print(f"  base images: {len(used_stems)}")
        rng = random.Random(args.seed)
        pool = collect_pool_all(src, used_stems)
        print(
            f"unused pool: point={len(pool['point'])}, line={len(pool['line'])}, good={len(pool['good'])}"
        )
        rows: list[dict] = []
        add_stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}
        vr = max(0.05, min(0.95, args.add_val_ratio))

        for cat, total in (
            ("point", args.add_point),
            ("line", args.add_line),
            ("good", args.add_good),
        ):
            if total <= 0:
                continue
            n_train, n_val = split_add_counts(total, vr)
            if len(pool[cat]) < total:
                print(f"[warn] {cat}: need {total}, pool only {len(pool[cat])}")
                n_train, n_val = split_add_counts(len(pool[cat]), vr)
            tr, va = pick_and_copy_additions(
                pool[cat], n_train, n_val, used_stems, out, rng, cat, rows,
            )
            add_stats["train"][cat] = tr
            add_stats["val"][cat] = va
            print(f"  add {cat}: train +{tr}, val +{va}")

        data_yaml = {
            "path": str(out).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "point", 1: "line"},
        }
        out.mkdir(parents=True, exist_ok=True)
        with (out / "data.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

        manifest = out / "sample_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["split", "category", "file", "score", "n_point", "n_line", "action"],
            )
            w.writeheader()
            w.writerows(rows)

        final = {"train": {}, "val": {}}
        for sp in ("train", "val"):
            for cat in ("point", "line", "good"):
                n = 0
                for f in (out / "labels" / sp).glob("*.txt"):
                    t = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if cat == "good" and not t:
                        n += 1
                    elif t:
                        h0 = h1 = False
                        for ln in t.splitlines():
                            if ln.strip():
                                c = int(float(ln.split()[0]))
                                if c == 0:
                                    h0 = True
                                elif c == 1:
                                    h1 = True
                        if cat == "point" and h0:
                            n += 1
                        if cat == "line" and h1:
                            n += 1
                final[sp][cat] = n
        print(f"\ndone -> {out}")
        print(f"added train: {add_stats['train']}")
        print(f"added val:   {add_stats['val']}")
        print(f"final train: {final['train']}")
        print(f"final val:   {final['val']}")
        print(f"data.yaml -> {out / 'data.yaml'}")
        return

    targets = {
        "train": {"point": args.train_point, "line": args.train_line, "good": args.train_good},
        "val": {"point": args.val_point, "line": args.val_line, "good": args.val_good},
    }

    if args.clean and out.exists():
        shutil.rmtree(out)

    rng = random.Random(args.seed)
    used_stems: set[str] = set()
    rows: list[dict] = []
    stats: dict[str, dict[str, int]] = {}

    for split in ("train", "val"):
        pool = collect_pool(src, split)
        stats[split] = {}
        print(f"\n[{split}] pool: point={len(pool['point'])}, line={len(pool['line'])}, good={len(pool['good'])}")

        for cat, n in targets[split].items():
            picked = pick_samples(pool[cat], n, used_stems, rng)
            if len(picked) < n:
                print(f"[warn] {split}/{cat}: need {n}, only {len(picked)} available")
            for s in picked:
                copy_sample(s, out)
                used_stems.add(s.img.stem)
                rows.append({
                    "split": split,
                    "category": cat,
                    "file": s.img.name,
                    "score": f"{s.score:.1f}",
                    "n_point": s.n_point,
                    "n_line": s.n_line,
                })
            stats[split][cat] = len(picked)
            avg = sum(s.score for s in picked) / len(picked) if picked else 0
            print(f"  {cat}: picked {len(picked)}, avg_score={avg:.1f}")

    data_yaml = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "point", 1: "line"},
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    manifest = out / "sample_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "category", "file", "score", "n_point", "n_line"])
        w.writeheader()
        w.writerows(rows)

    total = sum(sum(v.values()) for v in stats.values())
    print(f"\ndone -> {out}")
    print(f"total images: {total}")
    print(f"train: {stats['train']}")
    print(f"val:   {stats['val']}")
    print(f"data.yaml -> {out / 'data.yaml'}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
