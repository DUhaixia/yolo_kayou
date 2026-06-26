"""按缺陷类型细分 YOLO 数据集：复制图片+标签到新目录，不移动 .npy。"""
from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path

SRC_ROOT = Path(r"M:\压印 - 副本\dataSet-原始-切割-已筛选+neg")
DST_ROOT = Path(r"M:\压印 - 副本\dataSet-原始-切割-已筛选+neg_细分")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")
CATEGORIES = ("point", "line", "point_and_line", "background", "no_label")


def classify_label(lbl_path: Path) -> str:
    if not lbl_path.exists():
        return "no_label"

    content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return "background"

    classes = set()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        cls = int(float(line.split()[0]))
        if cls == 0:
            classes.add("point")
        elif cls == 1:
            classes.add("line")

    if classes == {"point"}:
        return "point"
    if classes == {"line"}:
        return "line"
    if classes == {"point", "line"}:
        return "point_and_line"
    return "no_label"


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    stats: dict[str, Counter] = {cat: Counter() for cat in CATEGORIES}
    copied_files = 0

    for split in SPLITS:
        src_img_dir = SRC_ROOT / "images" / split
        src_lbl_dir = SRC_ROOT / "labels" / split

        stems = set()
        for p in src_img_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                stems.add(p.stem)

        for stem in sorted(stems):
            img_path = find_image(src_img_dir, stem)
            lbl_path = src_lbl_dir / f"{stem}.txt"
            category = classify_label(lbl_path)

            dst_img_dir = DST_ROOT / category / "images" / split
            dst_lbl_dir = DST_ROOT / category / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            if img_path is not None:
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                copied_files += 1

            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
                copied_files += 1
            elif category == "no_label":
                # 无标签时创建空标签，保持 YOLO 结构一致
                (dst_lbl_dir / f"{stem}.txt").touch()

            stats[category][split] += 1

    report_lines = [
        f"src: {SRC_ROOT}",
        f"dst: {DST_ROOT}",
        "note: 仅复制图片与 txt 标签，未移动任何 .npy",
        "",
    ]
    for cat in CATEGORIES:
        if not stats[cat]:
            continue
        report_lines.append(f"{cat}:")
        for split in SPLITS:
            report_lines.append(f"  {split}: {stats[cat][split]}")
        report_lines.append(f"  total: {sum(stats[cat].values())}")
        report_lines.append("")

    report_path = DST_ROOT / "split_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"copied files: {copied_files}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
