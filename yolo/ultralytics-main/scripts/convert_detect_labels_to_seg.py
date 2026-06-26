"""将 YOLO 检测框标注 (cls cx cy w h) 转为分割多边形 (矩形四角点)，供 segment 训练使用。"""

from __future__ import annotations

import argparse
from pathlib import Path


def detect_line_to_seg(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    if len(parts) > 5:
        return line.strip()
    cls_id = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return (
        f"{cls_id} {x1:.6g} {y1:.6g} {x2:.6g} {y2:.6g} "
        f"{x3:.6g} {y3:.6g} {x4:.6g} {y4:.6g}"
    )


def convert_file(path: Path) -> tuple[int, int]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    n_bbox, n_seg = 0, 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) > 5:
            out.append(line)
            n_seg += 1
        elif len(parts) >= 5:
            seg = detect_line_to_seg(line)
            if seg:
                out.append(seg)
                n_bbox += 1
    path.write_text(("\n".join(out) + "\n") if out else "", encoding="utf-8")
    return n_bbox, n_seg


def main() -> None:
    p = argparse.ArgumentParser(description="检测框标注转分割矩形多边形")
    p.add_argument("--root", type=Path, required=True, help="数据集根目录 (含 labels/train, labels/val)")
    p.add_argument("--clean-cache", action="store_true", help="删除 labels/*.cache")
    args = p.parse_args()

    root = args.root.resolve()
    total_bbox = total_seg = 0
    n_files = 0
    for split in ("train", "val"):
        lbl_dir = root / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for txt in lbl_dir.glob("*.txt"):
            b, s = convert_file(txt)
            total_bbox += b
            total_seg += s
            n_files += 1
        print(f"[{split}] converted {len(list(lbl_dir.glob('*.txt')))} files")

    if args.clean_cache:
        for cache in (root / "labels").glob("*.cache"):
            cache.unlink()
            print(f"removed cache: {cache}")

    print(f"done: files={n_files}, bbox->seg={total_bbox}, already_seg={total_seg}")


if __name__ == "__main__":
    main()
