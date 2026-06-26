"""Consolidate the 骑行 dataset into 汇总2, grouped by category then by Stat.

Source root (``--root``) holds many top-level folders. We map them into a small
set of categories and, inside each category, route every ``.bmp`` into a
``statN`` sub-folder taken from the ``(StatN)`` token in the file name.

Category rules (matching the layout already prepared under 汇总2):
    正面好品*   -> 正面好品   (stat1/2/3)
    反面好品*   -> 反面好品   (stat4/5/6)
    反面划伤    -> 反面划伤
    反面压印    -> 反面压印
    正面压印11  -> 正面压印

Files with the same name coming from different source folders are kept and
de-duplicated with a " (2)", " (3)" ... suffix (same behaviour as the user's
manual 汇总 attempt).

Usage (PowerShell):
    # preview only, nothing copied:
    python scripts/consolidate_qixing.py --dry-run
    # actually copy:
    python scripts/consolidate_qixing.py
    # move instead of copy:
    python scripts/consolidate_qixing.py --move
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

STAT_RE = re.compile(r"\(stat(\d+)\)", re.IGNORECASE)

# exact-name source folder -> category
EXACT_MAP = {
    "反面划伤": "反面划伤",
    "反面压印": "反面压印",
    "正面压印11": "正面压印",
}
# prefix source folder -> category
PREFIX_MAP = {
    "正面好品": "正面好品",
    "反面好品": "反面好品",
}


def category_for(folder_name: str) -> str | None:
    if folder_name in EXACT_MAP:
        return EXACT_MAP[folder_name]
    for prefix, cat in PREFIX_MAP.items():
        if folder_name.startswith(prefix):
            return cat
    return None


def stat_for(filename: str) -> str | None:
    m = STAT_RE.search(filename)
    return f"stat{m.group(1)}" if m else None


def dedup_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 2
    while True:
        cand = target.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"J:\骑行")
    ap.add_argument("--dst", default=r"J:\骑行\汇总2")
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    args = ap.parse_args()

    root = Path(args.root)
    dst = Path(args.dst)
    if not root.is_dir():
        sys.exit(f"ERROR: root not found: {root}")

    # never treat the output folders as sources
    skip_names = {dst.name, "汇总"}

    # collect source folders per category
    sources: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in skip_names:
            continue
        cat = category_for(child.name)
        if cat:
            sources[cat].append(child)

    if not sources:
        sys.exit("No matching source folders found.")

    print("Plan (root =", root, "-> dst =", dst, ")")
    print(f"Mode: {'MOVE' if args.move else 'COPY'}{'  [DRY-RUN]' if args.dry_run else ''}\n")

    grand_total = 0
    grand_no_stat = 0
    grand_collisions = 0

    for cat in sorted(sources):
        src_folders = sources[cat]
        per_stat = Counter()
        no_stat = []
        collisions = 0
        copied = 0

        print(f"=== {cat} ===")
        print("  sources:", [s.name for s in src_folders])

        for sf in src_folders:
            for bmp in sf.rglob("*.bmp"):
                stat = stat_for(bmp.name)
                if stat is None:
                    no_stat.append(str(bmp))
                    grand_no_stat += 1
                    continue
                per_stat[stat] += 1
                out_dir = dst / cat / stat
                target = out_dir / bmp.name
                final = dedup_path(target) if not args.dry_run else target
                if not args.dry_run:
                    if final != target:
                        collisions += 1
                    out_dir.mkdir(parents=True, exist_ok=True)
                    if args.move:
                        shutil.move(str(bmp), str(final))
                    else:
                        shutil.copy2(str(bmp), str(final))
                    copied += 1

        total_cat = sum(per_stat.values())
        grand_total += total_cat
        grand_collisions += collisions
        print("  by stat:", dict(sorted(per_stat.items())), " total:", total_cat)
        if not args.dry_run:
            print(f"  {'moved' if args.move else 'copied'}: {copied}  renamed-on-collision: {collisions}")
        if no_stat:
            print(f"  WARNING: {len(no_stat)} file(s) had no (StatN) token, skipped. e.g. {no_stat[:3]}")
        print()

    print("=" * 60)
    print(f"GRAND TOTAL files routed: {grand_total}")
    if grand_collisions:
        print(f"Renamed on collision: {grand_collisions}")
    if grand_no_stat:
        print(f"Skipped (no StatN): {grand_no_stat}")
    if args.dry_run:
        print("\nDRY-RUN only. Re-run without --dry-run to perform the copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
