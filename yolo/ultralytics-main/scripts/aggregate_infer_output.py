"""汇总 infer_and_split_results / infer_sliding_simple 的分目录输出，便于三图对比预览。

默认在每个「大类」文件夹内各自生成汇总（保留 骑行ALL/线、骑行ALL/难例 这一层）：

  {root}/线/
    defect_point/ ...
    汇总/
      有缺陷_orig/
      有缺陷_pred/
      无缺陷/
      汇总清单.csv

  {root}/难例/
    汇总/
      ...

运行:
  python scripts/aggregate_infer_output.py ^
    --root "G:/卡游/压印testall庭晖/骑行ALL"

全局汇总（旧行为，所有大类合并到一个目录）:
  python scripts/aggregate_infer_output.py ^
    --root "G:/..." --scope global --out "G:/.../骑行ALL_汇总"
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFECT_DIRS = ("defect_point", "defect_line", "defect_point_line")
NO_DEFECT_DIR = "no_defect"
AGGREGATE_DIR = "汇总"
SKIP_PARTS = {AGGREGATE_DIR, "有缺陷_orig", "有缺陷_pred", "无缺陷", "汇总清单.csv"}
PRED_RE = re.compile(r"^(.+)_pred(?:_(\d+))?(\.[^.]+)$", re.IGNORECASE)
ORIG_RE = re.compile(r"^(.+)_orig(?:_(\d+))?(\.[^.]+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate split inference output for compare preview")
    p.add_argument(
        "--root",
        type=str,
        default=r"G:/卡游/压印testall庭晖/骑行ALL",
        help="推理输出根目录（含 线/点/难例 等大类子文件夹）",
    )
    p.add_argument(
        "--scope",
        choices=["per-group", "global"],
        default="per-group",
        help="per-group=每个大类内各自汇总（默认）；global=全部合并到一个 out 目录",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="仅 scope=global 时有效；默认 {root}_汇总",
    )
    p.add_argument(
        "--summary-name",
        type=str,
        default=AGGREGATE_DIR,
        help=f"汇总文件夹名称，默认「{AGGREGATE_DIR}」",
    )
    p.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="copy=复制（默认）；hardlink=硬链接（同盘省空间）",
    )
    p.add_argument(
        "--no-prefix",
        action="store_true",
        help="不加子路径前缀；重名时自动加 _1 _2",
    )
    p.add_argument("--dry-run", action="store_true", help="只统计不复制")
    return p.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_role(filename: str) -> tuple[str, str, str, str]:
    m = PRED_RE.match(filename)
    if m:
        return m.group(1), "pred", m.group(3), m.group(2) or ""
    m = ORIG_RE.match(filename)
    if m:
        return m.group(1), "orig", m.group(3), m.group(2) or ""
    p = Path(filename)
    return p.stem, "plain", p.suffix or ".bmp", ""


def unique_target(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    idx = 1
    while True:
        candidate = dst.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def aggregate_name(src: Path, group_root: Path, no_prefix: bool) -> str:
    """组内汇总：仅对大类下的子路径加前缀，不再重复大类名。"""
    if no_prefix:
        return src.name
    rel_parent = src.parent.relative_to(group_root)
    parts = rel_parent.parts
    if parts and parts[-1] in DEFECT_DIRS + (NO_DEFECT_DIR,):
        subset = "__".join(parts[:-1]) if len(parts) > 1 else ""
        prefix = f"{subset}__" if subset else ""
    else:
        prefix = "__".join(parts) + "__" if parts else ""
    return prefix + src.name


def transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def find_buckets(group_root: Path) -> list[tuple[str, Path]]:
    buckets: list[tuple[str, Path]] = []
    for defect in DEFECT_DIRS:
        for d in sorted(group_root.rglob(defect)):
            if not d.is_dir():
                continue
            if AGGREGATE_DIR in d.parts or any(p in SKIP_PARTS for p in d.parts):
                continue
            buckets.append((defect, d))

    for d in sorted(group_root.rglob(NO_DEFECT_DIR)):
        if not d.is_dir():
            continue
        if AGGREGATE_DIR in d.parts or any(p in SKIP_PARTS for p in d.parts):
            continue
        buckets.append((NO_DEFECT_DIR, d))
    return buckets


def find_groups(root: Path) -> list[Path]:
    groups: list[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("~$")):
        if d.name in SKIP_PARTS or d.name.endswith("_汇总"):
            continue
        if find_buckets(d):
            groups.append(d)
    return groups


def aggregate_into(
    group_root: Path,
    out_base: Path,
    args,
    group_label: str,
) -> dict:
    out_orig = out_base / "有缺陷_orig"
    out_pred = out_base / "有缺陷_pred"
    out_no = out_base / "无缺陷"
    manifest_path = out_base / "汇总清单.csv"

    stats = {"orig": 0, "pred": 0, "no_defect": 0, "buckets": 0}
    rows: list[dict] = []
    buckets = find_buckets(group_root)
    stats["buckets"] = len(buckets)

    if not buckets:
        return stats

    print(f"\n==== {group_label}  ->  {out_base}  ({len(buckets)} buckets) ====")

    if not args.dry_run:
        out_base.mkdir(parents=True, exist_ok=True)

    for bucket_type, folder in buckets:
        rel_bucket = folder.relative_to(group_root)
        images = list_images(folder)
        if not images:
            continue
        print(f"  [{bucket_type}] {rel_bucket}  ({len(images)} files)")

        for src in images:
            role = parse_role(src.name)[1]
            name = aggregate_name(src, group_root, args.no_prefix)

            if bucket_type == NO_DEFECT_DIR:
                dst = out_no / name
                bucket_out = "无缺陷"
                stats["no_defect"] += 1
            elif role == "orig":
                dst = out_orig / name
                bucket_out = "有缺陷_orig"
                stats["orig"] += 1
            elif role == "pred":
                dst = out_pred / name
                bucket_out = "有缺陷_pred"
                stats["pred"] += 1
            else:
                if bucket_type != NO_DEFECT_DIR:
                    dst = out_orig / name
                    bucket_out = "有缺陷_orig"
                    stats["orig"] += 1
                else:
                    dst = out_no / name
                    bucket_out = "无缺陷"
                    stats["no_defect"] += 1

            if dst.exists():
                dst = unique_target(dst)

            rows.append({
                "大类": group_label,
                "汇总目录": bucket_out,
                "汇总文件名": dst.name,
                "源路径": str(src.relative_to(group_root)),
                "源缺陷类型": bucket_type,
            })

            if not args.dry_run:
                transfer(src, dst, args.mode)

    if not args.dry_run and rows:
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(
        f"  => orig={stats['orig']}, pred={stats['pred']}, "
        f"no_defect={stats['no_defect']}"
    )
    if not args.dry_run and rows:
        print(f"  => 清单: {manifest_path}")
    return stats


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found: {root}")

    print(f"Root  : {root}")
    print(f"Scope : {args.scope}")
    if args.dry_run:
        print("(dry-run, no files copied)")

    grand = {"orig": 0, "pred": 0, "no_defect": 0, "groups": 0}

    if args.scope == "global":
        out = Path(args.out).expanduser().resolve() if args.out else Path(str(root) + "_汇总")
        stats = aggregate_into(root, out, args, group_label="ALL")
        grand["orig"] = stats["orig"]
        grand["pred"] = stats["pred"]
        grand["no_defect"] = stats["no_defect"]
        grand["groups"] = 1 if stats["buckets"] else 0
        if not args.dry_run and stats["buckets"]:
            print(f"\n三图对比:")
            print(f'  --orig  "{out / "有缺陷_orig"}"')
            print(f'  --pred1 "{out / "有缺陷_pred"}"')
    else:
        groups = find_groups(root)
        if not groups:
            print("未找到含 defect_*/no_defect 的大类文件夹")
            return
        print(f"Found {len(groups)} group(s): {', '.join(g.name for g in groups)}")

        for group in groups:
            out_base = group / args.summary_name
            stats = aggregate_into(group, out_base, args, group_label=group.name)
            grand["orig"] += stats["orig"]
            grand["pred"] += stats["pred"]
            grand["no_defect"] += stats["no_defect"]
            if stats["buckets"]:
                grand["groups"] += 1

        if not args.dry_run and grand["groups"]:
            example = groups[0] / args.summary_name
            print(f"\n三图对比示例（以「{groups[0].name}」为例）:")
            print(f'  --orig  "{example / "有缺陷_orig"}"')
            print(f'  --pred1 "{example / "有缺陷_pred"}"')
            print(f'  --pred2 "<另一模型/{groups[0].name}/{args.summary_name}/有缺陷_pred"')

    print()
    print("=== 全部汇总完成 ===")
    print(f"大类数      : {grand['groups']}")
    print(f"有缺陷_orig : {grand['orig']}")
    print(f"有缺陷_pred : {grand['pred']}")
    print(f"无缺陷      : {grand['no_defect']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
