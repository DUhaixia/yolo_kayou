"""双模型对比目录准备：按小类对齐全部原图，合并各模型有/无预测结果。

场景：模型1 有预测、模型2 无预测（或相反）时，仍能用 preview_infer_compare 三图并排查看。

输入:
  --source  原图根目录（各小类子文件夹），如 H:/卡游/压印testall2/骑行
  --model1  模型1推理输出根目录，如 G:/卡游/压印testall2/骑行ALL
  --model2  模型2推理输出根目录，如 G:/卡游/压印testall庭晖/骑行ALL

输出（每个小类 + 全部合并）:
  {out}/{小类}/
    orig/              原图（来自 source）
    {label1}_pred/     模型1：有缺陷用 _pred，无缺陷用原图（统一命名 *_pred 便于对比工具）
    {label2}_pred/     模型2 同上
    对比清单.csv       每张图的 m1/m2 预测类别、是否一致

同时在各模型根目录写入（可选）:
  {modelX}/{小类}/汇总/全部_pred/   该模型在本小类下有/无预测合并视图

运行:
  python scripts/prepare_dual_model_compare.py ^
    --source "H:/卡游/压印testall2/骑行" ^
    --model1 "G:/卡游/压印testall2/骑行ALL" ^
    --model2 "G:/卡游/压印testall庭晖/骑行ALL" ^
    --out    "G:/卡游/压印testall2/骑行双模型对比" ^
    --label1 testall2 --label2 tinghui
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
SKIP_TOP = {AGGREGATE_DIR, "有缺陷_orig", "有缺陷_pred", "无缺陷", "全部_pred", "对比清单.csv"}
PRED_RE = re.compile(r"^(.+)_pred(?:_(\d+))?(\.[^.]+)$", re.IGNORECASE)
ORIG_RE = re.compile(r"^(.+)_orig(?:_(\d+))?(\.[^.]+)$", re.IGNORECASE)
PLAIN_DEDUP_RE = re.compile(r"^(.+)_(\d+)(\.[^.]+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare dual-model compare folders")
    p.add_argument("--source", type=str, default=r"H:/卡游/压印testall2/骑行")
    p.add_argument("--model1", type=str, default=r"G:/卡游/压印testall2/骑行ALL")
    p.add_argument("--model2", type=str, default=r"G:/卡游/压印testall庭晖/骑行ALL")
    p.add_argument("--out", type=str, default=r"G:/卡游/压印testall2/骑行双模型对比")
    p.add_argument("--label1", type=str, default="model1")
    p.add_argument("--label2", type=str, default="model2")
    p.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    p.add_argument(
        "--write-model-merge",
        action="store_true",
        default=True,
        help="同时在各模型目录下写 {小类}/汇总/全部_pred（默认开启）",
    )
    p.add_argument("--no-write-model-merge", dest="write_model_merge", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_pair_name(filename: str) -> tuple[str, str, str, str]:
    m = PRED_RE.match(filename)
    if m:
        return m.group(1), "pred", m.group(3), m.group(2) or ""
    m = ORIG_RE.match(filename)
    if m:
        return m.group(1), "orig", m.group(3), m.group(2) or ""
    m = PLAIN_DEDUP_RE.match(filename)
    if m:
        return m.group(1), "plain", m.group(3), m.group(2)
    p = Path(filename)
    return p.stem, "plain", p.suffix or ".bmp", ""


def image_key(name: str) -> str:
    base, _role, _ext, tag = parse_pair_name(name)
    return f"{base}_{tag}" if tag else base


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


def transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def list_groups(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("~$") and d.name not in SKIP_TOP
    )


def index_source(group_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in list_images(group_dir):
        out[image_key(p.name)] = p
    return out


def index_model_group(group_dir: Path) -> dict[str, dict]:
    """key -> has_defect, cls, pred, orig, plain."""
    info: dict[str, dict] = {}

    def ensure(key: str) -> dict:
        return info.setdefault(key, {"has_defect": False, "cls": "no_defect"})

    for defect in DEFECT_DIRS:
        bucket = group_dir / defect
        if not bucket.is_dir():
            continue
        for p in list_images(bucket):
            key = image_key(p.name)
            row = ensure(key)
            _base, role, _ext, _tag = parse_pair_name(p.name)
            if role == "pred":
                row["has_defect"] = True
                row["cls"] = defect
                row["pred"] = p
            elif role == "orig":
                row["orig"] = p

    no_dir = group_dir / NO_DEFECT_DIR
    if no_dir.is_dir():
        for p in list_images(no_dir):
            key = image_key(p.name)
            row = ensure(key)
            if not row.get("has_defect"):
                row["cls"] = NO_DEFECT_DIR
                row["plain"] = p
    return info


def pick_pred_src(row: dict, fallback_orig: Path | None) -> Path | None:
    if row.get("has_defect") and row.get("pred"):
        return row["pred"]
    if row.get("orig"):
        return row["orig"]
    if row.get("plain"):
        return row["plain"]
    return fallback_orig


def cls_label(row: dict) -> str:
    if row.get("has_defect"):
        return str(row.get("cls", "defect"))
    return NO_DEFECT_DIR


def prepare_group(
    group: str,
    source_root: Path,
    model1_root: Path,
    model2_root: Path,
    out_base: Path,
    label1: str,
    label2: str,
    args,
    global_rows: list[dict],
    global_stats: dict,
) -> dict:
    src_dir = source_root / group
    m1_dir = model1_root / group
    m2_dir = model2_root / group

    src_map = index_source(src_dir) if src_dir.is_dir() else {}
    m1_map = index_model_group(m1_dir) if m1_dir.is_dir() else {}
    m2_map = index_model_group(m2_dir) if m2_dir.is_dir() else {}

    keys = sorted(set(src_map) | set(m1_map) | set(m2_map))
    if not keys:
        return {"keys": 0, "disagree": 0}

    out_orig = out_base / "orig"
    out_p1 = out_base / f"{label1}_pred"
    out_p2 = out_base / f"{label2}_pred"
    manifest = out_base / "对比清单.csv"
    m1_merge = m1_dir / AGGREGATE_DIR / "全部_pred" if args.write_model_merge else None
    m2_merge = m2_dir / AGGREGATE_DIR / "全部_pred" if args.write_model_merge else None

    if not args.dry_run:
        out_orig.mkdir(parents=True, exist_ok=True)
        out_p1.mkdir(parents=True, exist_ok=True)
        out_p2.mkdir(parents=True, exist_ok=True)
        if m1_merge:
            m1_merge.mkdir(parents=True, exist_ok=True)
        if m2_merge:
            m2_merge.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    disagree = 0

    for key in keys:
        src_path = src_map.get(key)
        if src_path is None:
            r1 = m1_map.get(key, {})
            r2 = m2_map.get(key, {})
            src_path = r1.get("orig") or r1.get("plain") or r2.get("orig") or r2.get("plain")
        if src_path is None:
            continue

        ext = src_path.suffix or ".bmp"
        r1 = m1_map.get(key, {"has_defect": False, "cls": "missing"})
        r2 = m2_map.get(key, {"has_defect": False, "cls": "missing"})
        c1, c2 = cls_label(r1), cls_label(r2)
        same = (r1.get("has_defect") == r2.get("has_defect")) and (c1 == c2 or (not r1.get("has_defect") and not r2.get("has_defect")))
        if not same:
            disagree += 1

        dst_orig = out_orig / f"{key}{ext}"
        dst_p1 = out_p1 / f"{key}_pred{ext}"
        dst_p2 = out_p2 / f"{key}_pred{ext}"

        p1_src = pick_pred_src(r1, src_path)
        p2_src = pick_pred_src(r2, src_path)

        rows.append({
            "小类": group,
            "图像key": key,
            "原图": src_path.name,
            f"{label1}_类别": c1,
            f"{label1}_有缺陷": "是" if r1.get("has_defect") else "否",
            f"{label2}_类别": c2,
            f"{label2}_有缺陷": "是" if r2.get("has_defect") else "否",
            "预测一致": "是" if same else "否",
        })
        global_rows.append({**rows[-1], "汇总前缀": group})

        if not args.dry_run:
            transfer(src_path, dst_orig, args.mode)
            if p1_src:
                transfer(p1_src, dst_p1, args.mode)
            if p2_src:
                transfer(p2_src, dst_p2, args.mode)
            if m1_merge and p1_src:
                transfer(p1_src, m1_merge / f"{key}_pred{ext}", args.mode)
            if m2_merge and p2_src:
                transfer(p2_src, m2_merge / f"{key}_pred{ext}", args.mode)

    if not args.dry_run and rows:
        with manifest.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    global_stats["keys"] += len(rows)
    global_stats["disagree"] += disagree
    print(
        f"  [{group}] keys={len(rows)}, 预测不一致={disagree}  ->  {out_base}"
    )
    return {"keys": len(rows), "disagree": disagree}


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    model1 = Path(args.model1).expanduser().resolve()
    model2 = Path(args.model2).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    for p, name in ((source, "source"), (model1, "model1"), (model2, "model2")):
        if not p.is_dir():
            raise FileNotFoundError(f"{name} not found: {p}")

    groups = sorted(set(list_groups(source)) | set(list_groups(model1)) | set(list_groups(model2)))
    print(f"Source : {source}")
    print(f"Model1 : {model1}  ({args.label1})")
    print(f"Model2 : {model2}  ({args.label2})")
    print(f"Out    : {out}")
    print(f"Groups : {', '.join(groups)}")
    if args.dry_run:
        print("(dry-run)\n")

    global_rows: list[dict] = []
    global_stats = {"keys": 0, "disagree": 0}

    for group in groups:
        prepare_group(
            group, source, model1, model2,
            out / group, args.label1, args.label2, args,
            global_rows, global_stats,
        )

    # 全部小类合并（带前缀，方便一次浏览）
    all_orig = out / "全部" / "orig"
    all_p1 = out / "全部" / f"{args.label1}_pred"
    all_p2 = out / "全部" / f"{args.label2}_pred"
    if not args.dry_run:
        all_orig.mkdir(parents=True, exist_ok=True)
        all_p1.mkdir(parents=True, exist_ok=True)
        all_p2.mkdir(parents=True, exist_ok=True)
        for row in global_rows:
            key = row["图像key"]
            group = row["小类"]
            ext = Path(row["原图"]).suffix or ".bmp"
            prefix = f"{group}__{key}"
            g_orig = out / group / "orig" / f"{key}{ext}"
            g_p1 = out / group / f"{args.label1}_pred" / f"{key}_pred{ext}"
            g_p2 = out / group / f"{args.label2}_pred" / f"{key}_pred{ext}"
            if g_orig.exists():
                transfer(g_orig, all_orig / f"{prefix}{ext}", args.mode)
            if g_p1.exists():
                transfer(g_p1, all_p1 / f"{prefix}_pred{ext}", args.mode)
            if g_p2.exists():
                transfer(g_p2, all_p2 / f"{prefix}_pred{ext}", args.mode)

        manifest_all = out / "全部" / "对比清单.csv"
        with manifest_all.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()) if global_rows else [])
            w.writeheader()
            w.writerows(global_rows)

    print()
    print("=== 完成 ===")
    print(f"总图像数    : {global_stats['keys']}")
    print(f"预测不一致  : {global_stats['disagree']}")
    if not args.dry_run:
        print()
        print("三图对比（全部小类）:")
        print(f'  --orig  "{all_orig}"')
        print(f'  --pred1 "{all_p1}"')
        print(f'  --pred2 "{all_p2}"')
        print(f'  --label1 "{args.label1}" --label2 "{args.label2}"')
        print()
        print("三图对比（单个小类，例如 线）:")
        ex = out / "线"
        print(f'  --orig  "{ex / "orig"}"')
        print(f'  --pred1 "{ex / f"{args.label1}_pred"}"')
        print(f'  --pred2 "{ex / f"{args.label2}_pred"}"')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
