"""批量裁切压印数据集 train/val，直接输出 YOLO 标准目录。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(r"M:\压印 - 副本\dataSet-原始")
DST_ROOT = Path(r"M:\压印 - 副本\dataSet-原始-切割")
CROP_SCRIPT = ROOT / "scripts" / "crop_defect_center_seg.py"


def run_split(split: str, preview: bool = True) -> None:
    cmd = [
        sys.executable,
        str(CROP_SCRIPT),
        "--src-images", str(SRC_ROOT / "images" / split),
        "--src-labels", str(SRC_ROOT / "labels" / split),
        "--dst-images", str(DST_ROOT / "images" / split),
        "--dst-labels", str(DST_ROOT / "labels" / split),
        "--manifest", str(DST_ROOT / f"manifest_{split}.csv"),
        "--patch", "512",
        "--line-classes", "1",
        "--min-line-len", "6",
        "--min-area", "4",
    ]
    if preview:
        cmd.extend(["--dst-preview", str(DST_ROOT / "preview" / split)])
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_data_yaml() -> None:
    text = f"""# auto-generated cropped dataset
path: {DST_ROOT}
train: images/train
val: images/val
names:
  0: point
  1: line
"""
    (DST_ROOT / "data.yaml").write_text(text, encoding="utf-8")
    print(f"data.yaml -> {DST_ROOT / 'data.yaml'}")


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    run_split("train", preview=True)
    run_split("val", preview=True)
    write_data_yaml()
    print("all done.")


if __name__ == "__main__":
    main()
