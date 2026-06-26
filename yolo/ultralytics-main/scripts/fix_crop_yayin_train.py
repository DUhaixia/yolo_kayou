"""合并中断的 train 裁切结果到标准目录。"""
import shutil
from pathlib import Path

DST = Path(r"M:\压印 - 副本\dataSet-原始-切割")
img_dst = DST / "images" / "train"
lbl_dst = DST / "labels" / "train"
prev_dst = DST / "preview" / "train"

for d in (img_dst, lbl_dst, prev_dst):
    d.mkdir(parents=True, exist_ok=True)

# images: merge temp + partial final
for src_dir in (DST / "train" / "images",):
    if not src_dir.exists():
        continue
    for f in src_dir.glob("*"):
        out = img_dst / f.name
        if out.exists():
            continue
        shutil.copy2(f, out)

# labels: copy all from temp
for f in (DST / "train" / "labels").glob("*.txt"):
    shutil.copy2(f, lbl_dst / f.name)

# preview
prev_src = DST / "train" / "preview"
if prev_src.exists():
    for f in prev_src.glob("*"):
        out = prev_dst / f.name
        if not out.exists():
            shutil.copy2(f, out)

# manifest
mf = DST / "train" / "manifest.csv"
if mf.exists():
    shutil.copy2(mf, DST / "manifest_train.csv")

print("images", len(list(img_dst.glob("*"))))
print("labels", len(list(lbl_dst.glob("*.txt"))))
print("preview", len(list(prev_dst.glob("*"))) if prev_dst.exists() else 0)
