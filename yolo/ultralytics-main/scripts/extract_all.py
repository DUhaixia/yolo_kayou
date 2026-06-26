"""Batch-extract every .rar/.zip/.7z archive in a folder.

Each archive is extracted into its own sub-folder (named after the archive,
without extension) under --dst (defaults to the source folder).

Uses 7-Zip (``7z.exe``). By default it looks for the local copy at
``scripts/tools/7z.exe`` (set up by ``setup_7zip.ps1``) and then falls back to
any ``7z`` on PATH. The Windows built-in ``tar`` is NOT used because its rar
decoder is incomplete and corrupts many archives.

Usage (PowerShell):
    python scripts/extract_all.py --src "G:/BaiduNetdiskDownload/骑行图"
    python scripts/extract_all.py --src "G:/.../骑行图" --dst "G:/out" --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ARCHIVE_SUFFIXES = {".rar", ".zip", ".7z"}

# Secondary volumes of multi-part archives that must NOT be opened directly;
# opening the first volume extracts them all.
_SECONDARY_RE = re.compile(
    r"(\.part(?!0*1\b)\d+\.rar"      # .part2.rar, .part3.rar ... (keep .part1.rar)
    r"|\.r\d{2,}"                      # .r00, .r01, ...
    r"|\.z\d{2,}"                      # .z01, .z02, ... (split zip)
    r"|\.7z\.\d{3,})$",                # .7z.002, .7z.003 ...
    re.IGNORECASE,
)

_HERE = Path(__file__).resolve().parent


def find_7z() -> str:
    local = _HERE / "tools" / "7z.exe"
    if local.is_file():
        return str(local)
    exe = shutil.which("7z") or shutil.which("7za")
    if exe:
        return exe
    sys.exit(
        "ERROR: 7z.exe not found.\n"
        "Run first:  powershell -ExecutionPolicy Bypass -File scripts/setup_7zip.ps1"
    )


def is_archive(p: Path) -> bool:
    if not p.is_file():
        return False
    if _SECONDARY_RE.search(p.name):
        return False
    return p.suffix.lower() in ARCHIVE_SUFFIXES


def target_dir_for(archive: Path, dst_root: Path) -> Path:
    stem = re.sub(r"\.part0*1$", "", archive.stem, flags=re.IGNORECASE)
    return dst_root / stem


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def count_files(folder: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for p in folder.rglob("*"):
        if p.is_file():
            files += 1
            try:
                size += p.stat().st_size
            except OSError:
                pass
    return files, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing the archives")
    ap.add_argument("--dst", default=None, help="output root (default: same as --src)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract even if the target folder already exists and is non-empty")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        sys.exit(f"ERROR: source folder not found: {src}")
    dst_root = Path(args.dst) if args.dst else src
    dst_root.mkdir(parents=True, exist_ok=True)

    sevenzip = find_7z()
    print("Using 7z:", sevenzip)

    archives = sorted((p for p in src.iterdir() if is_archive(p)), key=lambda p: p.stat().st_size)
    if not archives:
        print("No .rar/.zip/.7z archives found in", src)
        return 0

    print(f"Found {len(archives)} archive(s). Output root: {dst_root}\n")

    ok, failed, skipped = [], [], []
    for i, arc in enumerate(archives, 1):
        target = target_dir_for(arc, dst_root)
        print(f"[{i}/{len(archives)}] {arc.name}  ({human(arc.stat().st_size)})")
        print(f"        -> {target}")

        if target.exists() and any(target.iterdir()) and not args.overwrite:
            print("        SKIP (folder exists and is non-empty; use --overwrite to force)\n")
            skipped.append(arc.name)
            continue

        target.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        # 7z x: extract with full paths; -o<dir> (no space); -y assume yes;
        # -aoa overwrite all existing files without prompt.
        proc = subprocess.run(
            [sevenzip, "x", str(arc), f"-o{target}", "-y", "-aoa"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        dt = time.time() - t0
        nfiles, nsize = count_files(target)

        # 7z exit codes: 0 = OK, 1 = warning (non-fatal), 2 = fatal error
        if proc.returncode in (0, 1):
            warn = " (with warnings)" if proc.returncode == 1 else ""
            print(f"        OK{warn}  {nfiles} files, {human(nsize)}, {dt:.1f}s\n")
            ok.append(arc.name)
        else:
            print(f"        FAILED (exit {proc.returncode}) after {dt:.1f}s; extracted {nfiles} files so far")
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for line in [l for l in out.splitlines() if l.strip()][-6:]:
                print("        !", line)
            print()
            failed.append(arc.name)

    print("=" * 60)
    print(f"DONE. ok={len(ok)}  failed={len(failed)}  skipped={len(skipped)}")
    if failed:
        print("\nFailed archives (encrypted / corrupt / need a password):")
        for n in failed:
            print("  -", n)
    if skipped:
        print("\nSkipped (already extracted):")
        for n in skipped:
            print("  -", n)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
