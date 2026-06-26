from __future__ import annotations

import argparse
import itertools
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.defect_preprocess import apply_defect_preprocess, mixed_preprocess_torch


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build offline preprocessed dataset with same folder layout")
    parser.add_argument("--src", type=Path, default="M:/压印 - 副本/dataSet", help="Source dataset root directory")
    parser.add_argument("--dst", type=Path, default="G:/压印 - 副本/dataSet-mixed", help="Output dataset root directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["none", "point", "line", "mixed"],
        help="Defect preprocess mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove destination directory first if it exists",
    )
    parser.add_argument(
        "--keep-image-ext",
        action="store_true",
        help="Keep original image extension; otherwise save as .png",
    )
    parser.add_argument(
        "--copy-npy",
        action="store_true",
        help="Copy .npy/.npz files (default: skip them for speed and smaller dataset)",
    )
    parser.add_argument(
        "--preprocess-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Where to run mixed preprocess: cpu (threaded) or cuda (GPU). Only affects mixed.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Threads for read/encode/write/copy I/O")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for GPU mixed preprocess")
    return parser.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix.lower()
    if ext not in IMAGE_SUFFIXES:
        ext = ".png"
        path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def is_under_images_dir(rel: Path) -> bool:
    parts = [p.lower() for p in rel.parts]
    return len(parts) > 0 and parts[0] == "images"


def chunked(iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def _target_path(dst_file: Path, keep_image_ext: bool) -> Path:
    return dst_file if keep_image_ext else dst_file.with_suffix(".png")


def _process_image_cpu(src_file: Path, dst_file: Path, mode: str, keep_image_ext: bool) -> bool:
    """Read + CPU preprocess + write a single image. Returns True on success."""
    im = imread_unicode(src_file)
    if im is None:
        return False
    out = apply_defect_preprocess(im, mode=mode)
    return imwrite_unicode(_target_path(dst_file, keep_image_ext), out)


def build_dataset(
    src_root: Path,
    dst_root: Path,
    mode: str,
    keep_image_ext: bool,
    copy_npy: bool,
    preprocess_device: str,
    workers: int,
    batch: int,
) -> None:
    src_files = [p for p in src_root.rglob("*") if p.is_file()]

    image_jobs: list[tuple[Path, Path]] = []
    copy_jobs: list[tuple[Path, Path]] = []
    skipped_npy = 0
    for src_file in src_files:
        rel = src_file.relative_to(src_root)
        if not copy_npy and src_file.suffix.lower() in {".npy", ".npz"}:
            skipped_npy += 1
            continue
        dst_file = dst_root / rel
        if is_under_images_dir(rel) and src_file.suffix.lower() in IMAGE_SUFFIXES:
            image_jobs.append((src_file, dst_file))
        else:
            copy_jobs.append((src_file, dst_file))

    # Pre-create all destination directories once (avoids races inside worker threads).
    for _src, dst_file in itertools.chain(image_jobs, copy_jobs):
        dst_file.parent.mkdir(parents=True, exist_ok=True)

    processed_images = 0
    copied_files = 0
    failed_images: list[Path] = []

    workers = max(1, workers)
    use_gpu_mixed = preprocess_device == "cuda" and mode == "mixed"
    if preprocess_device == "cuda" and not use_gpu_mixed:
        print(f"Note: GPU preprocess only supports 'mixed'; falling back to CPU for '{mode}'.")

    total = len(image_jobs) + len(copy_jobs)
    done = 0

    def report() -> None:
        print(
            f"[{done}/{total}] processed_images={processed_images}, copied_files={copied_files}, "
            f"skipped_npy={skipped_npy}, failed_images={len(failed_images)}"
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Copy non-image files in parallel.
        for (src_file, _dst), ok in zip(copy_jobs, pool.map(lambda j: _copy_one(j), copy_jobs)):
            done += 1
            if ok:
                copied_files += 1
            if done % 200 == 0:
                report()

        if use_gpu_mixed:
            # Read in parallel (threads), preprocess on GPU (main thread), write in parallel (threads).
            torch_device = "cuda:0" if preprocess_device == "cuda" else "cpu"
            for batch_jobs in chunked(image_jobs, max(1, batch)):
                reads = list(pool.map(lambda j: (j[0], j[1], imread_unicode(j[0])), batch_jobs))
                write_jobs = []
                for src_file, dst_file, im in reads:
                    if im is None:
                        failed_images.append(src_file)
                        continue
                    out = mixed_preprocess_torch(im, device=torch_device)
                    write_jobs.append((src_file, _target_path(dst_file, keep_image_ext), out))
                for (src_file, _t, _o), ok in zip(write_jobs, pool.map(lambda w: imwrite_unicode(w[1], w[2]), write_jobs)):
                    if ok:
                        processed_images += 1
                    else:
                        failed_images.append(src_file)
                done += len(batch_jobs)
                report()
        else:
            # CPU preprocess fully parallel across threads.
            results = pool.map(lambda j: (j[0], _process_image_cpu(j[0], j[1], mode, keep_image_ext)), image_jobs)
            for src_file, ok in results:
                done += 1
                if ok:
                    processed_images += 1
                else:
                    failed_images.append(src_file)
                if done % 200 == 0:
                    report()

    report()
    print("\nDone.")
    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print(f"mode={mode}")
    print(f"preprocess_device={preprocess_device}")
    print(f"processed_images={processed_images}")
    print(f"copied_files={copied_files}")
    print(f"skipped_npy={skipped_npy}")
    print(f"failed_images={len(failed_images)}")
    if failed_images:
        print("Failed samples (first 20):")
        for p in failed_images[:20]:
            print(f"  - {p}")


def _copy_one(job: tuple[Path, Path]) -> bool:
    src_file, dst_file = job
    try:
        shutil.copy2(src_file, dst_file)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")

    if dst_root.exists():
        if args.overwrite:
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"Destination already exists: {dst_root}\n"
                "Use --overwrite to rebuild."
            )

    dst_root.mkdir(parents=True, exist_ok=True)
    build_dataset(
        src_root=src_root,
        dst_root=dst_root,
        mode=args.mode,
        keep_image_ext=args.keep_image_ext,
        copy_npy=args.copy_npy,
        preprocess_device=args.preprocess_device,
        workers=args.workers,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
