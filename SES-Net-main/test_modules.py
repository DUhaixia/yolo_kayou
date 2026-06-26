"""
SES-Net 模块本地测试脚本
测试 CSSG 与 ACFEE，支持本地图片或自动生成演示图（点 + 细线）

用法:
  python test_modules.py --image path/to/folder
  python test_modules.py --image img.bmp --preprocess gray3 --patch-size 16 --imgsz 1280
  # 灰度细线推荐: --preprocess gray3 --extractor fixed --patch-size 16 --feat-stride 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _stub(name: str) -> type:
    """占位类：源码 __init__ 引用但未在 forward 使用的依赖。"""

    class _M(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

    _M.__name__ = name
    return _M


def load_module_from_file(filepath: Path) -> dict:
    """不改动源码，通过注入命名空间加载 modules 下的 .py 文件。"""
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "CBAM": _stub("CBAM"),
        "HighFreqAttention": _stub("HighFreqAttention"),
        "LowFreqAttention": _stub("LowFreqAttention"),
    }
    code = filepath.read_text(encoding="utf-8")
    exec(compile(code, str(filepath), "exec"), namespace)  # noqa: S102
    return namespace


_cssg_ns = load_module_from_file(ROOT / "modules" / "CSSG.py")
_acfee_ns = load_module_from_file(ROOT / "modules" / "ACFEE.py")
CSSG = _cssg_ns["CSSG"]
FourierHighPassFilterWithAttention = _acfee_ns["FourierHighPassFilterWithAttention"]


class SimpleFeatureExtractor(nn.Module):
    """随机卷积模拟 backbone（未训练，细线场景效果差）。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedEdgeFeatureExtractor(nn.Module):
    """固定边缘/灰度多通道特征，适合灰度细线缺陷的结构验证（无需训练权重）。"""

    def __init__(self, out_ch: int = 64) -> None:
        super().__init__()
        self.out_ch = out_ch
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        kl = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("kl", kl)

    def _gray_edges(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        lap = torch.abs(F.conv2d(gray, self.kl, padding=1))
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + 1e-6)
        gray_n = (gray - gray.amin(dim=(2, 3), keepdim=True)) / (
            gray.amax(dim=(2, 3), keepdim=True) - gray.amin(dim=(2, 3), keepdim=True) + 1e-6
        )
        return torch.cat([gray_n, mag, lap, gx.abs(), gy.abs()], dim=1)  # [B, 5, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gray_edges(x)
        feats = [base]
        cur = base
        for _ in range(3):
            cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
            feats.append(cur)
        feat = F.interpolate(feats[-1], size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
        for f in feats[:-1]:
            up = F.interpolate(f, size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
            feat = torch.cat([feat, up], dim=1)
        if feat.shape[1] < self.out_ch:
            repeat = (self.out_ch + feat.shape[1] - 1) // feat.shape[1]
            feat = feat.repeat(1, repeat, 1, 1)[:, : self.out_ch]
        else:
            feat = feat[:, : self.out_ch]
        return feat


def contrast_stretch(gray: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    p_lo, p_hi = np.percentile(gray, (low, high))
    return np.clip((gray - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0).astype(np.float32)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    try:
        import cv2
        g8 = (gray * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(g8).astype(np.float32) / 255.0
    except ImportError:
        return contrast_stretch(gray)


def numpy_gray_to_edge_channels(gray: np.ndarray) -> np.ndarray:
    """gray -> [gray, sobel, laplacian] 三通道，突出细线。"""
    g = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    kl = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)[0, 0].numpy()
    lap = torch.abs(F.conv2d(g, kl, padding=1))[0, 0].numpy()
    mag = mag / (mag.max() + 1e-6)
    lap = lap / (lap.max() + 1e-6)
    return np.stack([gray, mag, lap], axis=0)


def load_image(path: Path, imgsz: int, preprocess: str = "gray3") -> torch.Tensor:
    """
    加载图像 -> [1, 3, H, W]
    preprocess:
      rgb      - 直接转 RGB（灰度图三通道相同，细线易被淹没）
      gray3    - 灰度 + Sobel + Laplacian（推荐用于细线）
      clahe    - CLAHE 增强后再 gray3
    """
    pil = Image.open(path)
    is_gray = pil.mode in ("L", "I", "I;16", "F")
    gray = np.asarray(pil.convert("L").resize((imgsz, imgsz), Image.LANCZOS), dtype=np.float32) / 255.0
    gray = contrast_stretch(gray)

    if preprocess == "rgb" and not is_gray and pil.mode == "RGB":
        arr = np.asarray(pil.convert("RGB").resize((imgsz, imgsz), Image.LANCZOS), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor

    if preprocess == "clahe":
        gray = apply_clahe(gray)

    if preprocess in ("gray3", "clahe"):
        arr = numpy_gray_to_edge_channels(gray)
    elif preprocess == "rgb":
        arr = np.stack([gray, gray, gray], axis=0)
    else:
        raise ValueError(f"unknown preprocess: {preprocess}")

    return torch.from_numpy(arr).unsqueeze(0).float()


def collect_image_paths(inputs: list[str]) -> list[Path]:
    """支持：多个文件路径、文件夹（递归扫描）、通配符。"""
    paths: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            candidates = [p]
        elif p.is_dir():
            candidates = [
                x for x in sorted(p.rglob("*"))
                if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
            ]
        elif any(ch in raw for ch in "*?[]"):
            candidates = [
                x for x in sorted(p.parent.glob(p.name))
                if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            raise FileNotFoundError(f"Path not found: {p}")

        if not candidates:
            raise FileNotFoundError(f"No images found under: {p}")

        for c in candidates:
            resolved = c.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(c)

    if not paths:
        raise FileNotFoundError("No valid image files collected.")
    return paths


def make_demo_image(imgsz: int = 640) -> torch.Tensor:
    """生成带点与细线的演示图，便于无本地数据时快速测试。"""
    canvas = np.ones((imgsz, imgsz, 3), dtype=np.float32) * 0.92

    # 细线
    canvas[120:122, 80:560] = [0.15, 0.15, 0.15]
    canvas[300:560, 410:412] = [0.2, 0.1, 0.1]
    canvas[200:205, 200:450] = [0.1, 0.15, 0.2]

    # 稀疏小点
    points = [(100, 500), (250, 180), (480, 320), (520, 90), (360, 520)]
    for y, x in points:
        canvas[y - 2 : y + 3, x - 2 : x + 3] = [0.05, 0.05, 0.05]

    tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x[:3]
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x.permute(1, 2, 0).numpy()


def visualize_cssg(
    img: torch.Tensor,
    feat: torch.Tensor,
    attentions: torch.Tensor,
    patch_size: int,
    save_path: Path,
    gray_bg: np.ndarray | None = None,
) -> None:
    """可视化 CSSG 跨尺度注意力，叠加到灰度底图上便于看细线。"""
    _, _, h, w = img.shape
    _, _, hf, wf = feat.shape
    n_high_h, n_high_w = h // patch_size, w // patch_size
    n_low_h, n_low_w = hf // patch_size, wf // patch_size

    attn = attentions.mean(dim=1)[0].softmax(dim=-1)  # [N_high, N_low]
    attn_high = attn.max(dim=-1).values.reshape(n_high_h, n_high_w).cpu().numpy()
    attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean().item()

    attn_low_idx = attn.argmax(dim=-1).reshape(n_high_h, n_high_w).cpu().numpy()
    attn_low_map = np.zeros((n_low_h, n_low_w), dtype=np.float32)
    for i in range(n_high_h):
        for j in range(n_high_w):
            idx = int(attn_low_idx[i, j])
            ly, lx = divmod(idx, n_low_w)
            attn_low_map[ly, lx] += 1.0
    if attn_low_map.max() > 0:
        attn_low_map /= attn_low_map.max()

    attn_high_up = np.kron(attn_high, np.ones((patch_size, patch_size)))
    attn_low_up = np.kron(attn_low_map, np.ones((patch_size, patch_size)))

    if gray_bg is None:
        gray_bg = img[0, 0].detach().cpu().numpy()
    edge_ch = img[0, 1].detach().cpu().numpy() if img.shape[1] > 1 else gray_bg
    feat_vis = feat[0].mean(0).detach().cpu().numpy()
    feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(gray_bg, cmap="gray")
    axes[0, 0].set_title("Gray (contrast stretched)")
    axes[0, 1].imshow(edge_ch, cmap="gray")
    axes[0, 1].set_title("Edge channel (Sobel)")
    axes[0, 2].imshow(feat_vis, cmap="magma")
    axes[0, 2].set_title(f"Feature mean ({hf}x{wf}, N_low={n_low_h}x{n_low_w})")

    axes[1, 0].imshow(gray_bg, cmap="gray")
    axes[1, 0].imshow(attn_high_up, cmap="hot", alpha=0.55, vmin=attn_high.min(), vmax=attn_high.max())
    axes[1, 0].set_title("Attention overlay (image patches)")

    axes[1, 1].imshow(gray_bg, cmap="gray")
    axes[1, 1].imshow(attn_low_up, cmap="viridis", alpha=0.55)
    axes[1, 1].set_title("Attention overlay (feature patches)")

    im = axes[1, 2].imshow(attn_high, cmap="hot")
    axes[1, 2].set_title(f"Patch attention grid ({n_high_h}x{n_high_w})")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"CSSG | patch={patch_size} | N_high={n_high_h}x{n_high_w} N_low={n_low_h}x{n_low_w} | "
        f"entropy={attn_entropy:.3f} (lower=more focused)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CSSG] saved -> {save_path}  (attn entropy={attn_entropy:.3f})")


def sobel_edge(x: torch.Tensor) -> torch.Tensor:
    """简易 Sobel 边缘幅值，用于 ACFEE 输出可视化。"""
    c = x.shape[1]
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).mean(dim=1, keepdim=True)


def visualize_acfee(
    feat: torch.Tensor,
    enhanced: torch.Tensor,
    save_path: Path,
) -> None:
    edge_in = sobel_edge(feat)
    edge_out = sobel_edge(enhanced)
    diff = (enhanced - feat).abs().mean(dim=1, keepdim=True)

    panels = [
        (tensor_to_rgb(feat), "Input Feature (channel mean)"),
        (tensor_to_rgb(enhanced), "ACFEE Output"),
        (edge_in[0, 0].cpu().numpy(), "Sobel Edge (before)"),
        (edge_out[0, 0].cpu().numpy(), "Sobel Edge (after ACFEE)"),
        (diff[0, 0].cpu().numpy(), "Abs Diff (single channel)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (data, title) in zip(axes.ravel(), panels):
        if data.ndim == 2:
            im = ax.imshow(data, cmap="gray")
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    axes[1, 2].axis("off")
    fig.suptitle("ACFEE Test (FourierHighPassFilterWithAttention)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ACFEE] saved -> {save_path}")


@torch.no_grad()
def process_one_image(
    img: torch.Tensor,
    name: str,
    item_dir: Path,
    device: torch.device,
    extractor: nn.Module,
    cssg: nn.Module,
    acfee: nn.Module,
    feat_sz: int,
    patch_size: int,
    gray_bg: np.ndarray | None = None,
) -> None:
    img = img.to(device)
    feat = extractor(img)
    if feat.shape[-2] != feat_sz or feat.shape[-1] != feat_sz:
        feat = F.interpolate(feat, size=(feat_sz, feat_sz), mode="bilinear", align_corners=False)

    item_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print(f"Processing  : {name}")
    print(f"Image shape : {tuple(img.shape)}")
    print(f"Feat shape  : {tuple(feat.shape)}")
    print(f"N_high      : {(img.shape[2] // patch_size) * (img.shape[3] // patch_size)}")
    print(f"N_low       : {(feat_sz // patch_size) ** 2}")
    print(f"Output dir  : {item_dir}")

    if gray_bg is None:
        gray_bg = img[0, 0].cpu().numpy()
    plt.imsave(item_dir / "input_gray.png", gray_bg, cmap="gray")
    plt.imsave(item_dir / "input_edge.png", img[0, 1].cpu().numpy(), cmap="gray")

    attentions = cssg(img, feat)
    print(f"[CSSG] output shape: {tuple(attentions.shape)}")
    visualize_cssg(
        img, feat, attentions, patch_size,
        save_path=item_dir / "cssg_result.png",
        gray_bg=gray_bg,
    )

    enhanced = acfee(feat)
    print(f"[ACFEE] output shape: {tuple(enhanced.shape)}")
    visualize_acfee(feat, enhanced, save_path=item_dir / "acfee_result.png")


@torch.no_grad()
def run_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = args.patch_size
    imgsz = (args.imgsz // patch_size) * patch_size
    feat_sz = imgsz // args.feat_stride
    feat_sz = (feat_sz // patch_size) * patch_size
    if feat_sz < patch_size:
        raise ValueError(f"feat size too small ({feat_sz}), try larger --imgsz or smaller --patch-size")

    if args.image:
        image_paths = collect_image_paths(args.image)
    else:
        image_paths = []

    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Preprocess  : {args.preprocess}")
    print(f"Extractor   : {args.extractor}")
    print(f"Patch size  : {patch_size}")
    print(f"Feat stride : {args.feat_stride}")
    print(f"Image size  : {imgsz}")
    print(f"Feat size   : {feat_sz}")
    n_images = len(image_paths) if image_paths else 1
    print(f"Images      : {n_images}{' (demo)' if not image_paths else ''}")
    print("=" * 60)

    if args.extractor == "fixed":
        extractor = FixedEdgeFeatureExtractor(out_ch=args.feat_ch).to(device).eval()
    else:
        extractor = SimpleFeatureExtractor(out_ch=args.feat_ch).to(device).eval()
    cssg = CSSG(in_channels=(3, args.feat_ch), patch_size=patch_size).to(device).eval()
    acfee = FourierHighPassFilterWithAttention(
        channel=args.feat_ch,
        cutoff_frequency=args.cutoff,
        use_attention=True,
        attention_type="cross",
    ).to(device).eval()

    if not image_paths:
        demo_dir = out_dir / "demo"
        demo_img = make_demo_image(imgsz)
        print(f"[INFO] no --image provided, using built-in demo")
        process_one_image(
            img=demo_img,
            name="demo",
            item_dir=demo_dir,
            device=device,
            extractor=extractor,
            cssg=cssg,
            acfee=acfee,
            feat_sz=feat_sz,
            patch_size=patch_size,
        )
        print("=" * 60)
        print(f"Done. Results in: {demo_dir.resolve()}")
        return

    ok, fail = 0, 0
    for img_path in image_paths:
        try:
            img = load_image(img_path, imgsz, preprocess=args.preprocess)
            gray_bg = img[0, 0].numpy()
            process_one_image(
                img=img,
                name=img_path.name,
                item_dir=out_dir / img_path.stem,
                device=device,
                extractor=extractor,
                cssg=cssg,
                acfee=acfee,
                feat_sz=feat_sz,
                patch_size=patch_size,
                gray_bg=gray_bg,
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERROR] {img_path.name}: {e}")

    print("=" * 60)
    print(f"Done. success={ok}, failed={fail}")
    print(f"Results in: {out_dir.resolve()}")
    print("  each image -> {output}/{image_stem}/cssg_result.png, acfee_result.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CSSG and ACFEE modules on local images")
    parser.add_argument(
        "--image", "-i",
        nargs="+",
        default=[],
        help="one or more image paths, folders, or glob patterns",
    )
    parser.add_argument("--output", "-o", type=str, default="test_outputs", help="output directory")
    parser.add_argument("--imgsz", type=int, default=1280, help="input size (multiple of patch_size)")
    parser.add_argument("--patch-size", type=int, default=16, help="CSSG patch size (细线建议 8~16)")
    parser.add_argument("--feat-stride", type=int, default=4, help="特征图相对原图下采样倍数 (越小 patch 越细)")
    parser.add_argument(
        "--preprocess",
        choices=["gray3", "clahe", "rgb"],
        default="gray3",
        help="gray3=灰度+Sobel+Laplacian(推荐细线); clahe=对比度增强; rgb=直接RGB",
    )
    parser.add_argument(
        "--extractor",
        choices=["fixed", "random"],
        default="fixed",
        help="fixed=固定边缘特征(推荐); random=随机CNN(未训练效果差)",
    )
    parser.add_argument("--feat-ch", type=int, default=64, help="feature channels")
    parser.add_argument("--cutoff", type=float, default=0.01, help="ACFEE frequency cutoff")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run_test(parse_args())
