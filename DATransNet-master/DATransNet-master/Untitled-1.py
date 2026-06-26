"""Defect-oriented 3-channel preprocessing for industrial images."""

from __future__ import annotations

import cv2
import numpy as nps


def _percentile_boost(src: np.ndarray, lo_p: float = 2.0, hi_p: float = 99.5, gamma: float = 0.7) -> np.ndarray:
    """Contrast boost for sparse response maps."""
    s = src.astype(np.uint8)
    nz = s[s > 0]
    if nz.size == 0:
        return s
    lo = np.percentile(nz, lo_p)
    hi = np.percentile(nz, hi_p)
    if hi <= lo:
        hi = lo + 1.0
    v = (s.astype(np.float32) - lo) / (hi - lo)
    v = np.clip(v, 0.0, 1.0)
    v = np.power(v, gamma)
    return np.clip(v * 255.0, 0, 255).astype(np.uint8)


def _suppress_texture(src: np.ndarray, ksize: int = 13) -> np.ndarray:
    """Suppress dense background texture while keeping thin structures."""
    ksize = max(3, ksize | 1)
    smooth = cv2.GaussianBlur(src, (ksize, ksize), 0)
    return cv2.subtract(src, smooth)


def _to_gray(im: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 grayscale."""
    if im.ndim == 2:
        gray = im
    elif im.shape[2] == 1:
        gray = im[..., 0]
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE enhancement."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(gray)


def _dog_response(gray: np.ndarray, sigma1: float = 1.2, sigma2: float = 3.2) -> np.ndarray:
    """Difference-of-Gaussians response."""
    k1 = max(3, int(round(sigma1 * 6 + 1)) | 1)
    k2 = max(3, int(round(sigma2 * 6 + 1)) | 1)
    g1 = cv2.GaussianBlur(gray, (k1, k1), sigma1)
    g2 = cv2.GaussianBlur(gray, (k2, k2), sigma2)
    return cv2.subtract(g1, g2)


def _scharr_mag(gray: np.ndarray) -> np.ndarray:
    """Scharr gradient magnitude."""
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)


def _line_kernel(length: int, orientation: str) -> np.ndarray:
    """Create normalized line kernel by orientation."""
    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    if orientation == "h":
        k[c, :] = 1.0
    elif orientation == "v":
        k[:, c] = 1.0
    elif orientation == "d1":
        for i in range(length):
            k[i, i] = 1.0
    elif orientation == "d2":
        for i in range(length):
            k[i, length - 1 - i] = 1.0
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    return k / np.sum(k)


def _shift_image(src: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift image while preserving size."""
    h, w = src.shape[:2]
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _line_responses_symmetric(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-orientation symmetric bright/dark line responses (no template subtraction).

    This is more robust than simple local background subtraction for thin line defects.
    """
    g = gray.astype(np.float32)
    dirs = [
        ("h", (0.0, 1.0)),
        ("v", (1.0, 0.0)),
        ("d1", (0.7071, -0.7071)),
        ("d2", (0.7071, 0.7071)),
    ]
    lengths = [9, 15]
    widths = [2, 4]
    eps = 6.0
    best_bright = np.zeros_like(g, dtype=np.float32)
    best_dark = np.zeros_like(g, dtype=np.float32)

    for ori, perp in dirs:
        for ln in lengths:
            k = _line_kernel(ln, ori)
            center = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
            for w in widths:
                dx, dy = perp[0] * w, perp[1] * w
                side1 = _shift_image(center, dx, dy)
                side2 = _shift_image(center, -dx, -dy)
                bg = 0.5 * (side1 + side2)
                signed = center - bg
                contrast = np.abs(signed)
                side_diff = np.abs(side1 - side2)
                symmetry = 1.0 - side_diff / (contrast + side_diff + eps)
                symmetry = np.clip(symmetry, 0.0, 1.0)
                response = contrast * symmetry
                bright = np.where(signed > 0, response, 0.0)
                dark = np.where(signed < 0, response, 0.0)
                best_bright = np.maximum(best_bright, bright)
                best_dark = np.maximum(best_dark, dark)

    bright_u8 = cv2.normalize(best_bright, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dark_u8 = cv2.normalize(best_dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bright_u8 = _percentile_boost(bright_u8, lo_p=2.0, hi_p=99.5, gamma=0.65)
    dark_u8 = _suppress_texture(dark_u8, ksize=13)
    dark_u8 = _percentile_boost(dark_u8, lo_p=1.0, hi_p=99.7, gamma=0.50)
    return bright_u8, dark_u8


def point_preprocess(im: np.ndarray) -> np.ndarray:
    """
    Point defect 3-channel enhancement.

    Channels:
    - R/BGR[2]: gray
    - G/BGR[1]: clahe(gray)
    - B/BGR[0]: scharr(gray) + dog(gray)
    """
    gray = _to_gray(im)
    clahe_im = _clahe(gray)
    scharr = _scharr_mag(gray)
    dog = _dog_response(gray)
    highfreq = cv2.addWeighted(scharr, 0.7, dog, 0.3, 0.0)
    return cv2.merge((highfreq, clahe_im, gray))


def line_preprocess(im: np.ndarray) -> np.ndarray:
    """
    Line defect 3-channel enhancement (no template difference).

    Channels:
    - R/BGR[2]: gray
    - G/BGR[1]: bright line response
    - B/BGR[0]: dark line response
    """
    gray = _to_gray(im)
    bright, dark = _line_responses_symmetric(gray)
    return cv2.merge((dark, bright, gray))


def mixed_preprocess(im: np.ndarray) -> np.ndarray:
    """
    Mixed Point+Line 3-channel enhancement.

    Keep line channels fixed to symmetric bright/dark, and inject point high-frequency
    into both channels to support joint Point+Line learning on the same image.

    Channels:
    - R/BGR[2]: gray
    - G/BGR[1]: bright_symmetric + point_highfreq
    - B/BGR[0]: dark_symmetric + point_highfreq
    """
    gray = _to_gray(im)
    bright, dark = _line_responses_symmetric(gray)
    scharr = _scharr_mag(gray)
    dog = _dog_response(gray)
    point_highfreq = cv2.addWeighted(scharr, 0.7, dog, 0.3, 0.0)
    # Keep dark-line channel almost pure to avoid texture pollution in B.
    g_ch = cv2.addWeighted(bright, 0.82, point_highfreq, 0.18, 0.0)
    b_ch = cv2.addWeighted(dark, 0.97, point_highfreq, 0.03, 0.0)
    return cv2.merge((b_ch, g_ch, gray))


def apply_defect_preprocess(im: np.ndarray, mode: str = "none") -> np.ndarray:
    """Apply defect preprocess by mode: none/point/line/mixed."""
    mode = (mode or "none").lower()
    if mode == "none":
        return im
    if mode == "point":
        return point_preprocess(im)
    if mode == "line":
        return line_preprocess(im)
    if mode == "mixed":
        return mixed_preprocess(im)
    return im


# ---------------------------------------------------------------------------
# GPU (PyTorch) implementation of mixed_preprocess.
#
# This mirrors the CPU mixed_preprocess math using torch ops on CUDA:
#   filter2D      -> F.conv2d (replicate pad)
#   Scharr        -> fixed 3x3 conv2d (reflect pad)
#   GaussianBlur  -> separable conv2d (reflect pad)
#   warpAffine    -> F.grid_sample (bilinear, border)
#   percentile    -> torch.quantile
# It is "close" but not bit-exact to OpenCV (border/rounding differ slightly).
# Use scripts/verify_mixed_gpu.py to check the max/mean diff on your images.
# ---------------------------------------------------------------------------

_TORCH_KERNEL_CACHE: dict = {}


def _gaussian_kernel1d(ksize: int, sigma: float) -> np.ndarray:
    """Mirror cv2.getGaussianKernel (including its default sigma rule)."""
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    k = np.exp(-(ax**2) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= k.sum()
    return k


def _line_kernel_2d(length: int, orientation: str) -> np.ndarray:
    """Same normalized line kernel as CPU _line_kernel, as a 2D float32 array."""
    return _line_kernel(length, orientation)


def mixed_preprocess_torch(im: np.ndarray, device: str = "cuda") -> np.ndarray:
    """GPU mixed preprocess for a single BGR uint8 image. Returns BGR uint8 ndarray.

    Falls back to the CPU implementation if torch/CUDA is unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return mixed_preprocess(im)

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return mixed_preprocess(im)

    # Grayscale (mirror cv2 BGR2GRAY weights).
    if im.ndim == 2:
        gray_np = im.astype(np.float32)
    elif im.shape[2] == 1:
        gray_np = im[..., 0].astype(np.float32)
    else:
        b = im[..., 0].astype(np.float32)
        g_ = im[..., 1].astype(np.float32)
        r = im[..., 2].astype(np.float32)
        gray_np = np.clip(np.round(0.299 * r + 0.587 * g_ + 0.114 * b), 0, 255)

    H, W = gray_np.shape[:2]
    g = torch.from_numpy(gray_np).to(dev).float()[None, None]

    cache_key = (dev.type, str(dev.index))
    kernels = _TORCH_KERNEL_CACHE.get(cache_key)
    if kernels is None:
        dirs = [("h", (0.0, 1.0)), ("v", (1.0, 0.0)), ("d1", (0.7071, -0.7071)), ("d2", (0.7071, 0.7071))]
        line_w = {}
        for ori, _perp in dirs:
            for ln in (9, 15):
                w = torch.from_numpy(_line_kernel_2d(ln, ori)).to(dev).float()[None, None]
                line_w[(ori, ln)] = w
        scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], device=dev)[None, None]
        scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], device=dev)[None, None]
        gk9 = torch.from_numpy(_gaussian_kernel1d(9, 1.2)).to(dev).float()
        gk21 = torch.from_numpy(_gaussian_kernel1d(21, 3.2)).to(dev).float()
        gk13 = torch.from_numpy(_gaussian_kernel1d(13, 0.0)).to(dev).float()
        kernels = {"dirs": dirs, "line": line_w, "sx": scharr_x, "sy": scharr_y, "gk9": gk9, "gk21": gk21, "gk13": gk13}
        _TORCH_KERNEL_CACHE[cache_key] = kernels

    def conv_replicate(x, weight):
        ph, pw = weight.shape[-2] // 2, weight.shape[-1] // 2
        return F.conv2d(F.pad(x, (pw, pw, ph, ph), mode="replicate"), weight)

    def conv_reflect(x, weight):
        ph, pw = weight.shape[-2] // 2, weight.shape[-1] // 2
        return F.conv2d(F.pad(x, (pw, pw, ph, ph), mode="reflect"), weight)

    def gaussian_reflect(x, k1d):
        ksize = k1d.numel()
        p = ksize // 2
        kx = k1d.view(1, 1, 1, ksize)
        ky = k1d.view(1, 1, ksize, 1)
        x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), kx)
        x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), ky)
        return x

    ys, xs = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    denom_x = max(W - 1, 1)
    denom_y = max(H - 1, 1)

    def shift(t, dx, dy):
        gx = (xs - dx) / denom_x * 2.0 - 1.0
        gy = (ys - dy) / denom_y * 2.0 - 1.0
        grid = torch.stack((gx, gy), dim=-1)[None]
        return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)

    eps = 6.0
    best_bright = torch.zeros_like(g)
    best_dark = torch.zeros_like(g)
    for ori, perp in kernels["dirs"]:
        for ln in (9, 15):
            center = conv_replicate(g, kernels["line"][(ori, ln)])
            for wdt in (2, 4):
                dx, dy = perp[0] * wdt, perp[1] * wdt
                s1 = shift(center, dx, dy)
                s2 = shift(center, -dx, -dy)
                bg = 0.5 * (s1 + s2)
                signed = center - bg
                contrast = signed.abs()
                side_diff = (s1 - s2).abs()
                symmetry = (1.0 - side_diff / (contrast + side_diff + eps)).clamp(0.0, 1.0)
                response = contrast * symmetry
                best_bright = torch.maximum(best_bright, torch.where(signed > 0, response, torch.zeros_like(response)))
                best_dark = torch.maximum(best_dark, torch.where(signed < 0, response, torch.zeros_like(response)))

    def norm_minmax_u8(t):
        mn, mx = t.min(), t.max()
        if (mx - mn) <= 0:
            return torch.zeros_like(t)
        return ((t - mn) * (255.0 / (mx - mn))).clamp(0, 255).floor()

    def percentile_boost(s, lo_p, hi_p, gamma):
        nz = s[s > 0]
        if nz.numel() == 0:
            return s
        lo = torch.quantile(nz, lo_p / 100.0)
        hi = torch.quantile(nz, hi_p / 100.0)
        if hi <= lo:
            hi = lo + 1.0
        v = ((s - lo) / (hi - lo)).clamp(0.0, 1.0).pow(gamma)
        return (v * 255.0).clamp(0, 255).floor()

    bright_u8 = percentile_boost(norm_minmax_u8(best_bright), 2.0, 99.5, 0.65)
    dark_u8 = norm_minmax_u8(best_dark)
    smooth = gaussian_reflect(dark_u8, kernels["gk13"]).round()
    dark_u8 = (dark_u8 - smooth).clamp(0, 255)
    dark_u8 = percentile_boost(dark_u8, 1.0, 99.7, 0.50)

    gx = conv_reflect(g, kernels["sx"])
    gy = conv_reflect(g, kernels["sy"])
    scharr_u8 = torch.sqrt(gx * gx + gy * gy).round().clamp(0, 255)
    g1 = gaussian_reflect(g, kernels["gk9"]).round()
    g2 = gaussian_reflect(g, kernels["gk21"]).round()
    dog_u8 = (g1 - g2).clamp(0, 255)
    point_highfreq = (0.7 * scharr_u8 + 0.3 * dog_u8).round().clamp(0, 255)

    g_ch = (0.82 * bright_u8 + 0.18 * point_highfreq).round().clamp(0, 255)
    b_ch = (0.97 * dark_u8 + 0.03 * point_highfreq).round().clamp(0, 255)

    out = torch.stack((b_ch[0, 0], g_ch[0, 0], g[0, 0]), dim=-1).clamp(0, 255).to(torch.uint8)
    return out.cpu().numpy()


def mixed_preprocess_batch_torch(images: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    """Apply GPU mixed preprocess to a list of BGR uint8 images (sizes may differ)."""
    return [mixed_preprocess_torch(im, device=device) for im in images]
