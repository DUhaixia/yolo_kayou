# Ultralytics — DFIR-DETR (2512.07078v4.pdf)
# DCFA: Dynamic Content-Feature Aggregation backbone
# DFPN: ANUPNorm + DPSC neck components
# FIRC / FIRC3: frequency-domain iterative refinement (Eq. 13–16)

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv

__all__ = (
    "ANUPNorm",
    "DAFB",
    "DCFA",
    "DCFAStage",
    "DKSA",
    "DPSC",
    "FIRC",
    "FIRC3",
    "SGLU",
)


def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class SGLU(nn.Module):
    """Spatial Gated Linear Unit (Eq. 4)."""

    def __init__(self, c: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.split = nn.Conv2d(c, 2 * c, 1, bias=False)
        self.dw = DWConv(c, c, 3)
        self.act = nn.GELU()
        self.out = nn.Conv2d(c, c, 1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.split(x).chunk(2, dim=1)
        gate = self.act(self.dw(g))
        return x + self.drop(self.out(gate * v))


class DKSA(nn.Module):
    """Dynamic K-Sparse Attention (Eq. 5–7, Alg. 1)."""

    def __init__(self, c: int, min_k_ratio: float = 0.05, max_k_ratio: float = 1.0) -> None:
        super().__init__()
        g = max(1, min(32, c // 4))
        self.lgn = nn.GroupNorm(g, c)
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(c, c // 4, 3, padding=1, bias=False),
            nn.Conv2d(c // 4, 1, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        x_n = self.lgn(x)
        q, k, v = self.qkv(x_n).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        rho = torch.sigmoid(self.gate(x_n)).view(b, 1, 1)
        ratio = self.min_k_ratio + (self.max_k_ratio - self.min_k_ratio) * rho
        k_keep = max(1, min(n, int(ratio.item() * n)))

        scale = c ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        topk_vals, topk_idx = torch.topk(attn, k=k_keep, dim=-1)
        sparse = torch.full_like(attn, float("-inf"))
        sparse.scatter_(-1, topk_idx, topk_vals)
        attn = torch.softmax(sparse, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.proj(out)


class DAFB(nn.Module):
    """Dynamic Attention Fusion Block (Eq. 3)."""

    def __init__(self, c: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dw = DWConv(c, c, 3)
        self.dksa = DKSA(c)
        self.sg = SGLU(c, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.dw(x)
        return self.sg(h + self.dksa(h))


class DCFA(nn.Module):
    """Dynamic Content-Feature Aggregation (Eq. 1–2, Alg. 1)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, stride: int = 1) -> None:
        super().__init__()
        c_ = max(8, int(c2 * e))
        self.cv1 = Conv(c1, 2 * c_, 1, s=stride)
        self.m = nn.ModuleList(DAFB(c_) for _ in range(n))
        self.cv2 = Conv((2 + n) * c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        f1, f2 = y.chunk(2, dim=1)
        feats = [f1, f2]
        g = f2
        for block in self.m:
            g = block(g)
            feats.append(g)
        return self.cv2(torch.cat(feats, dim=1))


class DCFAStage(nn.Module):
    """ResNet-style stage using DCFA blocks (replaces ResNetLayer for DFIR-DETR backbone)."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 2, e: float = 0.5) -> None:
        super().__init__()
        self.is_first = is_first
        if is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            out_ch = c2 * 4
            blocks = [DCFA(c1, out_ch, n=1, e=e, stride=s)]
            blocks.extend(DCFA(out_ch, out_ch, n=1, e=e) for _ in range(n - 1))
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ANUPNorm(nn.Module):
    """Amplitude-normalized upsampling (DFPN / Eq. 8): F_up = (1/s^2) * Upsample(F)."""

    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.beta = 1.0 / (scale_factor**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return y * self.beta


class DPSC(nn.Module):
    """Dual-path shuffle convolution (DFPN / Eq. 10–12)."""

    def __init__(self, c: int) -> None:
        super().__init__()
        c2 = max(8, c // 2)
        self.path1 = Conv(c, c2, 3)
        self.path2_pw = Conv(c2, c2, 3)
        self.path2_dw = DWConv(c2, c2, 3)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.path1(x)
        f2 = self.act(self.path2_dw(self.act(self.path2_pw(f1))))
        return _channel_shuffle(torch.cat((f1, f2), dim=1))


class FIRC(nn.Module):
    """Frequency Iterative Refinement Convolution (Eq. 14–16)."""

    def __init__(self, c: int, s: int = 2, k: int = 3) -> None:
        super().__init__()
        self.s = s
        self.k = k
        self.weight = nn.Parameter(torch.randn(c, 1, k, k) * 0.02)
        self.bias = nn.Parameter(torch.tensor(9.0))

    @staticmethod
    def _zero_upsample(x: torch.Tensor, s: int) -> torch.Tensor:
        b, c, h, w = x.shape
        z = x.new_zeros(b, c, h * s, w * s)
        z[:, :, ::s, ::s] = x
        return z

    @staticmethod
    def _kernel_fft(weight: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        c, _, kh, kw = weight.shape
        h, w = shape
        pad = torch.zeros(c, 1, h, w, device=weight.device, dtype=weight.dtype)
        pad[:, :, :kh, :kw] = weight
        pad = torch.roll(pad, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
        return torch.fft.rfft2(pad)

    @staticmethod
    def _avg_s(spec: torch.Tensor, s: int) -> torch.Tensor:
        b, c, h, wf = spec.shape
        hs, ws = h // s, max(1, wf // s)
        spec = spec[:, :, : hs * s, : ws * s]
        spec = spec.view(b, c, hs, s, ws, s)
        return spec.mean(dim=(-1, -3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.s
        dtype = x.dtype
        eps = torch.sigmoid(self.bias - 9.0) + 1e-5

        up_z = self._zero_upsample(x.float(), s)
        up_n = F.interpolate(x.float(), scale_factor=s, mode="nearest")
        hs, ws = h * s, w * s

        k_fft = self._kernel_fft(self.weight, (hs, ws))
        fr = k_fft.conj() * torch.fft.rfft2(up_z) + torch.fft.rfft2(eps * up_n)
        wf = fr.shape[-1]

        denom = self._avg_s(k_fft.abs().square(), s).real + eps
        w_inv = self._avg_s((k_fft.conj() * fr).real, s) / denom

        w_up = F.interpolate(w_inv, size=(fr.shape[-2], wf), mode="bilinear", align_corners=False)
        corr = k_fft.conj() * w_up.to(fr.dtype)
        out_spec = fr - corr / eps
        out = torch.fft.irfft2(out_spec, s=(hs, ws)).to(dtype=dtype)

        if out.shape[-2:] != (h, w):
            out = F.adaptive_avg_pool2d(out, (h, w))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class FIRC3(nn.Module):
    """Frequency-domain Iterative Refinement C3 (Eq. 13, replaces RepC3 in DFIR-DETR neck)."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 0.5, s: int = 2) -> None:
        super().__init__()
        c_ = max(8, int(c2 * e))
        self.cv1 = Conv(c1, 2 * c_, 1)
        self.cv2 = Conv(c1, 2 * c_, 1)
        self.m = nn.Sequential(*(FIRC(c_, s=s) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
