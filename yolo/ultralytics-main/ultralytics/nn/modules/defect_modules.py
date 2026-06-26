# Ultralytics — lightweight modules for defect detect ablation (ECA / conv high-pass FreqGate).

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv

__all__ = ("ECA", "ECABlock", "ELA", "ELABlock", "FreqBranch", "FreqGate", "HighPassConv")


class ECA(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, c: int, k_size: int | None = None) -> None:
        super().__init__()
        if k_size is None:
            k_size = int(abs(math.log2(max(c, 2)) + 1) // 2 * 2 + 1)
            k_size = k_size if k_size % 2 else k_size + 1
            k_size = max(3, k_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.act(y)


class ECABlock(nn.Module):
    """YAML block: channel attention on a neck feature map."""

    def __init__(self, c1: int, k_size: int | None = None) -> None:
        super().__init__()
        self.eca = ECA(c1, k_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eca(x)


class ELA(nn.Module):
    """Efficient Local Attention (horizontal + vertical 1D pooling)."""

    def __init__(self, c: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, c // reduction)
        self.conv_h = nn.Conv1d(c, mid, 1, bias=False)
        self.conv_w = nn.Conv1d(c, mid, 1, bias=False)
        self.conv_out = nn.Conv1d(mid, c, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = self.conv_h(x.mean(dim=3)).view(b, -1, h, 1)
        x_w = self.conv_w(x.mean(dim=2)).view(b, -1, 1, w)
        attn = self.act(self.conv_out(x_h.squeeze(-1) + x_w.squeeze(-2)).view(b, c, 1, 1))
        return x * attn


class ELABlock(nn.Module):
    """YAML block: ELA on a neck feature map."""

    def __init__(self, c1: int, reduction: int = 4) -> None:
        super().__init__()
        self.ela = ELA(c1, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ela(x)


class HighPassConv(nn.Module):
    """Fixed Laplacian or Scharr high-pass (conv-only, ONNX-friendly)."""

    def __init__(self, c: int, mode: str = "laplacian") -> None:
        super().__init__()
        self.mode = mode
        if mode == "laplacian":
            k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            w = k.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
            self.register_buffer("weight", w)
        elif mode == "scharr":
            sx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
            sy = sx.t()
            self.register_buffer("sx", sx.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
            self.register_buffer("sy", sy.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
        else:
            raise ValueError(f"Unknown high-pass mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "laplacian":
            return F.conv2d(x, self.weight, padding=1, groups=x.shape[1])
        ex = F.conv2d(x, self.sx, padding=1, groups=x.shape[1])
        ey = F.conv2d(x, self.sy, padding=1, groups=x.shape[1])
        return torch.sqrt(ex**2 + ey**2 + 1e-6)


class FreqBranch(nn.Module):
    """Conv high-pass -> depthwise 3x3 -> 1x1 -> ECA."""

    def __init__(self, c_out: int, hp_mode: str = "laplacian") -> None:
        super().__init__()
        self.hp = HighPassConv(1, mode=hp_mode)
        self.dw = DWConv(1, 1, 3)
        self.pw = Conv(1, c_out, 1)
        self.eca = ECA(c_out)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        x = self.hp(gray)
        x = self.dw(x)
        x = self.pw(x)
        return self.eca(x)


class FreqGate(nn.Module):
    """Gate-fuse spatial neck feature with conv high-pass branch (no FFT, no aux loss).

    YAML: FreqGate, [hp_mode] with f=[input_layer, feat_layer] or single feat layer.
    """

    def __init__(self, c_feat: int, hp_mode: str = "laplacian", img_ch: int = 3) -> None:
        super().__init__()
        self.img_ch = img_ch
        self.freq = FreqBranch(c_feat, hp_mode=hp_mode)
        self.gate = nn.Sequential(Conv(c_feat * 2, c_feat, 1), nn.Sigmoid())

    @staticmethod
    def _to_gray(img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] == 1:
            return img
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    def forward(self, x: list | torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            img, feat = x[0], x[1]
            gray = self._to_gray(img)
            gray = F.interpolate(gray, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            freq = self.freq(gray)
        else:
            feat = x
            gray = feat.mean(1, keepdim=True)
            freq = self.freq(gray)
        gate = self.gate(torch.cat([feat, freq], dim=1))
        return feat + gate * freq
