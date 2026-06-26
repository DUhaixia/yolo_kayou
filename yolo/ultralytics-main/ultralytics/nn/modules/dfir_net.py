# Ultralytics — lightweight DFIR-DETR modules (2512.07078v4.pdf)
# FIRC / FIRC3: frequency-domain iterative refinement; ANUP: amplitude-stable upsample (DFPN idea).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ("ANUP", "FGU", "FIRCLite", "FIRC3Lite")


def _prep_fft_mag(mag: torch.Tensor) -> torch.Tensor:
    """Log-compress + per-map scale so mag_refine BatchNorm does not explode running_var."""
    mag = torch.log1p(mag.float().clamp_min(0.0))
    scale = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (mag / scale).clamp(0.0, 10.0)


def _stabilize_bn_buffers(module: nn.Module) -> None:
    """Clamp BN running stats after FFT magnitude path (prevents NaN in running_var)."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.running_var is not None:
                m.running_var.data.clamp_(1e-6, 1e4)
                torch.nan_to_num(m.running_var, nan=1.0, posinf=1e4, neginf=1e-6, out=m.running_var.data)
            if m.running_mean is not None:
                torch.nan_to_num(m.running_mean, nan=0.0, posinf=0.0, neginf=0.0, out=m.running_mean.data)


class _MagRefineGN(nn.Module):
    """Magnitude refine without BatchNorm (GN only) — fixes v4 layer-25 running_var NaN on FFT mag."""

    def __init__(self, c: int) -> None:
        super().__init__()
        g = max(1, min(32, c))
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FIRCLite(nn.Module):
    """Lightweight FIRC (DFIR-DETR Alg.3): FFT magnitude refine + spatial residual, T steps."""

    def __init__(self, c: int, steps: int = 1) -> None:
        super().__init__()
        self.steps = max(1, int(steps))
        self.mag_refine = _MagRefineGN(c)
        self.spatial = Conv(c, c, 3)
        self.register_buffer("res_scale", torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        f = x
        capture = getattr(self, "capture_freq_viz", False)
        for _ in range(self.steps):
            # FFT/polar in fp32 only; Conv/BN use AMP dtype (weights stay fp16 under AMP).
            xf = torch.fft.fft2(f.float(), dim=(-2, -1))
            mag = torch.abs(xf).clamp(max=1e4)
            phase = torch.angle(xf)
            mag_in = _prep_fft_mag(mag).to(dtype=dtype)
            mag_refined = self.mag_refine(mag_in).clamp(0.0, 1e4)
            if self.training:
                _stabilize_bn_buffers(self.spatial)
            xf_new = torch.polar(mag_refined.float().clamp_min(1e-8), phase)
            f_new = torch.fft.ifft2(xf_new, dim=(-2, -1)).real
            f_new = torch.nan_to_num(f_new, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=dtype)
            residual = float(self.res_scale) * self.spatial(f_new)
            if capture:
                self._freq_viz = {
                    "spatial_in": f[:1].detach(),
                    "mag_in": mag[:1].detach(),
                    "mag_out": mag_refined[:1].detach(),
                    "spatial_residual": residual[:1].detach(),
                    "spatial_out": (f + residual)[:1].detach(),
                }
            f = f + residual
        return torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


class FIRC3Lite(nn.Module):
    """Lightweight FIRC3: dual-path CSP + cascaded FIRCLite (replaces RepC3 in neck for small objects)."""

    def __init__(self, c1: int, c2: int | None = None, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        c2 = c2 or c1
        h = max(8, int(c1 * e))
        self.cv1 = Conv(c1, 2 * h, 1)
        self.firc = FIRCLite(h, steps=n)
        self.cv2 = Conv(2 * h, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        a, b = y.chunk(2, dim=1)
        return self.cv2(torch.cat((self.firc(a), b), dim=1))


class FGU(nn.Module):
    """Frequency-Guided Unit: learnable high-frequency emphasis (DFIR freq-attention, lighter than full FIRC)."""

    def __init__(self, c: int, cutoff: float = 0.02) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.gate = nn.Sequential(Conv(c, c, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h, w = x.shape[-2:]
        xf = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing="ij",
        )
        cy, cx = h // 2, w // 2
        dist = torch.sqrt((xx - cx).float() ** 2 + (yy - cy).float() ** 2)
        mask = (dist > self.cutoff * max(h, w)).float()
        high = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(xf * mask, dim=(-2, -1)), dim=(-2, -1)))
        high = torch.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=dtype)
        gate = self.gate(high)
        out = x + gate * high
        if getattr(self, "capture_freq_viz", False):
            self._freq_viz = {
                "spatial_in": x[:1].detach(),
                "high_freq": high[:1].detach(),
                "gate": gate[:1].detach(),
                "spatial_out": out[:1].detach(),
            }
        return out


class ANUP(nn.Module):
    """Amplitude-normalized upsample (DFPN / ANUP): per-channel RMS norm, works at batch=1 inference."""

    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    @staticmethod
    def _amp_norm(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """DFPN-style amplitude scaling (scale only, avoids blow-up when var~0)."""
        scale = y.pow(2).mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(eps)
        return (y / scale).clamp(-50.0, 50.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self._amp_norm(y)
