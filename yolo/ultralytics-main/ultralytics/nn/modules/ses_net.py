# Ultralytics — SES-Net modules (ACFEE + CSSG) for sparse small defect detection/segmentation.
# Adapted from SES-Net-main/modules (Fourier + cross-scale patch attention).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.dfir_net import FIRC3Lite

__all__ = (
    "ACFEE",
    "ACFEEBlock",
    "CSSG",
    "SESInputCache",
    "SESFusion",
    "SmallDefectEnhance",
    "SobelConv",
    "build_golden_attention_from_bbox_by_class",
    "collect_ses_aux_loss",
    "set_ses_targets",
)


def _clamp_patch_range(x1: int, x2: int, y1: int, y2: int, max_x: int, max_y: int) -> tuple[int, int, int, int]:
    """Clamp patch grid indices to valid [0, max] (bbox float noise can exceed grid)."""
    x1, x2 = max(0, min(int(x1), max_x)), max(0, min(int(x2), max_x))
    y1, y2 = max(0, min(int(y1), max_y)), max(0, min(int(y2), max_y))
    if x2 < x1:
        x1 = x2
    if y2 < y1:
        y1 = y2
    return x1, x2, y1, y2


def _expand_patch_range(x1: int, x2: int, y1: int, y2: int, min_span: int, max_x: int, max_y: int) -> tuple[int, int, int, int]:
    """Ensure tiny bboxes cover at least min_span patches (helps point defects)."""
    x1, x2, y1, y2 = _clamp_patch_range(x1, x2, y1, y2, max_x, max_y)
    if x2 - x1 < min_span:
        cx, half = (x1 + x2) // 2, min_span // 2
        x1, x2 = max(0, cx - half), min(max_x, cx - half + min_span)
    if y2 - y1 < min_span:
        cy, half = (y1 + y2) // 2, min_span // 2
        y1, y2 = max(0, cy - half), min(max_y, cy - half + min_span)
    return _clamp_patch_range(x1, x2, y1, y2, max_x, max_y)


def build_golden_attention_from_bbox_by_class(
    targets, img_shape, feat_shape, patch_size, device, B, min_patch_span: int = 1
):
    """Build patch-level supervision map for CSSG (same-class high/low patch links)."""
    cls_all = targets["cls"].view(-1)
    bbox_all = targets["bboxes"]
    batch_idx = targets["batch_idx"]

    H, W = img_shape
    Hf, Wf = feat_shape
    n_high = (H // patch_size) * (W // patch_size)
    n_low = (Hf // patch_size) * (Wf // patch_size)
    golden_map = torch.zeros((B, 1, n_high, n_low), device=device)

    for b in range(B):
        idxs = (batch_idx == b).nonzero(as_tuple=True)[0]
        cls_b = cls_all[idxs]
        bbox_b = bbox_all[idxs]
        num_objs = len(cls_b)
        patch_groups_high, patch_groups_low = [], []

        for i in range(num_objs):
            cls_id = int(cls_b[i].item())
            cx, cy, bw, bh = bbox_b[i]
            cx_img, cy_img, bw_img, bh_img = cx * W, cy * H, bw * W, bh * H
            x1, x2 = int((cx_img - bw_img / 2) // patch_size), int((cx_img + bw_img / 2) // patch_size)
            y1, y2 = int((cy_img - bh_img / 2) // patch_size), int((cy_img + bh_img / 2) // patch_size)
            max_px, max_py = W // patch_size - 1, H // patch_size - 1
            x1, x2, y1, y2 = _expand_patch_range(x1, x2, y1, y2, min_patch_span, max_px, max_py)
            patch_high = [yy * (W // patch_size) + xx for yy in range(y1, y2 + 1) for xx in range(x1, x2 + 1)]

            scale_h, scale_w = H / Hf, W / Wf
            cx_f, cy_f = cx_img / scale_w, cy_img / scale_h
            bw_f, bh_f = bw_img / scale_w, bh_img / scale_h
            xf1, xf2 = int((cx_f - bw_f / 2) // patch_size), int((cx_f + bw_f / 2) // patch_size)
            yf1, yf2 = int((cy_f - bh_f / 2) // patch_size), int((cy_f + bh_f / 2) // patch_size)
            max_fx, max_fy = Wf // patch_size - 1, Hf // patch_size - 1
            xf1, xf2, yf1, yf2 = _expand_patch_range(xf1, xf2, yf1, yf2, min_patch_span, max_fx, max_fy)
            patch_low = [yy * (Wf // patch_size) + xx for yy in range(yf1, yf2 + 1) for xx in range(xf1, xf2 + 1)]

            patch_groups_high.append((cls_id, patch_high))
            patch_groups_low.append((cls_id, patch_low))

        for i in range(num_objs):
            for j in range(num_objs):
                if patch_groups_high[i][0] == patch_groups_low[j][0]:
                    for p1 in patch_groups_high[i][1]:
                        for p2 in patch_groups_low[j][1]:
                            if 0 <= p1 < n_high and 0 <= p2 < n_low:
                                golden_map[b, 0, p1, p2] = 1.0
    return golden_map


class CBAM_Channel_Att(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_attn(x)


class CrossWindowAttention(nn.Module):
    def __init__(self, dim: int, window_sizes=None, mode: str = "high"):
        super().__init__()
        window_sizes = window_sizes or [(1, 7), (7, 1), (7, 7)]
        self.dim = dim
        self.window_sizes = window_sizes
        self.mode = mode
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        if self.mode == "high":
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(len(window_sizes) * dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x_src: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_src.shape
        outputs = []
        sizes = self.window_sizes if self.mode == "high" else [self.window_sizes[0]]
        for Wh, Ww in sizes:
            pad_h = (Wh - H % Wh) % Wh
            pad_w = (Ww - W % Ww) % Ww
            src = F.pad(x_src, (0, pad_w, 0, pad_h))
            tgt = F.pad(x_tgt, (0, pad_w, 0, pad_h))
            Hp, Wp = src.shape[2], src.shape[3]

            def window_partition(x, wh, ww):
                x = x.view(B, C, Hp // wh, wh, Wp // ww, ww)
                x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
                return x.view(-1, wh * ww, C)

            def window_reverse(windows, wh, ww, h, w):
                bn, _, c = windows.shape
                b = bn // ((h // wh) * (w // ww))
                x = windows.view(b, h // wh, w // ww, wh, ww, c)
                x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
                return x.view(b, c, h, w)

            src_w = window_partition(src, Wh, Ww)
            tgt_w = window_partition(tgt, Wh, Ww)
            q, k, v = self.q_proj(src_w), self.k_proj(tgt_w), self.v_proj(tgt_w)
            attn = F.softmax((q @ k.transpose(-2, -1)) / (C**0.5), dim=-1)
            out = self.out_proj(attn @ v)
            out = window_reverse(out, Wh, Ww, Hp, Wp)[:, :, :H, :W]
            outputs.append(out)

        if self.mode == "high":
            return self.fuse_conv(torch.cat(outputs, dim=1))
        return outputs[0]


class ACFEE(nn.Module):
    """Adaptive cross-frequency enhancement (FFT high/low + cross-window attention)."""

    def __init__(
        self,
        channel: int,
        cutoff_frequency: float = 0.01,
        use_attention: bool = True,
        attention_type: str = "cross",
    ) -> None:
        super().__init__()
        self.cutoff_frequency = cutoff_frequency
        self.use_attention = use_attention
        self.attention_type = attention_type
        if use_attention:
            self.channel_Att = CBAM_Channel_Att(channel, reduction=4)
            if attention_type == "cross":
                self.cross_attn_low2high = CrossWindowAttention(
                    channel, window_sizes=[(1, 7), (7, 1), (7, 7)], mode="high"
                )
                self.cross_attn_high2low = CrossWindowAttention(channel, window_sizes=[(11, 11)], mode="low")
            else:
                self.self_attn = _SelfAttnBlock(2 * channel, channel)
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x_freq = torch.fft.fft2(x32)
        x_freq_shifted = torch.fft.fftshift(x_freq)
        h, w = x.shape[2], x.shape[3]
        cy, cx = h // 2, w // 2
        y, x_grid = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing="ij",
        )
        freq_map = torch.sqrt((x_grid - cx).float() ** 2 + (y - cy).float() ** 2)
        cutoff = self.cutoff_frequency * max(h, w)
        high_pass_filter = (freq_map > cutoff).float()
        low_pass_filter = 1 - high_pass_filter

        x_freq_high = x_freq_shifted * high_pass_filter
        x_freq_low = x_freq_shifted * low_pass_filter
        x_high = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_freq_high))).to(dtype=dtype)
        x_low = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_freq_low))).to(dtype=dtype)

        if not self.use_attention:
            return x_high

        x_high = self.channel_Att(x_high)
        x_low = self.channel_Att(x_low)
        if self.attention_type == "cross":
            enhanced_high = self.cross_attn_low2high(x_low, x_high)
            enhanced_low = self.cross_attn_high2low(x_high, x_low)
        else:
            combined = torch.cat([x_low, x_high], dim=1)
            enhanced = self.self_attn(combined, combined)
            enhanced_low, enhanced_high = torch.split(enhanced, x.shape[1], dim=1)
        combined_features = torch.cat([enhanced_low, enhanced_high], dim=1)
        return self.fusion_conv(combined_features) + x


class _SelfAttnBlock(nn.Module):
    def __init__(self, in_channels: int, inner_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, inner_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, inner_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, inner_channels, 1)
        self.output_conv = nn.Conv2d(inner_channels, in_channels, 1)
        self.norm = nn.BatchNorm2d(inner_channels)
        self.scale = torch.sqrt(torch.tensor(float(inner_channels)))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = context if context is not None else x
        query, key, value = self.query_conv(x), self.key_conv(context), self.value_conv(context)
        attn_scores = torch.einsum("bchw,bcHW->bhwHW", query, key) / self.scale.to(query.device)
        attn_weights = F.softmax(attn_scores.view(*attn_scores.shape[:3], -1), dim=-1).view(attn_scores.shape)
        output = torch.einsum("bhwHW,bcHW->bchw", attn_weights, value)
        output = self.output_conv(F.relu(self.norm(output), inplace=True))
        return output + x


class CSSG(nn.Module):
    """Cross-scale semantic guidance between image patches and feature patches."""

    def __init__(self, in_channels: tuple[int, int], patch_size: int = 32) -> None:
        super().__init__()
        img_channel, feat_channel = in_channels
        self.patch_size = patch_size
        self.conv_img = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=7, padding=3),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
        self.conv_feamap = nn.Conv2d(feat_channel, feat_channel, kernel_size=1, stride=1)
        self.register_buffer("feat_scale", torch.tensor(1.0 / 16.0))
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.resolution_trans = nn.Sequential(
            nn.Linear(patch_size * patch_size, 2 * patch_size * patch_size, bias=False),
            nn.Linear(2 * patch_size * patch_size, patch_size * patch_size, bias=False),
            nn.ReLU(),
        )

    def forward(self, img: torch.Tensor, feamap: torch.Tensor) -> torch.Tensor:
        ini_img = self.conv_img(img)
        feamap = self.conv_feamap(feamap) * self.feat_scale
        attentions = []
        unfold_img = self.unfold(ini_img).transpose(-1, -2)
        unfold_img = self.resolution_trans(unfold_img)
        for i in range(feamap.shape[1]):
            unfold_feamap = self.unfold(feamap[:, i : i + 1])
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)
            att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)
            attentions.append(att.unsqueeze(1).clamp(-20.0, 20.0))
        return torch.cat(attentions, dim=1)


class SobelConv(nn.Module):
    """Fixed Sobel edge magnitude (SES-Net semantic-edge branch)."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        sobel_x = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer("kx", sobel_x.repeat(channel, 1, 1, 1))
        self.register_buffer("ky", sobel_y.repeat(channel, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_x = F.conv2d(x, self.kx, padding=1, groups=x.shape[1])
        edge_y = F.conv2d(x, self.ky, padding=1, groups=x.shape[1])
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


def _cssg_modulate(
    feat: torch.Tensor,
    attentions: torch.Tensor,
    patch_size: int,
    alpha: float,
) -> torch.Tensor:
    """Apply CSSG patch attention; fold at feature resolution (img should be aligned to feat size)."""
    feat_save = feat
    B, C, H, W = feat.shape
    patch_area = patch_size * patch_size
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size).to(feat.device)
    fold = nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size).to(feat.device)
    x_unfold = unfold(feat).view(B, C, patch_area, -1)
    modulated = []
    for i in range(C):
        att_logits = attentions[:, i, :, :].float().clamp(-20.0, 20.0)
        att = F.softmax(att_logits, dim=-1).to(dtype=feat.dtype)
        f_i = x_unfold[:, i, :, :]
        mod = torch.matmul(att, f_i.transpose(-1, -2)).transpose(-1, -2).contiguous().view(B, -1, att.shape[1])
        modulated.append(fold(mod))
    feat_mod = torch.cat(modulated, dim=1)
    out = alpha * feat_mod + (1 - alpha) * feat_save
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class ACFEEBlock(nn.Module):
    """YOLO yaml block: single-feature ACFEE with residual."""

    def __init__(self, c1: int, cutoff: float = 0.01):
        super().__init__()
        self.acfee = ACFEE(c1, cutoff_frequency=cutoff, use_attention=True, attention_type="cross")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.acfee(x)


class SESInputCache(nn.Module):
    """Pass-through layer that caches raw input for SESFusion (use as backbone layer 0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SESFusion(nn.Module):
    """CSSG patch modulation + ACFEE frequency enhancement on a neck feature map.

    YAML: SESFusion, [patch_size, cutoff, img_ch] with f=[img_layer, feat_layer].
    """

    def __init__(self, c1: int, patch_size: int = 32, cutoff: float = 0.01, img_ch: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.cssg = CSSG((img_ch, c1), patch_size=patch_size)
        self.acfee = ACFEE(c1, cutoff_frequency=cutoff, use_attention=True, attention_type="cross")
        self.cssg_loss: torch.Tensor | None = None
        self._targets = None
        self.register_buffer("alpha", torch.tensor(0.5))

    def set_targets(self, targets: dict | None) -> None:
        self._targets = targets

    def forward(self, x: list | torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            img, feat = x[0], x[1]
        else:
            feat = x
            img = None
        if img is None:
            return self.acfee(feat)

        img_aligned = F.interpolate(img, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        attentions = self.cssg(img_aligned, feat)
        self.cssg_loss = None

        if self._targets is not None and self.training:
            B = img.shape[0]
            with torch.no_grad():
                golden = build_golden_attention_from_bbox_by_class(
                    self._targets,
                    img_shape=feat.shape[-2:],
                    feat_shape=feat.shape[-2:],
                    patch_size=self.patch_size,
                    device=img.device,
                    B=B,
                )
            pred_attn = attentions.mean(dim=1, keepdim=True)
            if golden.shape == pred_attn.shape:
                loss = F.mse_loss(pred_attn, golden)
                self.cssg_loss = loss if torch.isfinite(loss) else None

        feat = _cssg_modulate(feat, attentions, self.patch_size, float(self.alpha))
        return self.acfee(feat)


class SmallDefectEnhance(nn.Module):
    """SES edge-semantic (CSSG + Sobel) + DFIR FIRC3Lite on P2; no input preprocess required.

    YAML: SmallDefectEnhance, [patch_size, firc_steps] with f=[input_layer, feat_layer].
    Image is resized to feature map resolution before CSSG to save memory at imgsz=1024.
    """

    def __init__(self, c1: int, patch_size: int = 16, firc_steps: int = 1, img_ch: int = 3, edge_w: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.cssg = CSSG((img_ch, c1), patch_size=patch_size)
        self.sobel = SobelConv(c1)
        self.firc = FIRC3Lite(c1, c1, n=firc_steps, e=0.5)
        self.cssg_loss: torch.Tensor | None = None
        self._targets = None
        self.register_buffer("alpha", torch.tensor(0.25))
        self.register_buffer("edge_w", torch.tensor(edge_w))

    def set_targets(self, targets: dict | None) -> None:
        self._targets = targets

    def forward(self, x: list | torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            img, feat = x[0], x[1]
        else:
            return self.firc(x)

        img_aligned = F.interpolate(img, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        attentions = self.cssg(img_aligned, feat)
        self.cssg_loss = None
        if self._targets is not None and self.training:
            with torch.no_grad():
                golden = build_golden_attention_from_bbox_by_class(
                    self._targets,
                    img_shape=feat.shape[-2:],
                    feat_shape=feat.shape[-2:],
                    patch_size=self.patch_size,
                    device=img.device,
                    B=img.shape[0],
                    min_patch_span=1,
                )
            pred_attn = attentions.mean(dim=1, keepdim=True)
            if golden.shape == pred_attn.shape:
                loss = F.mse_loss(pred_attn, golden)
                self.cssg_loss = loss if torch.isfinite(loss) else None

        feat = _cssg_modulate(feat, attentions, self.patch_size, float(self.alpha))
        edge = self.sobel(feat)
        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        feat = feat + float(self.edge_w) * edge
        feat_pre_firc = feat
        out = self.firc(feat)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if getattr(self, "capture_freq_viz", False):
            attn_map = attentions.mean(dim=1, keepdim=True)
            if attn_map.shape[-2:] != feat.shape[-2:]:
                attn_map = F.interpolate(attn_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            self._freq_viz = {
                "cssg_attn": attn_map[:1].detach(),
                "sobel_edge": edge[:1].detach(),
                "feat_pre_firc": feat_pre_firc[:1].detach(),
                "feat_out": out[:1].detach(),
            }
        return out


def set_ses_targets(model: nn.Module, batch: dict) -> None:
    """Attach batch targets to SESFusion / SmallDefectEnhance modules before forward."""
    targets = {
        "cls": batch["cls"],
        "bboxes": batch["bboxes"],
        "batch_idx": batch["batch_idx"],
    }
    for m in model.modules():
        if isinstance(m, (SESFusion, SmallDefectEnhance)):
            m.set_targets(targets)


def collect_ses_aux_loss(model: nn.Module) -> torch.Tensor | None:
    """Sum CSSG auxiliary losses from SES-style neck modules."""
    total = None
    for m in model.modules():
        if isinstance(m, (SESFusion, SmallDefectEnhance)) and m.cssg_loss is not None:
            total = m.cssg_loss if total is None else total + m.cssg_loss
            m.cssg_loss = None
    return total
