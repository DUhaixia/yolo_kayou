# Ultralytics — frequency-module visualization during training (FIRC / FGU / SES neck).

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.modules.dfir_net import FGU, FIRCLite
from ultralytics.nn.modules.ses_net import SmallDefectEnhance
from ultralytics.utils import LOGGER, RANK


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def set_freq_capture(model: nn.Module, enabled: bool) -> None:
    """Toggle capture of frequency / SES debug tensors on custom neck modules."""
    for m in _unwrap(model).modules():
        if isinstance(m, (FIRCLite, FGU, SmallDefectEnhance)):
            m.capture_freq_viz = enabled


def clear_freq_viz_cache(model: nn.Module) -> None:
    """Drop cached viz tensors after panels are saved."""
    for m in _unwrap(model).modules():
        if isinstance(m, (FIRCLite, FGU, SmallDefectEnhance)):
            m._freq_viz = {}


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reduce BCHW / CHW tensors to HW for heatmap display."""
    t = t.detach().float()
    if t.ndim == 4:
        return t[0].mean(0)
    if t.ndim == 3:
        return t.mean(0)
    return t


def _spatial_map(t: torch.Tensor) -> np.ndarray:
    """Channel-mean spatial map, normalized to [0, 1]."""
    t = _to_2d(t)
    t = t.detach().cpu().numpy()
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(t, 2), np.percentile(t, 98)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((t - lo) / (hi - lo), 0.0, 1.0)


def _log_mag_map(t: torch.Tensor) -> np.ndarray:
    """Log-scaled channel-mean magnitude spectrum (fftshift for display)."""
    t = torch.fft.fftshift(_to_2d(t), dim=(-2, -1))
    t = torch.log1p(t).detach().cpu().numpy()
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(t, 2), np.percentile(t, 98)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((t - lo) / (hi - lo), 0.0, 1.0)


def _save_panel(images: list[tuple[str, np.ndarray]], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    fig.suptitle(title, fontsize=11)
    axes = np.array(axes).reshape(-1)
    for ax, (name, arr) in zip(axes, images):
        ax.imshow(arr, cmap="magma")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_firc_viz(viz: dict, path: Path, title: str = "FIRCLite") -> None:
    mag_in, mag_out = viz.get("mag_in"), viz.get("mag_out")
    panels = [("spatial_in", _spatial_map(viz["spatial_in"]))]
    if mag_in is not None:
        panels.append(("log_mag_in", _log_mag_map(mag_in)))
    if mag_out is not None:
        panels.append(("log_mag_refined", _log_mag_map(mag_out)))
    if mag_in is not None and mag_out is not None:
        delta = (_to_2d(mag_out) - _to_2d(mag_in)).cpu().numpy()
        delta = np.nan_to_num(delta, nan=0.0)
        lo, hi = np.percentile(delta, 2), np.percentile(delta, 98)
        if hi <= lo:
            hi = lo + 1e-6
        panels.append(("mag_delta", np.clip((delta - lo) / (hi - lo), 0.0, 1.0)))
    if "spatial_residual" in viz:
        panels.append(("spatial_residual", _spatial_map(viz["spatial_residual"])))
    panels.append(("spatial_out", _spatial_map(viz["spatial_out"])))
    _save_panel(panels, title, path)


def save_fgu_viz(viz: dict, path: Path, title: str = "FGU") -> None:
    panels = [
        ("spatial_in", _spatial_map(viz["spatial_in"])),
        ("high_freq", _spatial_map(viz["high_freq"])),
        ("gate", _spatial_map(viz["gate"])),
        ("spatial_out", _spatial_map(viz["spatial_out"])),
    ]
    _save_panel(panels, title, path)


def save_ses_neck_viz(viz: dict, path: Path, title: str = "SmallDefectEnhance") -> None:
    panels = [("cssg_attn", _spatial_map(viz["cssg_attn"])), ("sobel_edge", _spatial_map(viz["sobel_edge"]))]
    if "feat_pre_firc" in viz:
        panels.append(("feat_pre_firc", _spatial_map(viz["feat_pre_firc"])))
    panels.append(("feat_out", _spatial_map(viz["feat_out"])))
    _save_panel(panels, title, path)


def _dump_module_viz(model: nn.Module, out_dir: Path, prefix: str) -> int:
    n = 0
    for name, m in model.named_modules():
        if not hasattr(m, "_freq_viz") or not m._freq_viz:
            continue
        tag = name.replace(".", "_") if name else prefix
        viz = m._freq_viz
        if isinstance(m, FIRCLite):
            save_firc_viz(viz, out_dir / f"{tag}_firc.png", title=f"FIRCLite ({name})")
            n += 1
        elif isinstance(m, FGU):
            save_fgu_viz(viz, out_dir / f"{tag}_fgu.png", title=f"FGU ({name})")
            n += 1
        elif isinstance(m, SmallDefectEnhance):
            save_ses_neck_viz(viz, out_dir / f"{tag}_ses.png", title=f"SmallDefectEnhance ({name})")
            n += 1
    return n


def run_freq_training_viz(
    trainer,
    every_n: int = 5,
    sample_count: int = 2,
    epoch: int | None = None,
) -> None:
    """Run one forward on a train batch and save frequency / SES neck visualizations."""
    if RANK not in {-1, 0}:
        return
    epoch = trainer.epoch if epoch is None else epoch
    if every_n <= 0 or epoch % every_n != 0:
        return

    model = _unwrap(trainer.model)
    core = getattr(model, "model", model)  # DetectionModel stores layers on .model
    if not any(isinstance(m, (FIRCLite, FGU, SmallDefectEnhance)) for m in core.modules()):
        return

    save_dir = Path(trainer.save_dir) / "freq_viz" / f"epoch_{epoch:04d}"
    loader = trainer.train_loader
    if loader is None or len(loader) == 0:
        return

    batch = trainer.preprocess_batch(next(iter(loader)))
    n = max(1, min(sample_count, batch["img"].shape[0]))
    imgs = batch["img"][:n].to(trainer.device, non_blocking=True)

    was_training = model.training
    set_freq_capture(model, True)
    model.eval()
    try:
        with torch.no_grad():
            if hasattr(model, "predict"):
                model.predict(imgs)
            else:
                model(imgs)
        count = _dump_module_viz(core, save_dir, "root")
    finally:
        set_freq_capture(model, False)
        clear_freq_viz_cache(model)
        model.train(was_training)
    if count:
        LOGGER.info(f"Saved {count} frequency/SES viz panel(s) -> {save_dir}")
    else:
        LOGGER.warning(f"freq_viz: no tensors captured (epoch {epoch}); check model has FIRCLite/FGU/SES neck")
