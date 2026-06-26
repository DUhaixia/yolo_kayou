import numpy as np
import torch


def get_sliding_coords(img_h, img_w, patch_size, stride):
    """Return top-left (y, x) coords that cover the full image."""
    if img_h < patch_size or img_w < patch_size:
        raise ValueError(
            f'Image ({img_h}x{img_w}) is smaller than patch_size ({patch_size}).'
        )

    ys = list(range(0, img_h - patch_size + 1, stride))
    xs = list(range(0, img_w - patch_size + 1, stride))

    if not ys or ys[-1] + patch_size < img_h:
        ys.append(img_h - patch_size)
    if not xs or xs[-1] + patch_size < img_w:
        xs.append(img_w - patch_size)

    ys = sorted(set(ys))
    xs = sorted(set(xs))
    return [(y, x) for y in ys for x in xs]


def make_blend_weight(patch_size, mode='gaussian'):
    """Per-patch weight map used to fuse overlapping predictions."""
    if mode == 'uniform':
        return np.ones((patch_size, patch_size), dtype=np.float32)

    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    cy = (patch_size - 1) / 2.0
    cx = (patch_size - 1) / 2.0
    sigma = patch_size / 4.0
    weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    weight = weight.astype(np.float32)
    weight = weight / weight.max()
    return weight


def sliding_window_infer(
    net, img, patch_size=512, stride=256, blend='gaussian', device='cuda', batch_size=1,
):
    """
    Run sliding-window inference on a single grayscale image.

    Args:
        net: model in eval mode, expects input [B, 1, H, W] in [0, 1].
        img: np.ndarray [H, W], float32 in [0, 1].
        patch_size: window size, default 512.
        stride: step between windows.
        blend: 'gaussian' or 'uniform'.

    Returns:
        pred_map: np.ndarray [H, W], probability map in [0, 1].
    """
    assert img.ndim == 2, 'img must be grayscale [H, W]'
    h, w = img.shape
    coords = get_sliding_coords(h, w, patch_size, stride)

    pred_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)
    patch_weight = make_blend_weight(patch_size, mode=blend)

    batch_size = max(1, int(batch_size))
    batch_tensors = []
    batch_coords = []

    def _flush():
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0)
        out = net.forward(batch)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        preds = out.detach().cpu().numpy()
        for (y, x), patch_pred in zip(batch_coords, preds[:, 0]):
            pred_acc[y:y + patch_size, x:x + patch_size] += patch_pred * patch_weight
            weight_acc[y:y + patch_size, x:x + patch_size] += patch_weight
        batch_tensors.clear()
        batch_coords.clear()

    with torch.no_grad():
        for y, x in coords:
            patch = img[y:y + patch_size, x:x + patch_size]
            tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, :, :]).float().to(device)
            batch_tensors.append(tensor)
            batch_coords.append((y, x))
            if len(batch_tensors) >= batch_size:
                _flush()
        _flush()

    weight_acc = np.maximum(weight_acc, 1e-8)
    return pred_acc / weight_acc


def count_windows(img_h, img_w, patch_size=512, stride=256):
    """Utility: how many forward passes are needed."""
    return len(get_sliding_coords(img_h, img_w, patch_size, stride))
