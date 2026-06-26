"""
C++-style dynamic sliding-window deployment pipeline.

Architecture (mirrors typical C++ industrial vision code):

    SlidingWindowPlanner  -> enumerate (y, x) patch coords for any image size
    PatchPreprocessor     -> per-patch preprocess (normalize / defect enhance)
    ModelRunner           -> single-patch or batched forward
    FusionAccumulator     -> weighted overlap blend (gaussian / uniform)
    DefectPostProcessor   -> threshold, connected components, filter, draw

Usage:
    from utils.deploy_pipeline import SlidingConfig, SlidingWindowEngine

    engine = SlidingWindowEngine.from_checkpoint(
        checkpoint='./log5/IRSTD-1K/DATransNet/400.pth.tar',
        config=SlidingConfig(patch_size=512, stride=256),
        device='cuda:0',
    )
    result = engine.infer_file('image.bmp')
    result.save('./output')
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from net import Net
from utils.defect_preprocess import apply_defect_preprocess
from utils.sliding_window import get_sliding_coords, make_blend_weight


@dataclass
class SlidingConfig:
    """All tunable deployment parameters in one struct (like a C++ config)."""
    patch_size: int = 512
    stride: int = 256
    blend: str = 'gaussian'          # gaussian | uniform
    threshold: float = 0.5
    img_size: int = 512              # model training size, must match patch_size
    model_name: str = 'DATransNet'
    preprocess_mode: str = 'none'     # none | point | line | mixed
    input_channels: int = 1          # 1=gray, 3=BGR (requires retrained model)
    batch_size: int = 4
    edge_margin: int = 0             # ignore defects within margin px of border
    min_area: int = 0                # 0 = no filter
    max_area: int = 0                # 0 = no filter
    max_aspect: float = 0.0          # 0 = no filter; max w/h or h/w for blobs


@dataclass
class DefectInfo:
    """Single connected-component defect."""
    id: int
    cx: float
    cy: float
    area: float
    bbox: Tuple[int, int, int, int]   # x0, y0, x1, y1
    aspect: float
    mean_prob: float


@dataclass
class InferResult:
    """Per-image inference result."""
    filename: str
    image_h: int
    image_w: int
    prob_map: np.ndarray              # [H, W] float32 [0, 1]
    mask: np.ndarray                  # [H, W] uint8 0/255
    defects: List[DefectInfo]
    num_windows: int
    elapsed_sec: float

    def save(self, output_dir: str, save_vis: bool = True, src_bgr: Optional[np.ndarray] = None):
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(self.filename)[0]

        prob_png = (self.prob_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, stem + '_prob.png'), prob_png)
        cv2.imwrite(os.path.join(output_dir, stem + '_mask.png'), self.mask)

        if save_vis:
            vis = self._draw_vis(src_bgr)
            cv2.imwrite(os.path.join(output_dir, stem + '_vis.png'), vis)

    def _draw_vis(self, src_bgr: Optional[np.ndarray]) -> np.ndarray:
        if src_bgr is not None and src_bgr.ndim == 3:
            vis = src_bgr.copy()
        else:
            vis = cv2.cvtColor(
                (self.prob_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )

        overlay = vis.copy()
        overlay[self.mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

        n_point = sum(1 for d in self.defects if d.aspect <= 2.0)
        n_line = len(self.defects) - n_point
        title = f'{self.filename}  windows={self.num_windows}  defects={len(self.defects)}  point~{n_point}  line~{n_line}'
        cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        for d in self.defects:
            x0, y0, x1, y1 = d.bbox
            color = (0, 255, 255) if d.aspect <= 2.0 else (255, 128, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
            label = f'#{d.id} {d.area:.0f}px'
            cv2.putText(vis, label, (x0, max(y0 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return vis


class SlidingWindowPlanner:
    """Generate patch top-left coords for arbitrary image size."""

    @staticmethod
    def plan(h: int, w: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
        return get_sliding_coords(h, w, patch_size, stride)

    @staticmethod
    def count(h: int, w: int, patch_size: int, stride: int) -> int:
        return len(get_sliding_coords(h, w, patch_size, stride))


class PatchPreprocessor:
    """
    Per-patch preprocessing (C++ style: process one ROI at a time).

    Pipeline:
        raw patch (uint8 gray or BGR)
            -> optional defect_preprocess (point/line/mixed)
            -> normalize to [0, 1]
            -> layout [C, H, W]
    """

    def __init__(self, mode: str = 'none', input_channels: int = 1):
        self.mode = (mode or 'none').lower()
        self.input_channels = input_channels

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 2:
            bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        else:
            bgr = patch

        if self.mode != 'none':
            bgr = apply_defect_preprocess(bgr, self.mode)

        if self.input_channels == 1:
            if bgr.ndim == 3:
                ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                ch = bgr
            out = ch.astype(np.float32) / 255.0
            return out[np.newaxis, :, :]

        out = bgr.astype(np.float32) / 255.0
        return np.transpose(out, (2, 0, 1))  # HWC -> CHW


class FusionAccumulator:
    """Weighted overlap fusion (post-process stage 1)."""

    def __init__(self, h: int, w: int, patch_size: int, blend: str = 'gaussian'):
        self.h = h
        self.w = w
        self.ps = patch_size
        self.pred_acc = np.zeros((h, w), dtype=np.float32)
        self.weight_acc = np.zeros((h, w), dtype=np.float32)
        self.patch_weight = make_blend_weight(patch_size, mode=blend)

    def accumulate(self, y: int, x: int, patch_pred: np.ndarray):
        self.pred_acc[y:y + self.ps, x:x + self.ps] += patch_pred * self.patch_weight
        self.weight_acc[y:y + self.ps, x:x + self.ps] += self.patch_weight

    def finalize(self) -> np.ndarray:
        w = np.maximum(self.weight_acc, 1e-8)
        return self.pred_acc / w


class DefectPostProcessor:
    """Threshold + connected components + geometric filter + draw helpers."""

    def __init__(self, config: SlidingConfig):
        self.cfg = config

    def prob_to_mask(self, prob: np.ndarray) -> np.ndarray:
        return ((prob > self.cfg.threshold) * 255).astype(np.uint8)

    def extract_defects(self, mask: np.ndarray, prob: np.ndarray) -> List[DefectInfo]:
        h, w = mask.shape
        binary = (mask > 0).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        defects: List[DefectInfo] = []
        did = 0
        margin = self.cfg.edge_margin

        for i in range(1, n_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            y0 = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            x1, y1 = x0 + bw, y0 + bh
            cx, cy = centroids[i]

            if self.cfg.min_area > 0 and area < self.cfg.min_area:
                continue
            if self.cfg.max_area > 0 and area > self.cfg.max_area:
                continue

            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if self.cfg.max_aspect > 0 and aspect > self.cfg.max_aspect:
                continue

            if margin > 0:
                if x0 < margin or y0 < margin or x1 > w - margin or y1 > h - margin:
                    continue

            region = labels == i
            mean_prob = float(prob[region].mean()) if region.any() else 0.0

            did += 1
            defects.append(DefectInfo(
                id=did, cx=float(cx), cy=float(cy), area=area,
                bbox=(x0, y0, x1, y1), aspect=aspect, mean_prob=mean_prob,
            ))
        return defects


class ModelRunner:
    """Thin wrapper around PyTorch model forward."""

    def __init__(self, net: torch.nn.Module, device: str, batch_size: int = 4):
        self.net = net
        self.device = device
        self.batch_size = max(1, batch_size)

    @torch.no_grad()
    def forward_batch(self, tensors: Sequence[torch.Tensor]) -> List[np.ndarray]:
        batch = torch.cat(list(tensors), dim=0)
        out = self.net.forward(batch)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        preds = out.detach().cpu().numpy()
        return [preds[i, 0] for i in range(preds.shape[0])]


class SlidingWindowEngine:
    """
    Main deployment engine.

    Flow per image:
        1. load image
        2. plan sliding coords (dynamic for any HxW)
        3. for each patch (batched): preprocess -> infer -> accumulate
        4. fuse -> threshold -> extract defects -> return result
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: SlidingConfig,
        device: str = 'cuda:0',
    ):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device).eval()
        self.preprocessor = PatchPreprocessor(config.preprocess_mode, config.input_channels)
        self.postprocessor = DefectPostProcessor(config)
        self.runner = ModelRunner(self.model, self.device, config.batch_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        config: Optional[SlidingConfig] = None,
        device: str = 'cuda:0',
    ) -> 'SlidingWindowEngine':
        cfg = config or SlidingConfig()
        net = Net(model_name=cfg.model_name, mode='test', size=cfg.img_size)
        ckpt = torch.load(checkpoint, map_location=device)
        net.load_state_dict(ckpt['state_dict'], strict=False)
        return cls(net, cfg, device)

    def _load_image(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (gray_for_model_context, bgr_for_vis)."""
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        if raw.ndim == 2:
            gray = raw
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.shape[2] == 4:
            bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            bgr = raw
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray, bgr

    def infer_array(
        self,
        image: np.ndarray,
        filename: str = 'image',
        bgr_vis: Optional[np.ndarray] = None,
    ) -> InferResult:
        """Run sliding-window inference on a numpy image (gray or BGR)."""
        cfg = self.config
        ps = cfg.patch_size

        if image.ndim == 2:
            gray = image
            bgr_vis = bgr_vis or cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr_vis = bgr_vis or image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        coords = SlidingWindowPlanner.plan(h, w, ps, cfg.stride)
        fusion = FusionAccumulator(h, w, ps, cfg.blend)

        t0 = time.time()
        batch_tensors: List[torch.Tensor] = []
        batch_meta: List[Tuple[int, int]] = []

        with torch.no_grad():
            for y, x in coords:
                patch = gray[y:y + ps, x:x + ps] if cfg.preprocess_mode == 'none' else bgr_vis[y:y + ps, x:x + ps]
                chw = self.preprocessor(patch)
                tensor = torch.from_numpy(chw).float().unsqueeze(0).to(self.device)
                batch_tensors.append(tensor)
                batch_meta.append((y, x))

                if len(batch_tensors) >= cfg.batch_size:
                    self._flush_batch(batch_tensors, batch_meta, fusion)
                    batch_tensors, batch_meta = [], []

            if batch_tensors:
                self._flush_batch(batch_tensors, batch_meta, fusion)

        prob = fusion.finalize()
        mask = self.postprocessor.prob_to_mask(prob)
        defects = self.postprocessor.extract_defects(mask, prob)
        elapsed = time.time() - t0

        return InferResult(
            filename=filename,
            image_h=h, image_w=w,
            prob_map=prob, mask=mask,
            defects=defects,
            num_windows=len(coords),
            elapsed_sec=elapsed,
        )

    def _flush_batch(
        self,
        tensors: List[torch.Tensor],
        meta: List[Tuple[int, int]],
        fusion: FusionAccumulator,
    ):
        preds = self.runner.forward_batch(tensors)
        for (y, x), pred in zip(meta, preds):
            fusion.accumulate(y, x, pred)

    def infer_file(self, path: str) -> InferResult:
        gray, bgr = self._load_image(path)
        return self.infer_array(gray, filename=os.path.basename(path), bgr_vis=bgr)

    def infer_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
        save_vis: bool = True,
    ) -> List[InferResult]:
        os.makedirs(output_dir, exist_ok=True)
        names = sorted(
            f for f in os.listdir(input_dir)
            if f.lower().endswith(extensions)
        )
        if not names:
            raise FileNotFoundError(f'No images in {input_dir}')

        results: List[InferResult] = []
        for fname in names:
            src = os.path.join(input_dir, fname)
            _, bgr = self._load_image(src)
            result = self.infer_file(src)
            result.save(output_dir, save_vis=save_vis, src_bgr=bgr)
            results.append(result)
            print(f'[{fname}] {result.image_h}x{result.image_w} '
                  f'windows={result.num_windows} defects={len(result.defects)} '
                  f'time={result.elapsed_sec:.2f}s')

        self._write_summary(results, output_dir)
        return results

    @staticmethod
    def _write_summary(results: List[InferResult], output_dir: str):
        path = os.path.join(output_dir, 'summary.csv')
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'width', 'height', 'windows', 'defects',
                'point_like', 'line_like', 'time_sec',
            ])
            for r in results:
                n_line = sum(1 for d in r.defects if d.aspect > 2.0)
                n_point = len(r.defects) - n_line
                writer.writerow([
                    r.filename, r.image_w, r.image_h, r.num_windows,
                    len(r.defects), n_point, n_line, f'{r.elapsed_sec:.3f}',
                ])
